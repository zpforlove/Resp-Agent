import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchdiffeq import odeint

from utils import timestep_embedding

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Configure logging level to INFO


class MelSpectrogramExtractor:
    """Mel Spectrogram Extractor - Configured to match Vocos vocoder requirements"""

    def __init__(self, config, target_device=None):
        """
        Args:
            config (dict): A dictionary containing audio and vocoder configurations.
            target_device (str, optional): The desired computation device (e.g., "cuda:0", "cpu").
        """
        self.config = config
        self.vocos_config = config['vocos']  # Specific configuration for the Vocos vocoder

        # Mel-spectrogram parameters (obtained from vocos config to ensure compatibility)
        self.mel_config = self.vocos_config['mel']
        self.n_fft = self.mel_config['n_fft']  # FFT window size
        self.hop_length = self.mel_config['hop_length']  # Hop length
        self.win_length = self.mel_config['win_length']  # Window length
        self.n_mels = self.mel_config['n_mels']  # Number of mel filter banks
        self.f_min = self.mel_config['f_min']  # Minimum frequency
        self.f_max = self.mel_config['f_max']  # Maximum frequency
        self.center = self.mel_config['center']  # Whether to center the signal
        self.power = self.mel_config['power']  # Power of the magnitude spectrum (e.g., 2.0 for power spectrum)

        # Resampling parameters
        self.input_sr = self.config['audio']['sample_rate']  # Sample rate of the input audio
        self.target_sr = self.vocos_config['sample_rate']  # Target sample rate expected by Vocos

        # Global statistics (for Mel-spectrogram normalization)
        self.mel_mean = self.config['hyperparameters']['flow']['mel_mean']  # Pre-calculated Mel-spectrogram mean
        self.mel_std = self.config['hyperparameters']['flow'][
            'mel_std']  # Pre-calculated Mel-spectrogram standard deviation

        # Set device
        if target_device:
            self.device = torch.device(target_device)
        else:
            logger.warning("MelSpectrogramExtractor: target_device not provided. Falling back to CPU.")
            self.device = torch.device('cpu')
        logger.info(f"MelSpectrogramExtractor instance will use device: {self.device}")

        # Initialize torchaudio's MelSpectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,  # Target sample rate
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            power=self.power,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        ).to(self.device)  # Move the transform to the target device

    def __call__(self, waveform, normalize=True):
        """
        Extracts a Mel-spectrogram from an input waveform.
        Args:
            waveform (torch.Tensor): Input audio waveform, shape [B, T_wave] (batch size, waveform length).
            normalize (bool): Whether to normalize the Mel-spectrogram using global statistics. Defaults to True.
        Returns:
            torch.Tensor: The extracted Mel-spectrogram, shape [B, n_mels, T_mel] (batch size, number of mel bands, number of mel frames).
        """
        # Ensure input is on the correct device
        waveform = waveform.to(self.device)
        batch_size = waveform.shape[0]

        # If the input sample rate is different from the target sample rate, resample
        if self.input_sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=self.input_sr,  # Original sample rate
                new_freq=self.target_sr,  # Target sample rate
                # Get resampling parameters from vocos config
                lowpass_filter_width=self.vocos_config['resample']['lowpass_filter_width'],
                rolloff=self.vocos_config['resample']['rolloff'],
                resampling_method=self.vocos_config['resample']['method']
            ).to(self.device)  # Ensure the resampled result is on the target device

        # Extract Mel-spectrogram (process sample by sample to avoid potential batch issues or to simplify logic)
        mel_specs = []
        for i in range(batch_size):
            wav = waveform[i].to(self.device)  # Get a single waveform
            mel = self.mel_transform(wav).to(self.device)  # Calculate Mel-spectrogram
            # Take the logarithm of the mel-spectrogram and clamp to avoid log(0)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            mel_specs.append(mel)

        mel_specs = torch.stack(mel_specs)  # Stack the list of mel-spectrograms into a single batch tensor

        # If required, normalize using global statistics
        if normalize:
            mel_specs = (mel_specs - self.mel_mean) / self.mel_std

        return mel_specs.to(self.device)  # Ensure the final output is on the target device


class DiTBlock(nn.Module):
    """
    DiT Block (Diffusion Transformer Block), adapted from the DiT paper, using adaLN-Zero style modulation.
    A DiT block typically includes a self-attention layer and a feed-forward network layer, both with residual connections,
    and are modulated before input to these layers. The modulation parameters are derived from a time embedding.
    """

    def __init__(self, dim, heads=16, ff_mult=4, dropout=0.2):
        """
        Args:
            dim (int): The feature dimension for input and output.
            heads (int, optional): The number of heads in multi-head attention. Defaults to 16.
            ff_mult (int, optional): The multiplier for the hidden dimension of the feed-forward network relative to the input dimension. Defaults to 4.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()
        assert dim % heads == 0, f"Dimension ({dim}) must be divisible by the number of heads ({heads})"
        # The eps for LayerNorm is typically a small value to increase numerical stability, e.g., 1e-5 or 1e-6
        # elementwise_affine=False means no learnable affine parameters (gamma, beta) are used, as these will be provided by time modulation
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)  # Normalization before the attention layer
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False,
                                  eps=1e-6)  # Normalization before the feed-forward network
        # Multi-head self-attention layer
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

        ff_hidden_dim = int(dim * ff_mult)  # Hidden dimension of the feed-forward network
        # Feed-forward network (typically two linear layers with an activation function in between)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),  # First linear layer
            nn.GELU(),  # GELU is a common activation function in Transformers
            nn.Dropout(dropout),  # Dropout
            nn.Linear(ff_hidden_dim, dim),  # Second linear layer (output dimension back to dim)
            nn.Dropout(dropout)  # Dropout
        )
        # Time Modulation Network (Time MLP)
        # Used to generate modulation parameters (scale, shift, gate, etc.) from the time embedding
        # adaLN-Zero requires 6 modulation parameters (gamma1_attn, beta1_attn, gamma2_ffn, beta2_ffn, alpha_scale_attn, alpha_scale_ffn)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),  # SiLU (Swish) activation function, applied to the input time embedding
            nn.Linear(dim, 6 * dim)  # Linear layer maps the time embedding to 6 times the dimension
        )
        self._init_weights()  # Call the weight initialization method

    def _init_weights(self):
        """Initializes the weights of specific layers to help stabilize training. In the DiT paper, the last linear layer in a residual path is typically initialized to zero."""

        def _init_module_weights(m, gain=1.0, zero_out_last_linear=False):
            """Helper function to initialize the weights of a single module."""
            if isinstance(m, nn.Linear):
                if zero_out_last_linear:  # For the last linear layer in a residual block, typically initialize to zero
                    nn.init.zeros_(m.weight)
                else:
                    # Use Xavier uniform initialization with a gain parameter to help maintain signal variance
                    torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:  # Biases are typically initialized to zero
                    torch.nn.init.zeros_(m.bias)

        # Initialize the attention output projection layer (output projection weights are often initialized to zero as it's part of a residual path)
        if hasattr(self.attn, 'out_proj'):
            _init_module_weights(self.attn.out_proj, zero_out_last_linear=True)

        # Initialize the linear layers of the feed-forward network
        _init_module_weights(self.ff[0])  # First linear layer in FFN (expands dimension)
        if hasattr(self.ff[3], 'weight'):  # Second linear layer in FFN (contracts back to dim, part of a residual path)
            _init_module_weights(self.ff[3], zero_out_last_linear=True)

        # Initialize the linear layer of the time modulation network
        if hasattr(self.time_mlp[1], 'weight'):  # The second element in time_mlp is a linear layer
            nn.init.normal_(self.time_mlp[1].weight, std=0.02)
            if self.time_mlp[1].bias is not None:
                nn.init.zeros_(self.time_mlp[1].bias)

    def forward(self, x, t_emb):  # x: [B, T_seq, HiddenDim], t_emb: [B, HiddenDim]
        """
        Forward pass for the DiT block, using adaLN-Zero style modulation.
        """
        # 1. Generate modulation parameters from the time embedding
        time_params = self.time_mlp(t_emb)
        gamma1_attn, beta1_attn, gamma2_ffn, beta2_ffn, alpha_scale_attn, alpha_scale_ffn = \
            [p.unsqueeze(1) for p in torch.chunk(time_params, 6, dim=1)]

        # 2. First sub-block: Self-Attention + adaLN-Zero
        normed_x_for_attention = self.norm1(x)
        modulated_x_for_attention = normed_x_for_attention * (1 + gamma1_attn) + beta1_attn
        attention_output_raw, _ = self.attn(query=modulated_x_for_attention,
                                            key=modulated_x_for_attention,
                                            value=modulated_x_for_attention)
        scaled_attention_output = alpha_scale_attn * attention_output_raw
        x_after_attention_block = x + scaled_attention_output

        # 3. Second sub-block: Feed-Forward Network + adaLN-Zero
        normed_x_for_feedforward = self.norm2(x_after_attention_block)
        modulated_x_for_feedforward = normed_x_for_feedforward * (1 + gamma2_ffn) + beta2_ffn
        feedforward_output_raw = self.ff(modulated_x_for_feedforward)
        scaled_feedforward_output = alpha_scale_ffn * feedforward_output_raw
        x_after_feedforward_block = x_after_attention_block + scaled_feedforward_output
        return x_after_feedforward_block


class FlowMatchingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        h = config['hyperparameters']['flow']
        self.n_mels = config['vocos']['mel']['n_mels']
        self.ref_mel_dim = self.n_mels
        hidden_dim = h['hidden_dim']
        self.sigma = h['sigma']
        self.cond_drop_prob = h['cond_drop_prob']
        self.n_timesteps = h['n_timesteps']
        self.cfg_scale = h['cfg_scale']
        self.token_embedding_dim = config['hyperparameters']['flow']['token_embedding_dim']
        self.beats_feature_dim = config['hyperparameters']['flow']['feature_dim']

        self.token_embedding = nn.Embedding(
            num_embeddings=config['hyperparameters']['flow']['vocab_size'],
            embedding_dim=self.token_embedding_dim,
        )
        logger.info(f"FlowMatchingModel: Token Embedding Layer initialized internally "
                    f"with vocab_size {config['hyperparameters']['flow']['vocab_size']} "
                    f"and dim {self.token_embedding_dim}.")

        # Update fused_embed_dim, now includes token embedding and timbre embedding (BEATs feature)
        fused_embed_dim = self.token_embedding_dim + self.beats_feature_dim

        # Update input projection layer to accept conditions concatenated with timbre dimensions
        self.input_proj = nn.Linear(self.n_mels + fused_embed_dim + self.ref_mel_dim, hidden_dim)
        logger.info(f"FlowMatchingModel: Input projection expects n_mels ({self.n_mels}) + "
                    f"fused_embed_dim (token + timbre) ({fused_embed_dim}) + ref_mel_dim ({self.ref_mel_dim}).")

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(dim=hidden_dim,
                     heads=h['n_heads'],
                     ff_mult=h['ff_mult'],
                     dropout=h['dropout'])
            for _ in range(h['n_layers'])
        ])
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
            nn.Linear(hidden_dim, self.n_mels)
        )
        max_len = h.get('max_pos_emb_len', 4096)
        pe = self._get_sinusoidal_embedding(max_len, hidden_dim)
        self.register_buffer('pos_emb', pe)

    def _get_sinusoidal_embedding(self, max_len, dim):
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / dim)
        )
        emb = torch.zeros(max_len, dim)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div)
        return emb

    def _predict_velocity(self, xt, t, cond_with_ref_mel):
        """
        Predicts the velocity field v(xt, t).
        xt:   [B, n_mels, T]
        t:    [B] ∈ [0,1]
        cond_with_ref_mel: [B, fused_embed_dim + ref_mel_dim, T] (condition concatenated with reference mel)
        returns: [B, n_mels, T]
        """
        B, _, T = xt.shape
        x_f = xt.permute(0, 2, 1)
        c_f = cond_with_ref_mel.permute(0, 2, 1)
        h = torch.cat([x_f, c_f], dim=-1)
        h = self.input_proj(h)
        if T > self.pos_emb.size(0):
            logger.warning(f"Sequence length {T} exceeds maximum positional embedding length {self.pos_emb.size(0)}. "
                           f"Positional embeddings will be truncated, which may cause errors. Consider increasing max_pos_emb_len.")
            h = h + self.pos_emb[:T].unsqueeze(0) if T <= self.pos_emb.size(0) else h + self.pos_emb.unsqueeze(0)
        else:
            h = h + self.pos_emb[:T].unsqueeze(0)
        time_emb_dim = h.shape[-1]
        te = timestep_embedding(t, time_emb_dim)
        te = self.time_embed(te)
        for block in self.blocks:
            h = block(h, te)
        v = self.output_layer(h)
        return v.permute(0, 2, 1)

    def forward(self, x, t, cond_embed_dict):
        """
        Training interface:
          x:                [B, n_mels, T_full_mel]  Ground truth full mel
          t:                [B]                      Random progress
          cond_embed_dict:  A dictionary containing:
                            'fused_embed': [B, fused_embed_dim, T_full_mel] fused_embed (from tokens)
                            'ref_mel_for_cond': [B, ref_mel_dim, T_full_mel] Reference mel part, padded to T_full_mel
        returns:
          loss:       MSE loss of the predicted velocity v(xt, t)
        """
        B, _, T_full_mel = x.shape
        device = x.device

        fused_embed = cond_embed_dict['fused_embed']
        ref_mel_for_cond = cond_embed_dict['ref_mel_for_cond']

        # 1) Sample-level random dropout of the entire condition (fused_embed)
        mask_fused = (torch.rand(B, device=device) > self.cond_drop_prob).float().view(B, 1, 1)
        masked_fused_embed = fused_embed * mask_fused
        # The reference mel part is not subject to random dropout and is always used as a condition
        final_cond_embed = torch.cat([masked_fused_embed, ref_mel_for_cond],
                                     dim=1)  # Concatenate along the feature dimension

        noise = torch.randn_like(x) * self.sigma
        tb = t.view(B, 1, 1)
        xt = (1 - tb) * noise + tb * x

        pred_v = self._predict_velocity(xt, t, final_cond_embed)
        target_v = x - noise
        loss = F.mse_loss(pred_v, target_v)
        return loss

    def _ode_func(self, t_scalar, x_current, cond_embed_dict_full, cfg_scale):
        B = x_current.shape[0]
        current_T_duration = x_current.shape[-1]
        device = x_current.device
        t_batch = torch.full((B,), t_scalar, device=device, dtype=torch.float32)

        fused_embed_full = cond_embed_dict_full['fused_embed']
        ref_mel_for_cond_full = cond_embed_dict_full['ref_mel_for_cond']

        # Crop or interpolate conditional embeddings to match the current sequence length x_current.shape[-1]
        if fused_embed_full.shape[-1] < current_T_duration:
            logger.warning(
                f"ODE: fused_embed_full length {fused_embed_full.shape[-1]} is shorter than x_current length {current_T_duration}.")
            fused_effective = F.pad(fused_embed_full, (0, current_T_duration - fused_embed_full.shape[-1]))
        else:
            fused_effective = fused_embed_full[:, :, :current_T_duration]

        if ref_mel_for_cond_full.shape[-1] < current_T_duration:
            logger.warning(
                f"ODE: ref_mel_for_cond_full length {ref_mel_for_cond_full.shape[-1]} is shorter than x_current length {current_T_duration}.")
            ref_mel_effective = F.pad(ref_mel_for_cond_full, (0, current_T_duration - ref_mel_for_cond_full.shape[-1]))
        else:
            ref_mel_effective = ref_mel_for_cond_full[:, :, :current_T_duration]

        # Concatenate conditions
        final_cond_effective = torch.cat([fused_effective, ref_mel_effective], dim=1)

        v_cond = self._predict_velocity(x_current, t_batch, final_cond_effective)

        if cfg_scale > 1e-8:
            # Unconditional branch (uses zero embedding for the fused_embed part, retains the reference mel part)
            null_fused_cond = torch.zeros_like(fused_effective)
            null_cond_with_ref = torch.cat([null_fused_cond, ref_mel_effective], dim=1)
            v_uncond = self._predict_velocity(x_current, t_batch, null_cond_with_ref)
            effective_velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            effective_velocity = v_cond
        return effective_velocity

    @torch.no_grad()
    def sample(self, cond_embed_dict, *, target_duration_frames=None, steps=None, cfg_scale=None, method="euler",
               sway_sampling_coef=None):
        """
        Inference interface
        cond_embed_dict: A dictionary containing:
                         'fused_embed': [B, fused_embed_dim, T_cond_full]
                         'ref_mel_for_cond': [B, ref_mel_dim, T_cond_full]
        target_duration_frames: The target number of frames for the mel-spectrogram. If None, T_cond_full is used.
        sway_sampling_coef: Optional parameter to adjust the values of t_span.
        """
        self.eval()
        device = next(self.parameters()).device
        fused_embed = cond_embed_dict['fused_embed']
        B, _, T_cond_full = fused_embed.shape

        steps_to_take = steps if steps is not None else self.n_timesteps
        current_cfg_scale = cfg_scale if cfg_scale is not None else self.cfg_scale
        max_supported_len = self.pos_emb.size(0)

        if target_duration_frames is None:
            actual_duration = T_cond_full
        else:
            actual_duration = target_duration_frames

        if actual_duration > max_supported_len:
            logger.warning(
                f"Sample: Requested duration {actual_duration} exceeds max positional embedding length {max_supported_len}. "
                f"Output will be truncated to {max_supported_len}.")
            actual_duration = max_supported_len

        x0 = torch.randn(B, self.n_mels, actual_duration, device=device) * self.sigma
        t_span = torch.linspace(0., 1., steps_to_take, device=device)

        # Adjust the t_span values according to sway_sampling_coef
        if sway_sampling_coef is not None:
            t_span = t_span + sway_sampling_coef * (torch.cos(math.pi / 2 * t_span) - 1 + t_span)

        # Pass cond_embed_dict to _ode_func
        solution_trajectory = odeint(
            lambda t_scalar, x_current: self._ode_func(t_scalar, x_current, cond_embed_dict, current_cfg_scale),
            x0,
            t_span,
            method=method
        )
        final_mel = solution_trajectory[-1]
        return final_mel
