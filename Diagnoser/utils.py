import logging
import math
import sys

import torch
import yaml

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - UTILS - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_config(config_path):
    """Load and validate the configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate necessary configuration keys
    required_keys = ['data', 'audio', 'hyperparameters', 'vocos', 'logging', 'paths']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    return config


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Converts timesteps to sinusoidal position embeddings, recording global time information for the 'denoising progress'.
    Args:
        timesteps: A tensor of timesteps, shape [B,].
        dim: The output embedding dimension.
        max_period: The maximum period length.
    Returns:
        An embedding vector of shape [B, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def peak_norm(wav: torch.Tensor, peak: float = 0.99, eps: float = 1e-9) -> torch.Tensor:
    """
    Performs Peak Normalization on an audio waveform.

    Args:
    ----------
    wav : torch.Tensor
        Input waveform tensor, shape can be (T), (1, T), or (C, T),
        with values typically in the [-1, 1] range or smaller.
    peak : float
        The desired "maximum absolute amplitude" after normalization.
        Using 0.99 leaves a small headroom below the full scale of 1.0, which can prevent
        clipping when writing to PCM-16/PCM-24 formats.
    eps : float
        A very small constant to prevent division by zero in silent segments (where max==0)
        and to control numerical stability.

    Returns:
    -------
    torch.Tensor
        The normalized waveform, with a tensor shape identical to the input.
    """

    # 1) Calculate the current peak (maximum absolute amplitude) of the waveform.
    #    .abs() takes the absolute value; .max() returns a scalar tensor.
    current_peak = wav.abs().max()

    # 2) Calculate the scaling factor.
    #    If current_peak is very small (close to 0), this results in a
    #    large scale; eps protects against division by zero.
    scale = peak / (current_peak + eps)

    # 3) Apply the same scaling factor to the entire waveform to achieve linear amplification or reduction.
    wav_norm = wav * scale

    # 4) Return the processed waveform.
    return wav_norm
