import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
import torchaudio
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Flow model and Mel extractor
from models import FlowMatchingModel, MelSpectrogramExtractor
# BEATs related
from beats.Tokenizers import TokenizersConfig, Tokenizers
from beats.BEATs import BEATs, BEATsConfig
from utils import load_config, peak_norm


def setup_device(device_arg: str):
    """Set up the computation device"""
    if device_arg and device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def extend_tokenizer_vocab(tokenizer: AutoTokenizer, config: dict):
    """Extend the tokenizer vocabulary consistent with train_llm.py"""
    K = int(config['hyperparameters']['llm'].get('style_token_count', 16))
    base_special_tokens = ['[DIAGNOSIS]', '[END]', '[BEATs_MASK]', '[PAD]']
    style_tokens = [f'[AUDIO_{i}]' for i in range(K)]
    beats_vocab_size = int(config['hyperparameters']['flow']['vocab_size'])
    beats_tokens = [f'[BEATs_{j}]' for j in range(beats_vocab_size)]

    tokenizer.add_tokens(base_special_tokens + style_tokens + beats_tokens, special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    special_token_ids = {
        'diagnosis': tokenizer.convert_tokens_to_ids('[DIAGNOSIS]'),
        'end': tokenizer.convert_tokens_to_ids('[END]'),
        'beats_mask': tokenizer.convert_tokens_to_ids('[BEATs_MASK]'),
        'pad': tokenizer.convert_tokens_to_ids('[PAD]'),
    }
    audio_style_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in style_tokens]
    return special_token_ids, audio_style_token_ids


def build_model_and_tokenizer(config: dict, device: torch.device):
    """Build the LLM, extend vocabulary, and add the style prefix projection module as in the training script"""
    model_name = config['hyperparameters']['llm']['model_name']

    # 1) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    special_token_ids, audio_style_token_ids = extend_tokenizer_vocab(tokenizer, config)

    # 2) Model
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))

    # Pad/EOS fix & make [END] a terminable ID
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    end_id = tokenizer.convert_tokens_to_ids('[END]')
    base_eos = getattr(model.config, "eos_token_id", None)
    if base_eos is None:
        model.config.eos_token_id = end_id
    else:
        if isinstance(base_eos, (list, tuple, set)):
            eos_ids = sorted(set(list(base_eos) + [end_id]))
        else:
            eos_ids = [base_eos] if base_eos == end_id else [base_eos, end_id]
        model.config.eos_token_id = eos_ids
    try:
        model.generation_config.eos_token_id = model.config.eos_token_id
    except Exception:
        pass

    # 3) Style prefix structure (consistent with training)
    llm_hidden_size = int(config['hyperparameters']['llm'].get('llm_hidden_size', 768))
    beats_feature_dim = int(config['hyperparameters']['llm'].get('beats_feature_size', 768))
    K = int(config['hyperparameters']['llm'].get('style_token_count', 16))

    model.style_token_count = K
    model.beats_feature_dim = beats_feature_dim
    model.style_pool = torch.nn.AdaptiveAvgPool1d(K)

    hidden = int(getattr(model.config, "hidden_size", llm_hidden_size))
    model.style_proj = torch.nn.Sequential(
        torch.nn.Linear(beats_feature_dim, hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden)
    )

    model.to(device).eval()
    return model, tokenizer, special_token_ids, audio_style_token_ids


def load_audio_modules(config: dict, device: torch.device):
    """Load BEATs Tokenizer (discrete tokens) and BEATs Feature Extractor (continuous features)"""
    # 1) BEATs Tokenizer
    tok_ckpt = config['paths']['beats_tokenizer']
    tk_cp = torch.load(tok_ckpt, map_location='cpu')
    tcfg = TokenizersConfig(tk_cp['cfg'])
    beats_tokenizer = Tokenizers(tcfg)
    beats_tokenizer.load_state_dict(tk_cp['model'])
    beats_tokenizer.to(device).eval()

    # 2) BEATs Feature Extractor
    fe_ckpt = config['paths']['beats_feature_extractor_checkpoint']
    fe_cp = torch.load(fe_ckpt, map_location='cpu')
    fcfg = BEATsConfig(fe_cp['cfg'])
    beats_feature_extractor = BEATs(fcfg)
    beats_feature_extractor.load_state_dict(fe_cp['model'])
    beats_feature_extractor.to(device).eval()

    return beats_tokenizer, beats_feature_extractor


def find_best_llm_ckpt(ckpt_dir: Path) -> Path:
    """Find the best model with the minimum loss in the LLM checkpoint directory"""
    pattern = str(ckpt_dir / "best_model_loss_*.pth")
    cands = glob.glob(pattern)
    if not cands:
        print(f"[ERROR] No best_model_loss_*.pth found in {ckpt_dir}.", file=sys.stderr)
        sys.exit(1)

    def _parse_loss(p):
        m = re.search(r"best_model_loss_([0-9.]+)_epoch_", Path(p).name)
        return float(m.group(1)) if m else float('inf')

    best = min(cands, key=_parse_loss)
    return Path(best)


def load_llm_ckpt(model, ckpt_path: Path, device: torch.device):
    """Load an LLM checkpoint into the model"""
    cp = torch.load(ckpt_path, map_location=device)
    state_dict = cp.get('model_state_dict', None)
    if state_dict is None:
        print(f"[ERROR] 'model_state_dict' not found in checkpoint: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing weights during loading: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected weights found: {unexpected}")


def prepare_prompt_inputs(
        model,
        tokenizer,
        audio_style_token_ids,
        beats_feature_extractor,
        waveform_1d: torch.Tensor,
        disease_text: str,
        device: torch.device,
):
    """Construct inputs_embeds and attention_mask for the prompt"""
    if waveform_1d.dim() != 1:
        waveform_1d = waveform_1d.view(-1)
    wave_B1 = waveform_1d.unsqueeze(0).to(device)

    with torch.no_grad():
        feats, _ = beats_feature_extractor.extract_features(wave_B1, padding_mask=None)
        feats_BDT = feats.transpose(1, 2)
        pooled_BDK = model.style_pool(feats_BDT)
        pooled_BKD = pooled_BDK.transpose(1, 2)
        style_embeds_BKH = model.style_proj(pooled_BKD)

        K = int(model.style_token_count)
        audio_prefix_text = " ".join([f"[AUDIO_{j}]" for j in range(K)])
        prompt_text = f"[DIAGNOSIS] {disease_text} {audio_prefix_text}"

        tokenized_prompt = tokenizer(prompt_text, return_tensors='pt').to(device)
        input_ids = tokenized_prompt['input_ids']
        attention_mask = tokenized_prompt['attention_mask']

        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        for j, tok_id in enumerate(audio_style_token_ids):
            pos = (input_ids[0] == tok_id).nonzero(as_tuple=True)[0]
            if pos.numel() > 0:
                idx = pos[0].item()
                if j < style_embeds_BKH.size(1):
                    inputs_embeds[0, idx, :] = style_embeds_BKH[0, j, :]

    return inputs_embeds, attention_mask


def decode_beats_tokens_from_ids(tokenizer, token_ids):
    """Convert a sequence of token ids into a list of [BEATs_*] integer indices"""
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    pred_indices = []
    for tk in tokens:
        if tk == '[END]':
            break
        if tk.startswith('[BEATs_') and tk.endswith(']'):
            try:
                idx = int(tk[7:-1])
                pred_indices.append(idx)
            except Exception:
                pass
    return pred_indices, tokens


def find_best_flow_ckpt(flow_dir: Path) -> Path:
    """Scan the Flow checkpoint directory and select the one with the minimum val loss"""
    cands = list(flow_dir.glob("best_ep*_val_loss_*.pt"))
    if not cands:
        print(f"[ERROR] No best_ep*_val_loss_*.pt found in {flow_dir}", file=sys.stderr)
        sys.exit(1)

    def _parse_loss(p: Path):
        m = re.search(r"val_loss_([0-9.]+)_step", p.name)
        return float(m.group(1)) if m else float('inf')

    return min(cands, key=_parse_loss)


def load_wav_mono(path: str, target_sr: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Modified audio loading function:
    Loads an audio file, converts it to mono, applies peak normalization and clamps to [-1, 1],
    and finally resamples to the target sample rate.
    This method mimics the logic in data_loader.py's _safe_load_audio.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        wav, sr = torchaudio.load(path)

        if wav.numel() == 0:
            print(f"[WARN] Loaded audio is empty: {path}")
            return wav.squeeze(0)

        # Convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)

        # Peak normalization (from data_loader.py)
        max_val = torch.abs(wav).max()
        # Use logic from peak_norm, but normalize directly to 1.0 without the peak parameter
        safe_divisor = max_val if max_val > eps else torch.tensor(1.0, device=wav.device, dtype=wav.dtype)
        wav = wav / safe_divisor
        wav = torch.clamp(wav, -1.0, 1.0)

        # Resample
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        return wav.squeeze(0)  # -> [T]
    except Exception as e:
        print(f"[ERROR] Failed to load or process audio {path}: {e}", file=sys.stderr)
        return torch.tensor([])


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(
        description="Generate BEATs tokens with LLM + Reconstruct audio with CFM (Asthma + reference audio)")
    parser.add_argument("--config", type=str, default="/mnt/data/RespAgent/Agent/Generator/config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device, e.g., cuda:4 / cuda:0 / cpu")
    parser.add_argument("--ref_audio", type=str,
                        default="/mnt/data/RespAgent/Agent/Generator/wav/reference_audio.wav",
                        help="Path to the reference audio file")
    parser.add_argument("--out_dir", type=str, default="/mnt/data/RespAgent/Agent/Generator/output_generate",
                        help="Output directory")
    parser.add_argument("--disease", type=str, default="COPD", help="Disease text (used for generating BEATs tokens)")
    parser.add_argument("--ref_ratio", type=float, default=0.30,
                        help="Ratio of the reference prefix to the total mel frames")
    parser.add_argument("--max_new_tokens_pad", type=int, default=8,
                        help="Number of extra tokens to pad the generation length")
    args = parser.parse_args()

    # Device
    device = setup_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Configuration
    config = load_config(args.config)
    ckpt_root = Path(config["paths"]["checkpoint_dir"])

    # ========== 1) LLM: Build and load best weights ==========
    model, tokenizer, special_token_ids, audio_style_token_ids = build_model_and_tokenizer(config, device)
    llm_ckpt_dir = ckpt_root / "llm"
    best_llm = find_best_llm_ckpt(llm_ckpt_dir)
    print(f"[INFO] Using best LLM checkpoint: {best_llm.name}")
    load_llm_ckpt(model, best_llm, device)

    # ========== 2) BEATs modules: Discrete tokenizer and continuous feature extractor ==========
    beats_tokenizer, beats_feature_extractor = load_audio_modules(config, device)

    # ========== 3) Read reference audio, construct LLM prompt, and generate BEATs tokens ==========
    llm_audio_sr = int(config["audio"]["sample_rate"])  # 16k (consistent with LLM side)
    ref_wav_16k = load_wav_mono(args.ref_audio, llm_audio_sr)
    if ref_wav_16k.numel() == 0:
        print(f"[ERROR] Reference audio is empty or failed to load: {args.ref_audio}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Disease text: {args.disease}")
    inputs_embeds, attention_mask = prepare_prompt_inputs(
        model=model,
        tokenizer=tokenizer,
        audio_style_token_ids=audio_style_token_ids,
        beats_feature_extractor=beats_feature_extractor,
        waveform_1d=ref_wav_16k,
        disease_text=args.disease,
        device=device,
    )

    with torch.no_grad():
        gt_flat = beats_tokenizer.extract_labels(ref_wav_16k.unsqueeze(0).to(device))
        est_len = int(gt_flat.numel())

    max_new = est_len + max(1, int(args.max_new_tokens_pad))
    print(f"[INFO] Generating BEATs tokens, max_new_tokens={max_new}")
    with torch.no_grad():
        gen_out = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=tokenizer.convert_tokens_to_ids('[END]'),
            pad_token_id=tokenizer.pad_token_id
        )
        gen_ids = gen_out[0].tolist()

    pred_indices, _ = decode_beats_tokens_from_ids(tokenizer, gen_ids)
    if len(pred_indices) == 0:
        print(
            "[WARN] The generated BEATs token sequence is empty. Proceeding to reconstruction, but the result may be poor.")

    # ========== 4) Flow Matching: Build and load best weights ==========
    flow_model = FlowMatchingModel(config).to(device).eval()
    flow_ckpt_dir = ckpt_root / "flow"
    best_flow = find_best_flow_ckpt(flow_ckpt_dir)
    print(f"[INFO] Using best Flow checkpoint: {best_flow.name}")

    ckpt = torch.load(best_flow, map_location=device)
    if "model_state_dict" in ckpt:
        missing, unexpected = flow_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:   print(f"[WARN] Flow missing weights on load: {missing}")
        if unexpected: print(f"[WARN] Flow unexpected weights on load: {unexpected}")
    else:
        print(f"[ERROR] 'model_state_dict' not found in {best_flow}", file=sys.stderr)
        sys.exit(1)

    # ========== 5) Mel extractor and condition preparation ==========
    vocos_sr = int(config["vocos"]["sample_rate"])
    mel_extractor = MelSpectrogramExtractor(config, target_device=device)

    ref_mel_full = mel_extractor(ref_wav_16k.unsqueeze(0), normalize=True)
    B, n_mels, T_full = ref_mel_full.shape

    ref_len = max(1, int(T_full * float(args.ref_ratio)))
    ref_prefix = ref_mel_full[:, :, :ref_len]

    with torch.no_grad():
        if len(pred_indices) == 0:
            token_indices = torch.tensor([[0]], dtype=torch.long, device=device)
        else:
            token_indices = torch.tensor([pred_indices], dtype=torch.long, device=device)

        token_embed = flow_model.token_embedding(token_indices).transpose(1, 2)
        token_embed_up = torch.nn.functional.interpolate(
            token_embed, size=T_full, mode="linear", align_corners=False
        )
        feats_16k, _ = beats_feature_extractor.extract_features(ref_wav_16k.unsqueeze(0).to(device), padding_mask=None)
        timbre_mean = feats_16k.mean(dim=1, keepdim=True).transpose(1, 2)
        timbre_up = timbre_mean.expand(-1, -1, T_full)
        fused_embed = torch.cat([token_embed_up, timbre_up], dim=1)

        ref_pad = torch.zeros_like(ref_mel_full, device=device)
        ref_pad[:, :, :ref_len] = ref_prefix
        cond_embed_dict = {
            "fused_embed": fused_embed,
            "ref_mel_for_cond": ref_pad,
        }

    # ========== 6) Sample to generate full mel, then overwrite with reference prefix ==========
    n_steps = int(config["hyperparameters"]["flow"]["n_timesteps"])
    cfg_scale = float(config["hyperparameters"]["flow"]["cfg_scale"])
    target_full_len = T_full

    print(
        f"[INFO] Flow sampling: steps={n_steps}, cfg_scale={cfg_scale}, ref_len={ref_len}, total_gen_len={target_full_len}")
    with torch.no_grad():
        # 1. Generate the complete mel-spectrogram
        generated_full_mel = flow_model.sample(
            cond_embed_dict=cond_embed_dict,
            target_duration_frames=target_full_len,
            steps=n_steps,
            cfg_scale=cfg_scale,
            sway_sampling_coef=-1.0
        )

        # 2. Replace the prefix of the generated result with the actual reference prefix
        final_mel = generated_full_mel.clone()
        final_mel[:, :, :ref_len] = ref_prefix

    full_mel_norm = final_mel
    mel_mean = float(config["hyperparameters"]["flow"]["mel_mean"])
    mel_std = float(config["hyperparameters"]["flow"]["mel_std"])
    full_mel_raw = full_mel_norm * mel_std + mel_mean

    # ========== 7) Decode to audio with Vocos ==========
    try:
        from vocos import Vocos
    except Exception:
        print("[ERROR] vocos is not installed. Please `pip install vocos` first.", file=sys.stderr)
        raise

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    with torch.inference_mode():
        # Use the integrated peak_norm function and set peak=0.99 to prevent clipping
        audio_24k = vocos.decode(full_mel_raw.to(device))
        audio_24k = peak_norm(audio_24k.cpu(), peak=0.99)

    # ========== 8) Save the output ==========
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "pred_beats_tokens.json", "w", encoding="utf-8") as f:
        json.dump({"disease": args.disease, "pred_indices": pred_indices}, f, ensure_ascii=False, indent=2)

    wav_path = out_dir / "generated_from_llm_cfm.wav"
    torchaudio.save(str(wav_path), audio_24k, vocos_sr, format="wav")

    print("\n====== Generation Complete ======")
    print(f"BEATs tokens saved to: {out_dir / 'pred_beats_tokens.json'}")
    print(f"Audio saved to:        {wav_path.resolve()}")
    print(f"Reference audio:       {args.ref_audio}")
    print(f"Disease text:          {args.disease}")


if __name__ == "__main__":
    seed = 666
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main()
