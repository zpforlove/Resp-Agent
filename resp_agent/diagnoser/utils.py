"""
Utility functions for the diagnoser module.
"""

import logging
import math
import os
import sys

import torch
import yaml

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - DIAGNOSER - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_config(config_path):
    """Load and validate the configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate necessary configuration keys
    required_keys = ["data", "audio", "hyperparameters", "vocos", "logging", "paths"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    return config


def safe_load_audio(path: str):
    """Fault-tolerant loading as a mono channel [1, T]"""
    import torchaudio

    try:
        if (not os.path.exists(path)) or os.path.getsize(path) == 0:
            return None, 0
        waveform, sr = torchaudio.load(path)
        if waveform.numel() == 0:
            return None, 0
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, sr
    except Exception:
        return None, 0


def resample_pad_or_trim(
    waveform: torch.Tensor, sr: int, target_sr: int, target_len_samples: int
):
    """Resample to target_sr and trim/pad to a fixed length"""
    import torchaudio

    if sr != target_sr:
        try:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        except Exception:
            return None
    cur = waveform.shape[1]
    if cur > target_len_samples:
        waveform = waveform[:, :target_len_samples]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_len_samples - cur))
    return waveform


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Converts timesteps to sinusoidal position embeddings.
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
