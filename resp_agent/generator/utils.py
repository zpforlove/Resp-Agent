"""
Utility functions for the generator module.
"""

import logging
import math
import sys

import torch
import yaml

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - GENERATOR - %(levelname)s - %(message)s"
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


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Converts timesteps to sinusoidal position embeddings.
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
        wav: Input waveform tensor.
        peak: The desired maximum absolute amplitude after normalization.
        eps: A small constant to prevent division by zero.

    Returns:
        The normalized waveform.
    """
    current_peak = wav.abs().max()
    scale = peak / (current_peak + eps)
    wav_norm = wav * scale
    return wav_norm
