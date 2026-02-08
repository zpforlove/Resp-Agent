"""
BEATs: Audio Pre-Training with Acoustic Tokenizers

This module provides the BEATs audio feature extractor and tokenizer.
Based on: https://github.com/microsoft/unilm/tree/master/beats
"""

from .BEATs import BEATs, BEATsConfig
from .Tokenizers import Tokenizers, TokenizersConfig

__all__ = [
    "BEATs",
    "BEATsConfig",
    "Tokenizers",
    "TokenizersConfig",
]
