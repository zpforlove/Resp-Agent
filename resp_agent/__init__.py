"""
Resp-Agent: A multi-agent framework for respiratory sound diagnosis and generation.

This package provides:
- BEATs audio feature extraction and tokenization
- Diagnoser pipeline for respiratory sound classification
- Generator pipeline for respiratory sound synthesis
- Interactive agent interface (Chinese/English)
"""

__version__ = "0.1.0"
__author__ = "Austin"

# Core audio modules
# Agent modules
from .agent import RespAgentChinese, RespAgentEnglish
from .beats import BEATs, BEATsConfig, Tokenizers, TokenizersConfig

# Pipeline modules
from .diagnoser import DiagnoserPipeline, run_diagnoser
from .generator import FlowMatchingModel, GeneratorPipeline, run_generator

__all__ = [
    # Version
    "__version__",
    # BEATs
    "BEATs",
    "BEATsConfig",
    "Tokenizers",
    "TokenizersConfig",
    # Diagnoser
    "DiagnoserPipeline",
    "run_diagnoser",
    # Generator
    "GeneratorPipeline",
    "run_generator",
    "FlowMatchingModel",
    # Agent
    "RespAgentChinese",
    "RespAgentEnglish",
]
