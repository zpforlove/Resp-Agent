"""
Agent module for interactive respiratory sound analysis.

This module provides the Thinker agent that coordinates the Diagnoser and Generator.
"""

from .chinese import RespAgentChinese
from .chinese import main as main_chinese
from .english import RespAgentEnglish
from .english import main as main_english

__all__ = [
    "RespAgentChinese",
    "RespAgentEnglish",
    "main_chinese",
    "main_english",
]
