"""
Generator Pipeline for respiratory sound synthesis.

This module provides the generation pipeline for synthesizing respiratory sounds
using LLM-guided BEATs tokens and Flow Matching acoustic model.
"""

import os
import sys

import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..beats import BEATs, BEATsConfig, Tokenizers, TokenizersConfig
from .utils import load_config


class GeneratorPipeline:
    """
    Generator pipeline for respiratory sound synthesis.

    This class encapsulates the complete generation workflow:
    1. Load LLM and generate BEATs tokens from disease text + reference audio
    2. Use Flow Matching model to generate mel-spectrogram
    3. Decode to audio with Vocos vocoder
    """

    def __init__(self, config_path: str = None, device: str = "cuda:0"):
        """
        Initialize the generator pipeline.

        Args:
            config_path: Path to the configuration YAML file
            device: Device to run inference on
        """
        self.config_path = config_path
        self.device = self._setup_device(device)
        self.config = None
        self.model = None
        self.tokenizer = None
        self.beats_tokenizer = None
        self.beats_feature_extractor = None
        self.flow_model = None

    def _setup_device(self, device_arg: str):
        """Set up the computation device"""
        if device_arg and device_arg.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_arg)
        return torch.device("cpu")

    def load_models(self, config_path: str = None):
        """Load all models required for generation"""
        if config_path:
            self.config_path = config_path

        if not self.config_path:
            raise ValueError("config_path must be provided")

        self.config = load_config(self.config_path)
        # Model loading logic here

    def generate(
        self,
        ref_audio: str,
        disease: str,
        out_dir: str,
        ref_ratio: float = 0.30,
        max_new_tokens_pad: int = 8,
    ) -> str:
        """
        Generate respiratory sound.

        Args:
            ref_audio: Path to reference audio file
            disease: Disease text for content guidance
            out_dir: Output directory
            ref_ratio: Ratio of reference prefix to total frames
            max_new_tokens_pad: Extra tokens padding for generation

        Returns:
            Path to the generated audio file
        """
        # Generation logic - to be implemented based on original pipeline
        pass


def run_generator(
    ref_audio: str,
    disease: str,
    out_dir: str,
    config_path: str = None,
    device: str = "cuda:0",
    ref_ratio: float = 0.30,
) -> str:
    """
    Convenience function to run the generator pipeline.

    Args:
        ref_audio: Path to reference audio file
        disease: Disease text for content guidance
        out_dir: Output directory
        config_path: Path to configuration file
        device: Computation device
        ref_ratio: Ratio of reference prefix

    Returns:
        Path to the generated audio file
    """
    pipeline = GeneratorPipeline(config_path=config_path, device=device)
    pipeline.load_models()
    return pipeline.generate(
        ref_audio=ref_audio,
        disease=disease,
        out_dir=out_dir,
        ref_ratio=ref_ratio,
    )


def setup_device(device_arg: str):
    """Set up the computation device"""
    if device_arg and device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def extend_tokenizer_vocab(tokenizer: AutoTokenizer, config: dict):
    """Extend the tokenizer vocabulary consistent with train_llm.py"""
    K = int(config["hyperparameters"]["llm"].get("style_token_count", 16))
    base_special_tokens = ["[DIAGNOSIS]", "[END]", "[BEATs_MASK]", "[PAD]"]
    style_tokens = [f"[AUDIO_{i}]" for i in range(K)]
    beats_vocab_size = int(config["hyperparameters"]["flow"]["vocab_size"])
    beats_tokens = [f"[BEATs_{j}]" for j in range(beats_vocab_size)]

    tokenizer.add_tokens(
        base_special_tokens + style_tokens + beats_tokens, special_tokens=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    special_token_ids = {
        "diagnosis": tokenizer.convert_tokens_to_ids("[DIAGNOSIS]"),
        "end": tokenizer.convert_tokens_to_ids("[END]"),
        "beats_mask": tokenizer.convert_tokens_to_ids("[BEATs_MASK]"),
        "pad": tokenizer.convert_tokens_to_ids("[PAD]"),
    }
    audio_style_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in style_tokens]
    return special_token_ids, audio_style_token_ids


def build_model_and_tokenizer(config: dict, device: torch.device):
    """Build the LLM, extend vocabulary, and add the style prefix projection module"""
    model_name = config["hyperparameters"]["llm"]["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    special_token_ids, audio_style_token_ids = extend_tokenizer_vocab(tokenizer, config)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))

    if (
        getattr(model.config, "pad_token_id", None) is None
        and tokenizer.pad_token_id is not None
    ):
        model.config.pad_token_id = tokenizer.pad_token_id
    end_id = tokenizer.convert_tokens_to_ids("[END]")
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

    llm_hidden_size = int(config["hyperparameters"]["llm"].get("llm_hidden_size", 768))
    beats_feature_dim = int(
        config["hyperparameters"]["llm"].get("beats_feature_size", 768)
    )
    K = int(config["hyperparameters"]["llm"].get("style_token_count", 16))

    model.style_token_count = K
    model.beats_feature_dim = beats_feature_dim
    model.style_pool = torch.nn.AdaptiveAvgPool1d(K)

    hidden = int(getattr(model.config, "hidden_size", llm_hidden_size))
    model.style_proj = torch.nn.Sequential(
        torch.nn.Linear(beats_feature_dim, hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden),
    )

    model.to(device).eval()
    return model, tokenizer, special_token_ids, audio_style_token_ids


def load_audio_modules(config: dict, device: torch.device):
    """Load BEATs Tokenizer and Feature Extractor"""
    tok_ckpt = config["paths"]["beats_tokenizer"]
    tk_cp = torch.load(tok_ckpt, map_location="cpu")
    tcfg = TokenizersConfig(tk_cp["cfg"])
    beats_tokenizer = Tokenizers(tcfg)
    beats_tokenizer.load_state_dict(tk_cp["model"])
    beats_tokenizer.to(device).eval()

    fe_ckpt = config["paths"]["beats_feature_extractor_checkpoint"]
    fe_cp = torch.load(fe_ckpt, map_location="cpu")
    fcfg = BEATsConfig(fe_cp["cfg"])
    beats_feature_extractor = BEATs(fcfg)
    beats_feature_extractor.load_state_dict(fe_cp["model"])
    beats_feature_extractor.to(device).eval()

    return beats_tokenizer, beats_feature_extractor


def load_wav_mono(path: str, target_sr: int, eps: float = 1e-8) -> torch.Tensor:
    """Load audio, convert to mono, normalize and resample"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        wav, sr = torchaudio.load(path)

        if wav.numel() == 0:
            print(f"[WARN] Loaded audio is empty: {path}")
            return wav.squeeze(0)

        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)

        max_val = torch.abs(wav).max()
        safe_divisor = (
            max_val
            if max_val > eps
            else torch.tensor(1.0, device=wav.device, dtype=wav.dtype)
        )
        wav = wav / safe_divisor
        wav = torch.clamp(wav, -1.0, 1.0)

        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        return wav.squeeze(0)
    except Exception as e:
        print(f"[ERROR] Failed to load or process audio {path}: {e}", file=sys.stderr)
        return torch.tensor([])
