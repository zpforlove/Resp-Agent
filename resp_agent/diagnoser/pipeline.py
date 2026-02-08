"""
Diagnoser Pipeline for respiratory sound classification.

This module provides the diagnosis pipeline for analyzing respiratory sounds
using BEATs features and Longformer classification.
"""

import glob
import os
import re

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..beats import BEATs, BEATsConfig
from .utils import load_config

# Constants
FIXED_AUDIO_STEPS = 496
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_MAX_TEXT_TOKENS = 128


class DiagnoserPipeline:
    """
    Diagnoser pipeline for respiratory sound classification.

    This class encapsulates the complete diagnosis workflow:
    1. Generate text descriptions from audio+EHR using LLM
    2. Extract BEATs features from audio
    3. Classify using Longformer model
    """

    def __init__(self, config_path: str = None, device: str = "cuda:0"):
        """
        Initialize the diagnoser pipeline.

        Args:
            config_path: Path to the configuration YAML file
            device: Device to run inference on
        """
        self.config_path = config_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = None
        self.model = None
        self.tokenizer = None
        self.beats = None

    def load_models(self, ckpt_dir: str = None, config_path: str = None):
        """Load all models required for diagnosis"""
        if config_path:
            self.config_path = config_path

        if not self.config_path:
            raise ValueError("config_path must be provided")

        self.config = load_config(self.config_path)
        # Model loading logic here

    def diagnose(
        self,
        audio_dir: str,
        output_dir: str,
        metadata_csv: str,
    ) -> pd.DataFrame:
        """
        Run diagnosis on audio files.

        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save results
            metadata_csv: Path to metadata CSV file

        Returns:
            DataFrame with diagnosis results
        """
        # Diagnosis logic - to be implemented based on original pipeline
        pass


def run_diagnoser(
    audio_dir: str,
    output_dir: str,
    metadata_csv: str,
    config_path: str = None,
    ckpt_dir: str = None,
    device: str = "cuda:0",
) -> str:
    """
    Convenience function to run the diagnoser pipeline.

    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save results
        metadata_csv: Path to metadata CSV file
        config_path: Path to configuration file
        ckpt_dir: Path to checkpoint directory
        device: Computation device

    Returns:
        Path to the results CSV file
    """
    pipeline = DiagnoserPipeline(config_path=config_path, device=device)
    pipeline.load_models(ckpt_dir=ckpt_dir)
    results = pipeline.diagnose(
        audio_dir=audio_dir,
        output_dir=output_dir,
        metadata_csv=metadata_csv,
    )
    return results


def build_prompt(row: pd.Series, audio_filename: str, audio_type: str) -> str:
    """Build English summary prompt template"""
    patient_info = f"""- Patient ID: {row["participant_identifier"]}
- Age: {row["age"]}
- Gender: {row["gender"]}
- Region: {row["region_name"]}
- Smoker Status: {row["smoker_status"]}
- History of Asthma: {"Yes" if row["respiratory_condition_asthma"] == 1 else "No"}
- Other Respiratory Conditions: {"Yes" if row["respiratory_condition_other"] == 1 else "No"}"""

    symptoms = []
    if row.get("symptom_cough_any", 0) == 1:
        symptoms.append("Any cough")
    if row.get("symptom_new_continuous_cough", 0) == 1:
        symptoms.append("New continuous cough")
    if row.get("symptom_shortness_of_breath", 0) == 1:
        symptoms.append("Shortness of breath")
    if row.get("symptom_sore_throat", 0) == 1:
        symptoms.append("Sore throat")
    if row.get("symptom_fatigue", 0) == 1:
        symptoms.append("Fatigue")
    if row.get("symptom_fever_high_temperature", 0) == 1:
        symptoms.append("Fever")
    symptoms_str = ", ".join(symptoms) if symptoms else "No reported symptoms"

    covid_info = f"""- COVID Viral Load Category: {row.get("covid_viral_load_category", "Unknown")}
- Reported Symptoms: {symptoms_str}"""

    audio_info = f"""- Audio Type: {audio_type}"""

    return f"""Based on the following information, please generate a professional, fluent, and natural English summary description for the audio file '{audio_filename}'. Output the summary description directly, without any explanation or thought process.

### Patient Information
{patient_info}

### COVID-19 Related Information
{covid_info}

### Audio File Information
{audio_info}

### Summary Description:
"""


def clean_llm_output(response_text: str) -> str:
    """Post-processing and cleaning of the generated output"""
    text_to_clean = (
        response_text.split("</think>")[-1]
        if "</think>" in response_text
        else response_text
    )
    if "### Summary Description:" in text_to_clean:
        text_to_clean = text_to_clean.split("### Summary Description:")[-1]
    cleaned_text = re.sub(
        r"^\s*Here is the summary.*\s*[:：]\s*", "", text_to_clean, flags=re.IGNORECASE
    )
    return cleaned_text.strip()


def detect_audio_type(row: pd.Series, filename: str) -> str:
    """Determine the file type based on the metadata row"""
    for t in ["exhalation", "cough", "three_cough"]:
        col = f"{t}_file_name"
        if col in row and pd.notna(row[col]) and str(row[col]) == filename:
            return t
    return "unknown"


def extract_pid_from_filename(fname: str) -> str:
    """Match the leading numbers in the filename as the patient ID"""
    m = re.match(r"^(\d+)", fname)
    return m.group(1) if m else ""


def find_best_checkpoint(ckpt_dir: str) -> str:
    """Find the best checkpoint with minimum loss"""
    pattern = os.path.join(ckpt_dir, "best_longformer_loss_*.pth")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No 'best_longformer_loss_*.pth' found in {ckpt_dir}.")
    best_path, best_loss = None, float("inf")
    for f in files:
        m = re.search(
            r"best_longformer_loss_(\d+\.\d+)_epoch_\d+\.pth$", os.path.basename(f)
        )
        if not m:
            continue
        loss = float(m.group(1))
        if loss < best_loss:
            best_loss, best_path = loss, f
    if best_path is None:
        raise RuntimeError("Could not parse loss value from filename.")
    return best_path


class LongformerWithBEATsInfer:
    """Inference process wrapper for Longformer with BEATs features"""

    def __init__(
        self,
        config: dict,
        checkpoint_path: str,
        idx_to_disease: dict,
        device: torch.device,
    ):
        self.cfg = config
        self.device = device

        pack = torch.load(checkpoint_path, map_location=device)
        client = pack.get("client_state", {})
        self.train_cfg = self.cfg

        self.model_name = (
            self.train_cfg.get("hyperparameters", {})
            .get("longformer", {})
            .get("model_name", "allenai/longformer-base-4096")
        )
        self.max_seq_len = (
            self.train_cfg.get("hyperparameters", {})
            .get("longformer", {})
            .get("max_sequence_length", DEFAULT_MAX_SEQ_LEN)
        )
        self.max_text_tokens = (
            self.train_cfg.get("hyperparameters", {})
            .get("longformer", {})
            .get("max_text_tokens", DEFAULT_MAX_TEXT_TOKENS)
        )
        self.beats_feat_dim = (
            self.train_cfg.get("hyperparameters", {})
            .get("longformer", {})
            .get("beats_feature_size", 768)
        )
        self.audio_global_stride = int(
            self.train_cfg.get("hyperparameters", {})
            .get("longformer", {})
            .get("audio_global_stride", 8)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token:
                self.tokenizer.add_special_tokens(
                    {"pad_token": self.tokenizer.eos_token}
                )
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "right"

        base_special_tokens = ["[DESCRIPTION]"]
        audio_embed_tokens = [
            f"[AUDIO_EMBED_{i}]" for i in range(1, FIXED_AUDIO_STEPS + 1)
        ]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": base_special_tokens + audio_embed_tokens}
        )

        no_split = set(getattr(self.tokenizer, "unique_no_split_tokens", []))
        no_split.update(base_special_tokens + audio_embed_tokens)
        self.tokenizer.unique_no_split_tokens = list(no_split)

        self.desc_tok_id = self.tokenizer.convert_tokens_to_ids("[DESCRIPTION]")
        self.audio_embed_ids = self.tokenizer.convert_tokens_to_ids(audio_embed_tokens)
        self.audio_embed_start_id = self.tokenizer.convert_tokens_to_ids(
            "[AUDIO_EMBED_1]"
        )

        num_labels = int(client.get("num_labels", len(idx_to_disease)))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        hidden = self.model.get_input_embeddings().embedding_dim
        self.model.projection_layer = nn.Linear(self.beats_feat_dim, hidden)

        if "model_state_dict" in pack:
            missing, unexpected = self.model.load_state_dict(
                pack["model_state_dict"], strict=False
            )
            if missing:
                print(f"[CKPT] Missing keys: {len(missing)}")
            if unexpected:
                print(f"[CKPT] Unexpected keys: {len(unexpected)}")

        if "projection_layer_state_dict" in pack:
            try:
                self.model.projection_layer.load_state_dict(
                    pack["projection_layer_state_dict"], strict=False
                )
            except Exception as e:
                print(f"[CKPT] Failed to load projection_layer_state_dict: {e}")

        self.model.to(self.device).eval()

        beats_ckpt_path = self.train_cfg.get("paths", {}).get(
            "beats_feature_extractor_checkpoint"
        )
        if not beats_ckpt_path or not os.path.exists(beats_ckpt_path):
            raise FileNotFoundError(f"BEATs checkpoint not found: {beats_ckpt_path}")
        ckpt = torch.load(beats_ckpt_path, map_location="cpu")
        bcfg = BEATsConfig(ckpt["cfg"])
        self.beats = BEATs(bcfg)
        self.beats.load_state_dict(ckpt["model"])
        self.beats.eval().to(self.device)

    @staticmethod
    def align_to_fixed_steps(
        x: torch.Tensor, steps: int = FIXED_AUDIO_STEPS, mode: str = "center"
    ):
        B, T, D = x.shape
        if T == steps:
            return x
        if T > steps:
            if mode == "left":
                return x[:, :steps, :]
            if mode == "right":
                return x[:, -steps:, :]
            start = (T - steps) // 2
            return x[:, start : start + steps, :]
        pad_len = steps - T
        pad = x.new_zeros(B, pad_len, D)
        return torch.cat([x, pad], dim=1)
