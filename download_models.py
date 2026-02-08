#!/usr/bin/env python3
"""
Download model files from HuggingFace Hub for Resp-Agent.

This script downloads all required model files and places them in the correct directories.

Usage:
    python download_models.py

Requirements:
    pip install huggingface_hub
"""

import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


# HuggingFace repository for Resp-Agent models
RESP_AGENT_MODELS_REPO = "AustinZhang/resp-agent-models"

# DeepSeek-R1 model (use official repo)
DEEPSEEK_R1_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.resolve()


def download_resp_agent_models():
    """Download Resp-Agent models from HuggingFace."""
    project_root = get_project_root()

    print("=" * 60)
    print("Downloading Resp-Agent Models from HuggingFace")
    print("=" * 60)

    # Download all models from AustinZhang/resp-agent-models
    print(f"\n[1/2] Downloading models from {RESP_AGENT_MODELS_REPO}...")
    try:
        local_path = snapshot_download(
            repo_id=RESP_AGENT_MODELS_REPO,
            local_dir=project_root,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", ".gitattributes"],
        )
        print(f"✓ Models downloaded to: {local_path}")
    except Exception as e:
        print(f"✗ Error downloading Resp-Agent models: {e}")
        return False

    # Download DeepSeek-R1 model
    deepseek_dir = project_root / "Diagnoser" / "checkpoints" / "deepseek-r1"
    print(f"\n[2/2] Downloading DeepSeek-R1 from {DEEPSEEK_R1_REPO}...")
    try:
        snapshot_download(
            repo_id=DEEPSEEK_R1_REPO,
            local_dir=deepseek_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ DeepSeek-R1 downloaded to: {deepseek_dir}")
    except Exception as e:
        print(f"✗ Error downloading DeepSeek-R1: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All models downloaded successfully!")
    print("=" * 60)

    return True


def verify_models():
    """Verify that all required model files exist."""
    project_root = get_project_root()

    required_files = [
        # Diagnoser models
        "Diagnoser/checkpoints/deepseek-r1/config.json",
        "Diagnoser/checkpoints/longformer/best_longformer_loss_0.3374_epoch_2.pth",
        "Diagnoser/pretrained_models/BEATs_iter3_plus_AS2M.pt",
        "Diagnoser/pretrained_models/Tokenizer_iter3_plus_AS2M.pt",
        # Generator models
        "Generator/checkpoints/llm/best_model_loss_1.2393_epoch_3.pth",
        "Generator/checkpoints/flow/best_ep5_val_loss_0.0638_step164485.pt",
        "Generator/pretrained_models/BEATs_iter3_plus_AS2M.pt",
        "Generator/pretrained_models/Tokenizer_iter3_plus_AS2M.pt",
    ]

    print("\nVerifying model files...")
    missing = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (MISSING)")
            missing.append(file_path)

    if missing:
        print(f"\n⚠ Warning: {len(missing)} files are missing.")
        return False
    else:
        print("\n✓ All required model files are present!")
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Resp-Agent models from HuggingFace"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing files"
    )
    args = parser.parse_args()

    if args.verify_only:
        success = verify_models()
    else:
        success = download_resp_agent_models()
        if success:
            verify_models()

    sys.exit(0 if success else 1)
