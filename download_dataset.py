#!/usr/bin/env python3
"""
Download dataset files from HuggingFace Hub for Resp-Agent.

This script downloads the Resp-229K dataset (~66GB, 229K audio files, 407+ hours)
required for training and fine-tuning the Resp-Agent models.

Usage:
    python download_dataset.py
    python download_dataset.py --output-dir ./data
    python download_dataset.py --verify-only

Requirements:
    pip install huggingface_hub
"""

import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


# HuggingFace repository for Resp-Agent dataset
RESP_AGENT_DATASET_REPO = "AustinZhang/resp-agent-dataset"


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.resolve()


def download_dataset(output_dir=None):
    """Download Resp-Agent dataset from HuggingFace.
    
    Args:
        output_dir: Optional output directory. Defaults to ./data in project root.
    
    Returns:
        bool: True if download successful, False otherwise.
    """
    project_root = get_project_root()
    
    if output_dir is None:
        output_dir = project_root / "data"
    else:
        output_dir = Path(output_dir).resolve()

    print("=" * 60)
    print("Downloading Resp-Agent Dataset from HuggingFace")
    print("=" * 60)
    print(f"\nRepository: {RESP_AGENT_DATASET_REPO}")
    print(f"Output directory: {output_dir}")
    print("\n⚠ Note: This dataset is ~66GB. Download may take a while.\n")

    try:
        local_path = snapshot_download(
            repo_id=RESP_AGENT_DATASET_REPO,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        print(f"\n✓ Dataset downloaded to: {local_path}")
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ Dataset downloaded successfully!")
    print("=" * 60)
    
    # Print next steps
    print("\n📋 Next Steps:")
    print("1. Update paths in Diagnoser/config.yaml:")
    print("   data:")
    print(f'     train_root: "{output_dir}/train"')
    print(f'     val_root: "{output_dir}/valid"')
    print(f'     test_root: "{output_dir}/test"')

    return True


def verify_dataset(output_dir=None):
    """Verify that the dataset files exist.
    
    Args:
        output_dir: Directory to check. Defaults to ./data in project root.
    
    Returns:
        bool: True if all required directories exist, False otherwise.
    """
    project_root = get_project_root()
    
    if output_dir is None:
        output_dir = project_root / "data"
    else:
        output_dir = Path(output_dir).resolve()

    expected_dirs = ["train", "valid", "test"]
    
    print(f"\nVerifying dataset in: {output_dir}")
    
    if not output_dir.exists():
        print(f"  ✗ Directory does not exist: {output_dir}")
        return False
    
    missing = []
    for dir_name in expected_dirs:
        dir_path = output_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            # Count files in directory
            file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
            print(f"  ✓ {dir_name}/ ({file_count} files)")
        else:
            print(f"  ✗ {dir_name}/ (MISSING)")
            missing.append(dir_name)

    if missing:
        print(f"\n⚠ Warning: {len(missing)} directories are missing.")
        print("Run: python download_dataset.py")
        return False
    else:
        print("\n✓ Dataset structure verified!")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Resp-Agent dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for dataset (default: ./data)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset files"
    )
    args = parser.parse_args()

    if args.verify_only:
        success = verify_dataset(args.output_dir)
    else:
        success = download_dataset(args.output_dir)
        if success:
            verify_dataset(args.output_dir)

    sys.exit(0 if success else 1)
