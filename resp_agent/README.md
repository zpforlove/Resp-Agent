# Resp-Agent

A multi-agent framework for respiratory sound diagnosis and generation using deep learning.

## Installation

### 1. Install PyTorch with CUDA Support

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install resp-agent

```bash
pip install resp-agent -i https://pypi.org/simple/
```

> This installs all dependencies for both inference and training (including `deepspeed`, `wandb`, `matplotlib`, etc.).

## Quick Start

### Command Line Interface

```bash
# Start interactive chat agent
resp-agent chat --lang zh  # Chinese
resp-agent chat --lang en  # English
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0 (recommended: 2.8.0 with CUDA 12.8)
- CUDA-capable GPU (recommended for training and inference)

## Environment Variables

Set your DeepSeek API key for the Thinker agent:
```bash
export DEEPSEEK_API_KEY='your-api-key'
```

## License

MIT License
