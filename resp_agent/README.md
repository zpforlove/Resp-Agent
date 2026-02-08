# Resp-Agent

A multi-agent framework for respiratory sound diagnosis and generation using deep learning.

## Installation

### Basic Installation

```bash
pip install resp-agent
```

### With CUDA Support (Recommended)

First install PyTorch with CUDA support:
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install resp-agent:
```bash
pip install resp-agent
```

Or install with CUDA dependencies together:
```bash
pip install resp-agent[cuda]
```

## Quick Start

### Python API

```python
from resp_agent import BEATs, BEATsConfig
from resp_agent.diagnoser import run_diagnoser
from resp_agent.generator import run_generator

# Diagnose respiratory sounds
result = run_diagnoser(
    audio_dir="./audio",
    output_dir="./output",
    metadata_csv="./metadata.csv"
)

# Generate respiratory sounds
audio = run_generator(
    ref_audio="./reference.wav",
    disease="Asthma",
    out_dir="./generated"
)
```

### Command Line Interface

```bash
# Run diagnosis
resp-agent diagnose --audio_dir ./audio --output_dir ./output --metadata_csv ./metadata.csv

# Run generation
resp-agent generate --ref_audio ./ref.wav --disease Asthma --out_dir ./output

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
