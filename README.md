# Resp-Agent

> An intelligent respiratory sound analysis and generation system powered by DeepSeek API

[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](./README_zh-CN.md)

## 📖 Overview

**Resp-Agent** is an AI agent system built on the ReAct (Reasoning + Acting) architecture, designed for respiratory sound medical diagnosis research. The system uses DeepSeek LLM as the "Thinker" to coordinate two core tools for a closed-loop "diagnose–reflect–plan–act" workflow:

- **Diagnoser**: Analyzes respiratory audio and EHR data, outputs detailed diagnosis reports
- **Generator**: Synthesizes high-fidelity respiratory sound samples for data augmentation

## 📁 Project Structure

```
Agent/
├── Resp_agent_chinese.py    # Chinese version of the Agent
├── Resp_agent_english.py    # English version of the Agent
├── Diagnoser/               # Diagnoser module
│   ├── diagnoser_pipeline.py    # Main diagnosis pipeline
│   ├── config.yaml              # Model configuration
│   ├── checkpoints/             # Model checkpoints
│   ├── pretrained_models/       # Pretrained models
│   └── example/                 # Example data (audio + metadata)
└── Generator/               # Generator module
    ├── generator_pipeline.py    # Main generation pipeline
    ├── models.py                # Model definitions
    ├── config.yaml              # Model configuration
    ├── checkpoints/             # Model checkpoints
    ├── pretrained_models/       # Pretrained models
    └── wav/                     # Reference audio directory
```

## ⚙️ Setup

### Option A: Install as Package (Recommended)

**1. Create a clean conda environment:**
```bash
conda create -n breath python=3.10 -y
conda activate breath
```

**2. Install PyTorch with CUDA support:**
```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

**3. Install resp-agent:**
```bash
pip install resp-agent
```

Then use directly:
```bash
resp-agent chat --lang en   # English interactive agent
resp-agent chat --lang zh   # Chinese interactive agent
resp-agent diagnose --audio_dir ./audio --output_dir ./output --metadata_csv ./metadata.csv
resp-agent generate --ref_audio ./ref.wav --disease Asthma --out_dir ./output
```

### Option B: Clone Repository

```bash
git clone https://github.com/zpforlove/Resp-Agent.git
cd Resp-Agent
conda create -n breath python=3.10 -y
conda activate breath
pip install -e .
```

### 2. Configure DeepSeek API Key

> [!IMPORTANT]
> **This is a required step to run the Agent!**

1. Get your API key from [DeepSeek Platform](https://platform.deepseek.com/)
2. Set the environment variable:

**Linux/macOS:**
```bash
export DEEPSEEK_API_KEY='your-api-key-here'

# For permanent setup, add to ~/.bashrc or ~/.zshrc
echo "export DEEPSEEK_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):**
```powershell
$env:DEEPSEEK_API_KEY = "your-api-key-here"
```

### 3. Verify Configuration

```bash
python -c "import os; print('API Key configured' if os.environ.get('DEEPSEEK_API_KEY') else 'API Key not found')"
```

## 🚀 Quick Start

```bash
cd /path/to/Agent

# English version
python Resp_agent_english.py

# Chinese version
python Resp_agent_chinese.py
```

## 📋 Usage Guide

### Task Types

| Type | Description | Example |
|------|-------------|---------|
| **Simple** | Single tool call | `Diagnose respiratory sounds with audio_dir=./Diagnoser/example/audio` |
| **Advanced** | Closed-loop iteration | `Run a full iteration: diagnose, then generate targeted data based on weaknesses` |

### Tool Call Formats

**Diagnoser:**
```
[Call:Diagnoser] Diagnose respiratory sounds with audio_dir=<path>, output_dir=<path>, metadata_csv=<path>
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `audio_dir` | Audio files directory | `./Diagnoser/example/audio` |
| `output_dir` | Output directory | `./Diagnoser/output_diagnose` |
| `metadata_csv` | EHR CSV file path | `./Diagnoser/example/combined_metadata.csv` |

**Generator:**
```
[Call:Generator] Generate <disease> respiratory audio with ref_audio=<path.wav>, disease=<disease>, out_dir=<path>
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ref_audio` | Reference audio (.wav) | `./Generator/wav/reference_audio.wav` |
| `disease` | Target disease label | `Asthma` |
| `out_dir` | Output directory | `./Generator/output_generate` |

### Interactive Commands

| Command | Function |
|---------|----------|
| `help` / `examples` | Show task templates |
| `quit` / `exit` | Exit the Agent |

## 🔄 Workflow

```mermaid
graph TD
    A[User Input] --> B{Task Type?}
    B -->|Advanced| C[Step 1: Diagnose]
    C --> D[Call Diagnoser]
    D --> E[Step 2: Reflect]
    E --> F[Analyze Report<br>Identify Weaknesses]
    F --> G[Step 3: Plan & Act]
    G --> H[Call Generator<br>Generate Targeted Data]
    H --> I[Step 4: Summarize]
    I --> J[Final Report]
    B -->|Simple| K[Call Single Tool]
    K --> L[Output Result]
```

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `DEEPSEEK_API_KEY not set` | Configure the environment variable as shown above |
| `API connection failed` | Check API key validity and network connectivity |
| `Script not found` | Run from the `Agent/` directory |

## 🔧 Technical Details

### DeepSeek API Configuration

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
```

- **Model**: `deepseek-chat`
- **Temperature**: `0.7`
- **Top-P**: `0.9`

### Supported Disease Types

- Asthma
- COVID-19
- COPD
- Other respiratory conditions (based on training data)

## 🙏 Acknowledgements

This project is built upon the following excellent open-source projects:

- [BEATs](https://github.com/microsoft/unilm/tree/master/beats) - Audio Pre-Training with Acoustic Tokenizers (Microsoft)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) - Reasoning model powering the Thinker agent
- [Longformer](https://github.com/allenai/longformer) - Long document transformer for EHR processing
- [Qwen3](https://github.com/QwenLM/Qwen3) - Large language model foundation

We sincerely thank the developers and contributors of these projects!

## 📄 License

This project is for academic research purposes only.

## 📧 Contact

For questions or feedback, please contact:

**Email**: [pzhang176@connect.hkust-gz.edu.cn](mailto:pzhang176@connect.hkust-gz.edu.cn)

---

*Last updated: February 2026*
