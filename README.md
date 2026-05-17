# <img src="assets/shor.png" width="32" valign="middle"> Towards Direct Evaluation of Harness Optimizers via Priority Ranking

[![PDF](https://img.shields.io/badge/PDF-Preprint-red?logo=arxiv)](assets/Shor.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-informational?logo=huggingface)](https://huggingface.co/datasets/LangAGI-Lab/SHOR)

This work takes a first step towards the **direct evaluation of harness optimizers** via quantifying their **step-level optimization ability** in a **cost- and time-efficient manner**.

## Overview
<p align="center">
  <img src="assets/main_motivation.png" alt="Shor Overview" width="100%"/>
</p>


### Key Features

- **Direct Optimizer Evaluation**: Evaluates harness optimizers directly rather than using target agents' task improvement as proxy
- **Priority Ranking**: Quantifying optimizer ability to prioritize harness components (i.e, prompt, tool, workflow, and memory) that are expected to bring more improvement to the target agent
- **Human-Verified Dataset, SHOR**: Includes 182 curated optimization scenarios collected from real optimization trajectories
- **Multi-Domain Coverage**: Supports SWE-bench Verified, GAIA, Spider 2.0-lite, and τ²-Bench
- **Cost-Efficient Evaluation**: By utilizing SHOR, evaluating harness optimizer via priority ranking is on average 8× cheaper and 17× faster than conventional end-improvement observation from full harness optimization.

### Dataset Statistics

- **SHOR**
  - 182 human-verified harnesses
- **SHOR-Flaw**
  - 122 flawed harnesses

## Quick Start

### 1. Setup

```bash
conda create -n shor python=3.10
conda activate shor
pip install -r requirements.txt  
```

Set the API keys for the providers you plan to use:

```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export GEMINI_API_KEY=your_gemini_api_key
export OPENROUTER_API_KEY=your_openrouter_api_key
export SERPER_API_KEY=your_serper_api_key  # SerpAPI for web search (only needed for the GAIA domain)
export LLM_API_KEY=your_api_key        # optional for Openhands-cli
export LLM_BASE_URL=https://your-proxy.example.com/v1       # optional for Openhands-cli
```

### 2. Data Download

```bash
bash scripts/download_shor_data.sh
```

### 3. Basic Usage

To implement your own coding agent as harness optimizer, follow [build_harness_optimizer.md](docs/build_harness_optimizer.md).

Built-in optimizers for references:

- `openhands_cli`: OpenHands CLI adapter
- `claude_code_cli`: Claude Code CLI adapter
- `codex_cli`: Codex CLI adapter

### Install Optimizer CLIs

If you want to use the built-in optimizer adapters, install the corresponding CLI tools first.

**OpenHands CLI**

Install via `uv` (recommended):
```bash
uv tool install openhands --python 3.12
```

**Claude Code**
```bash
npm install -g @anthropic-ai/claude-code
```

**Codex CLI**
```bash
npm install -g @openai/codex
```

### 4. SHOR Evaluation

```bash
python src/shor/run_shor.py --optimizer your_optimizer_name

# Run only the first 10
python src/shor/run_shor.py --optimizer your_optimizer_name --limit 10

# Run in Parrallel
python src/shor/run_shor.py --optimizer your_optimizer_name --parallel 4
```

### 5. View Results

```bash
python src/shor/eval/evaluate_shor_results.py result/your_optimizer_name
```

## Citation

If you use this repository in your research, please cite:

```bibtex
TBD
```

## Project Structure

```bash
.
├── data/                        # Agent assets including Shor, Shor-flaw
│   ├── gaia/                    # GAIA-domain agent assets
│   └── ...                      # Other domain data
├── scripts/
│   └── download_shor_data.sh    # Script for downloading data
└── src/
    ├── harness_optimizer/       # Harness optimizer interfaces and built-in adapters
    └── shor/                    # SHOR execution, configuration, and evaluation pipeline
```