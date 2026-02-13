# 🔬 Debug-XAI: Explainable AI for Transformer Debugging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+-ee4c2c.svg)](https://pytorch.org/)

This repository provides a comprehensive attribution analysis toolkit for Transformer-based models, supporting the paper **[Contrastive Attribution in the Wild (CAIW)](doc/CAIW.pdf)**. It combines Layer-wise Relevance Propagation (LRP) with contrastive attribution methods to enable in-depth interpretability analysis and failure debugging of large language models.

## 🌟 Key Features

### 🔍 Advanced LRP Implementation
- **Efficient AutoGrad-based LRP**: PyTorch automatic differentiation framework for fast attribution computation
- **Dual Implementation Modes**:
  - `lxt/efficient/`: Production-ready implementation using Input×Gradient framework (recommended)
  - `lxt/explicit/`: Research implementation for detailed mathematical exploration
- **Multi-Model Support**: Pre-configured patches for Qwen3， LLaMA, Olmo3, etc.
- **Memory-Efficient**: 4-bit quantization support via BitsAndBytes with gradient checkpointing

### 📊 Interactive Circuit Explorer
- **Web-based Visualization**: FastAPI backend + HTML/CSS/JS frontend for real-time exploration
- **Layer-wise Analysis**: Visualize token-to-token relevance circuits between any two layers
- **Multiple Backprop Strategies**:
  - Max Logit: Standard attribution to highest-probability token
  - Logit Difference: Contrastive attribution (top vs. reference tokens)
  - Reference Token Comparison: User-defined contrast targets
- **Graph Metrics**: Compute attribution statistics and circuit topology metrics

### 🎯 Contrastive Attribution Analysis
- **Benchmark Integration**: Pre-configured notebooks for GAIA2, HumanEval, IFEval, and MATH datasets
- **Failure Case Investigation**: Specialized tools for debugging model predictions
- **Heatmap Generation**: LaTeX-based publication-quality attribution visualizations
- **Batch Processing**: Efficient analysis of multiple samples with configurable parameters

## 📁 Repository Structure

```
Debug-XAI-main/
├── analysis_tool/              # Main attribution analysis toolkit
│   ├── lxt/                    # Core LRP library (LRP-eXplains-Transformers)
│   │   ├── efficient/          # Efficient LRP implementation (recommended)
│   │   │   ├── core.py         # Monkey-patching framework
│   │   │   ├── rules.py        # LRP propagation rules
│   │   │   ├── patches.py      # Layer-specific patches
│   │   │   └── models/         # Model-specific configurations
│   │   │       ├── qwen2.py, qwen3.py, llama.py, gemma3.py
│   │   │       └── bert.py, gpt2.py, olmo3.py, vit_torch.py
│   │   ├── explicit/           # Explicit LRP implementation (research)
│   │   └── utils.py            # Visualization utilities (heatmaps, token cleaning)
│   │
│   ├── attnlrp_circuit/        # Interactive circuit explorer
│   │   ├── backend/            # FastAPI server
│   │   │   ├── app.py          # Main API endpoints
│   │   │   ├── core.py         # Attribution engine
│   │   │   ├── circuit.py      # Circuit computation logic
│   │   │   ├── metrics.py      # Attribution metrics
│   │   │   ├── graph_metrics.py # Graph-based analysis
│   │   │   └── models/         # Model loading utilities
│   │   ├── frontend/           # Web UI
│   │   │   ├── index.html      # Main interface
│   │   │   ├── css/style.css   # Styling
│   │   │   └── js/main.js      # Interactive visualization logic
│   │   └── tests/              # API and backend tests
│   │
│   ├── examples/               # Example scripts and outputs
│   │   ├── quantized_qwen3.py  # Qwen3 attribution example
│   │   ├── quantized_llama.py  # LLaMA attribution example
│   │   ├── latent_feature_attr_qwen3.py # Latent feature analysis
│   │   ├── visualization.py    # Heatmap generation utilities
│   │   └── paper/              # Reproducibility scripts
│   │
│   ├── docs/                   # Sphinx documentation
│   ├── tests/                  # Unit tests for lxt package
│   ├── setup.py                # Package installation
│   └── *.ipynb                 # Analysis notebooks
│       ├── analysis_MATH.ipynb
│       ├── analysis_HumanEval.ipynb
│       └── analysis_IFEval.ipynb
│
└── doc/                        # Paper and documentation
    └── CAIW.pdf                # Perview version of paper CAIW
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.11 or higher
- **CUDA GPU**: Recommended for large models (11GB+ VRAM for 7B models)
- **LaTeX**: Optional, for heatmap generation (install `xelatex` or `pdflatex`)

### Installation

```bash
# Navigate to the analysis tool directory
cd Debug-XAI-main/analysis_tool

# Create virtual environment (using uv - recommended)
uv venv --python=3.11
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install the package
pip install -e .
# Or with uv
uv pip install -e .
```

**Dependencies** (automatically installed):
- `torch>=2.9.0`
- `transformers>=4.57.3`
- `accelerate`
- `bitsandbytes` (for 4-bit quantization)
- `zennit` (LRP library)
- `matplotlib`, `tabulate`
- `open_clip_torch` (for vision models)

#### 1. Interactive Circuit Explorer

```bash
# Start the web server
uvicorn attnlrp_circuit.backend.app:app --host 0.0.0.0 --port 8000

# Open browser and navigate to:
# http://localhost:8000
```

**Web Interface Features**:
1. **Model Loading**: Select any Hugging Face model with optional 4-bit quantization
2. **Prompt Input**: Enter text and compute top-k predictions
3. **Backprop Configuration**:
   - Max Logit: Attribute to highest probability token
   - Logit Difference: Contrastive attribution (top vs. reference)
4. **Circuit Visualization**: Interactive canvas showing token-to-token relevance
5. **Layer Selection**: Choose source and target layers for analysis
6. **Export**: Download visualizations and attribution data

## 📖 Documentation

### LRP Implementation Details

This toolkit implements **Layer-wise Relevance Propagation (LRP)** using an efficient Input×Gradient framework inspired by:

> **A Close Look at Decomposition-based XAI-Methods for Transformer Language Models**  
> Leila Arras, et al. "A Close Look at Decomposition-based XAI-Methods for Transformer Language Models。" arXiv preprint [arXiv:2502.15886 (2025)](https://arxiv.org/abs/2502.15886)

### Backpropagation Strategies

1. **Max Logit** (`mode="max_logit"`):
   - Standard attribution to the highest-probability token
   - Useful for understanding confident predictions

2. **Logit Difference** (`mode="logit_diff"`):
   - Contrastive attribution: top logit vs. reference logit
   - Strategies:
     - `by_topk_avg`: Contrast with average of top-k tokens
     - `demean`: Contrast with mean of all logits
     - `by_ref_token`: Contrast with specific token ID
   - **Example**: Explain why model predicts "Paris" over "London"

### Example Notebooks

- **[analysis_MATH.ipynb](analysis_tool/analysis_MATH.ipynb)**: Mathematical reasoning failure analysis
- **[analysis_HumanEval.ipynb](analysis_tool/analysis_HumanEval.ipynb)**: Code generation debugging
- **[analysis_IFEval.ipynb](analysis_tool/analysis_IFEval.ipynb)**: Instruction following analysis
- **[analysis.ipynb](analysis_tool/analysis.ipynb)**: General attribution examples

## 📚 Advanced Features

### Custom Model Integration

To add support for a new model architecture:

1. Create model-specific patch file in `lxt/efficient/models/your_model.py`
2. Implement patch map following existing examples (e.g., `qwen3.py`)
3. Register in `lxt/efficient/models/__init__.py`

```python
# Example: lxt/efficient/models/your_model.py
from lxt.efficient.patches import patch_attention, patch_mlp

def get_qwen3_map(module):
    return {
        module.Qwen3Attention: patch_attention,
        module.Qwen3MLP: patch_mlp,
        # Add more layer types as needed
    }
```
## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for contribution**:
- New model architectures (Mistral, Phi, etc.)
- Additional visualization methods
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The `lxt` library components are based on:
> **AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers**  
> ICML 2024. [GitHub](https://github.com/rachtibat/LRP-eXplains-Transformers)  

## 🔗 Related Resources

- **Full Documentation**: [analysis_tool/README.md](analysis_tool/README.md)
- **Circuit Explorer Guide**: [attnlrp_circuit/README.md](analysis_tool/attnlrp_circuit/README.md)
- **Example Scripts**: [examples/README.md](analysis_tool/examples/README.md)