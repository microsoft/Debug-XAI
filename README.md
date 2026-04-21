# 🔬 Debug-XAI: Explainable AI for Transformer Debugging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0+-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19-n4e1pB5JSKLWwds3QQlGfT2vcjH73e?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2604.17761-b31b1b.svg)](https://arxiv.org/abs/2604.17761)

🌐 **Project Page:** <https://jzxycsjzy.github.io/Debug-XAI/>

🚀 **Interactive Demo (Colab):** <https://colab.research.google.com/drive/19-n4e1pB5JSKLWwds3QQlGfT2vcjH73e?usp=sharing>

📄 **Paper (arXiv):** <https://arxiv.org/abs/2604.17761>

This repository provides a comprehensive attribution analysis toolkit for Transformer-based models. It combines Layer-wise Relevance Propagation (LRP) with contrastive attribution methods to enable in-depth interpretability analysis and failure debugging of large language models.

## 🌟 Key Features

### 🔍 LRP Extensions on Top of AttnLRP / `lxt`
Our attribution engine reuses [AttnLRP](https://github.com/rachtibat/LRP-eXplains-Transformers) and its `lxt` library as the underlying Input×Gradient LRP runtime — the propagation rules themselves are **not** our contribution. On top of this backbone, Debug-XAI contributes three extensions:

- **Contrastive Attribution Objective**: Failure analysis is cast as attributing the logit difference $\Delta\ell = \ell(t_\text{tgt}) - \ell(t_\text{con})$ between an incorrect target token and a contrast alternative, implemented as first-class backprop modes (`max_logit`, `logit_diff` with `by_topk_avg` / `demean` / `by_ref_token`).
- **Batch-Packed Multi-Target Backpropagation**: Our core method contribution — packing multiple attribution targets into the batch dimension so cross-layer attribution-graph edges are recovered in $O(\lceil n/B \rceil)$ backward passes instead of $O(n)$, making long-context (10k+ token) attribution graphs tractable.
- **Graph Pruning + Coarse-to-Fine Analysis**: Global-threshold and per-layer cumulative-mass edge pruning, connected-subgraph extraction back to the final token, and a hidden-state-level → neuron-level refinement pipeline for long-context failure cases.

Engineering add-ons (extended `lxt/efficient/models/` patches for Qwen3/Olmo3, 4-bit BitsAndBytes + gradient-checkpointing wiring) are bundled for convenience but are not claimed as method contributions.

### 📊 Interactive Circuit Explorer (our main system contribution)
Implemented in `analysis_tool/attnlrp_circuit/backend/` — a FastAPI service that turns raw LRP signals into browsable, quantifiable circuits.

- **Attribution Engine (`core.py`) + Circuit Builder (`circuit.py`)**: Runs a single backward pass and materializes dense token-to-token connection matrices for every layer transition, with automatic Attn/MLP (mid-block) splitting when hooks are available.
- **Cached Layer Transitions (`app.py`)**: Connection matrices are hashed by backprop config + layer set, so slider/UI interactions re-render without recomputing attribution.
- **Graph Metric Suite (`graph_metrics.py`, `metrics.py`, `graph_metrics.md`)**: Layer-wise profiles covering connectivity/sparsity, information flow & Gini concentration, verticality (residual vs. mixing), and temporal dynamics of the circuit.
- **Batch Analysis Harness (`batch_config.py`)**: Declarative configs for sweeping prompts, models, and backprop modes — the basis for the CAIW failure-case studies.
- **Model Loader (`models/`)**: Unified loading of HF models with optional 4-bit quantization, shared between the web UI and batch runs.

### 🎯 Contrastive Attribution Analysis
- **Benchmark Integration**: Pre-configured notebooks for GAIA2, HumanEval, IFEval, and MATH datasets
- **Failure Case Investigation**: Specialized tools for debugging model predictions
- **Heatmap Generation**: LaTeX-based publication-quality attribution visualizations
- **Batch Processing**: Efficient analysis of multiple samples with configurable parameters

## 📁 Repository Structure

```
Debug-XAI/
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
    └── CAIW.pdf                # Preview version of paper CAIW
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.11 or higher
- **CUDA GPU**: Recommended for large models (11GB+ VRAM for 7B models)
- **LaTeX**: Optional, for heatmap generation (install `xelatex` or `pdflatex`)

### Installation

```bash
# Navigate to the analysis tool directory
cd Debug-XAI/analysis_tool

# Create virtual environment (using uv - recommended)
uv venv --python=3.11
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

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

### Usage

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
> Leila Arras, et al. "A Close Look at Decomposition-based XAI-Methods for Transformer Language Models." arXiv preprint [arXiv:2502.15886 (2025)](https://arxiv.org/abs/2502.15886)

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
## 📦 Data

**No data, traces, or prompts are released as part of this repository.** All analysis notebooks and scripts expect locally available benchmark data. To run the code, you must download the benchmarks directly from the official benchmark owners using their original links and licenses:

- **GAIA2**: [https://huggingface.co/datasets/gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- **IFEval**: [https://huggingface.co/datasets/google/IFEval](https://huggingface.co/datasets/google/IFEval)
- **MATH**: [https://huggingface.co/datasets/hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math)
- **HumanEval**: [https://huggingface.co/datasets/openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)

Please review and comply with each benchmark's license and terms of use before downloading or using the data.

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

### Acknowledgment: `lxt` / AttnLRP

The `analysis_tool/lxt/` subtree (both `efficient/` and `explicit/`) is derived from the **LRP-eXplains-Transformers (`lxt`)** library, released under the **BSD 3-Clause License**. We reuse it as the underlying Input×Gradient LRP runtime and contribute additional model patches, quantization-compatible wiring, and the circuit-analysis system built on top. All original copyright notices and the BSD-3-Clause terms from the upstream project are retained in the corresponding source files.

> **AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers**  
> Achtibat et al., ICML 2024. [GitHub](https://github.com/rachtibat/LRP-eXplains-Transformers) · License: [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)

## 🔗 Related Resources

- **Project Page**: <https://jzxycsjzy.github.io/Debug-XAI/>
- **Colab Demo**: <https://colab.research.google.com/drive/19-n4e1pB5JSKLWwds3QQlGfT2vcjH73e?usp=sharing>
- **Full Documentation**: [analysis_tool/README.md](analysis_tool/README.md)
- **Circuit Explorer Guide**: [attnlrp_circuit/README.md](analysis_tool/attnlrp_circuit/README.md)
- **Example Scripts**: [examples/README.md](analysis_tool/examples/README.md)

> **Trademarks**  
> This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.


### 📚 Citation

If you find this toolkit useful, please cite:

```bibtex
@misc{tan2026contrastiveattributionwildinterpretability,
      title={Contrastive Attribution in the Wild: An Interpretability Analysis of LLM Failures on Realistic Benchmarks},
      author={Rongyuan Tan and Jue Zhang and Zhuozhao Li and Qingwei Lin and Saravan Rajmohan and Dongmei Zhang},
      year={2026},
      eprint={2604.17761},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.17761},
}
```