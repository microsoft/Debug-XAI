# Contrastive Attribution in the Wild

**Contrastive Attribution in the Wild** is a comprehensive framework for analyzing and understanding Transformer-based language models through **Layer-wise Relevance Propagation (LRP)** and **contrastive attribution methods**. This repository provides tools for attribution analysis, circuit exploration, and failure case investigation across multiple benchmarks including GAIA2, HumanEval, IFEval, and MATH.

---

## 🌟 Features

- **🔍 Efficient LRP Implementation**: Fast, PyTorch-based Layer-wise Relevance Propagation using automatic differentiation
- **📊 Interactive Circuit Explorer**: Web-based visualization tool for exploring relevance circuits in Transformer models
- **🎯 Contrastive Attribution**: Support for multiple backpropagation strategies including logit difference and reference token comparisons
- **📈 Multi-Benchmark Analysis**: Pre-configured notebooks for analyzing models on GAIA2, HumanEval, IFEval, and MATH datasets
- **⚡ Quantization Support**: 4-bit quantized model support via BitsAndBytes for memory-efficient analysis
- **🎨 Heatmap Visualization**: LaTeX-based attribution heatmap generation

---

## 📁 Repository Structure

```
analysis_tool/
├── .gitignore                 # Git ignore rules
├── .readthedocs.yaml          # ReadTheDocs configuration
├── LICENSE                    # BSD 3-Clause License
├── README.md                  # This file
├── setup.py                   # Package installation script
│
├── lxt/                       # Core LRP library
│   ├── __init__.py
│   ├── utils.py               # Utility functions for visualization
│   ├── efficient/             # Efficient LRP implementation (recommended)
│   └── explicit/              # Explicit LRP implementation (for research)
│
├── examples/                  # Example scripts and outputs
│   ├── README.md              # Examples documentation
│   ├── quantized_qwen3.py     # Qwen3 model attribution example
│   ├── quantized_llama.py     # LLaMA model attribution example
│   ├── quantized_gemma3.py    # Gemma3 model attribution example
│   ├── quantized_qwen2.py     # Qwen2 model attribution example
│   ├── latent_feature_attr_qwen3.py  # Latent feature attribution
│   ├── visualization.py       # Visualization utilities
│   ├── vit_torch.py           # Vision Transformer attribution
│   ├── heatmaps/              # Generated attribution heatmaps
│   └── paper/                 # Paper reproducibility scripts
│
├── attnlrp_circuit/           # Interactive circuit explorer
│   ├── README.md              # Circuit explorer documentation
│   ├── backend_arch.md        # Backend architecture documentation
│   ├── backend/               # FastAPI backend server
│   │   ├── __init__.py
│   │   ├── app.py             # Main API server
│   │   ├── circuit.py         # Circuit computation logic
│   │   ├── core.py            # Core attribution engine
│   │   ├── metrics.py         # Attribution metrics
│   │   ├── graph_metrics.py   # Graph-based metrics
│   │   ├── batch_config.py    # Batch processing configuration
│   │   └── models/            # Model loading utilities
│   ├── frontend/              # Web UI (HTML/CSS/JS)
│   │   ├── index.html         # Main HTML interface
│   │   ├── css/               # Stylesheets
│   │   └── js/                # JavaScript logic
│   └── tests/                 # Unit tests
│       ├── test_api_sanity.py
│       ├── test_backend_sanity.py
│       ├── test_backend_full.py
│       └── test_layer_decomposition.py
│
├── docs/                      # Sphinx documentation
│   ├── Makefile
│   ├── make.bat
│   ├── requirements_sphinx.txt
│   └── source/                # Documentation source files
│
├── tests/                     # Package-level tests
│   ├── test_functional.py
│   ├── test_modules.py
│   └── test_rules.py
│
├── exp/                       # Experimental results and analysis
│
└── *.ipynb                    # Analysis notebooks
    ├── analysis.ipynb
    ├── analysis_MATH.ipynb
    ├── analysis_HumanEval.ipynb
    ├── analysis_IFEval.ipynb
    └── anaysis_checkpoints.ipynb
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (recommended for large models)

### Installation

1. **Clone the repository**
   ```bash
   cd contrastive-attribution-in-the-wild/analysis_tool
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using uv (recommended)
   uv venv --python=3.11
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows

   # Or using standard venv
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the package**
   ```bash
   pip install -e .
   # Or with uv
   uv pip install -e .

## 🛠️ Usage

### 1. Interactive Circuit Explorer

Launch the web-based circuit visualization tool:

```bash
# Start the FastAPI server
uvicorn attnlrp_circuit.backend.app:app --host 0.0.0.0 --port 8000

# Open in browser
# Navigate to http://localhost:8000
```

**Features:**
- Load any Hugging Face Transformer model
- Compute and visualize token-to-token relevance circuits
- Interactive layer-wise analysis
- Support for multiple backpropagation strategies
- Export visualizations

### 2. Running Example Scripts

The `examples/` directory contains ready-to-use scripts for various models:

```bash
cd examples

# Run Qwen3 attribution analysis
python -B quantized_qwen3.py

# Run LLaMA attribution analysis
python -B quantized_llama.py

# Run Gemma3 attribution analysis
python -B quantized_gemma3.py
```

**Configuration Options:**
- `BACKPROP_MODE`: Choose between `"max_logit"` or `"logit_diff"`
- `CONTRAST_RANK`: Rank of the logit to contrast with (for logit_diff mode)
- Quantization settings via `BitsAndBytesConfig`

### 3. Benchmark Analysis

Explore pre-configured Jupyter notebooks for different benchmarks:

```bash
jupyter notebook

# Open one of the analysis notebooks:
# - analysis_MATH.ipynb       (MATH benchmark)
# - analysis_HumanEval.ipynb  (HumanEval benchmark)
# - analysis_IFEval.ipynb     (IFEval benchmark)
# - metric.ipynb              (General attribution metrics)
```

### 4. Custom Attribution Analysis

Create your own attribution pipeline:

```python
from lxt.efficient import monkey_patch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Patch your model architecture
from transformers.models.llama import modeling_llama
monkey_patch(modeling_llama)

# 2. Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 3. Prepare input
text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 4. Forward + backward for attribution
outputs = model(**inputs)
logits = outputs.logits[0, -1]

# Choose attribution target
target = logits[target_token_id] - logits[contrast_token_id]  # Contrastive
# OR
target = logits.max()  # Max logit

target.backward()

# 5. Extract relevance
relevance = inputs['input_ids'].grad.cpu().numpy()
```

---

## 📊 Supported Models

The framework supports various Transformer architectures:

- ✅ **Qwen/Qwen3** (tested: 0.6B, 1.7B, 4B)
- ✅ **LLaMA/Llama-2**
- ✅ **Gemma/Gemma3**
- ✅ **GPT-2**
- ✅ **OLMo**
- ⚙️ Other Hugging Face Transformers (may require custom patching)

