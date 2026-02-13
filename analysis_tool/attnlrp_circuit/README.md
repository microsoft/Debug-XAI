# AttnLRP Circuit Explorer

This is an interactive tool for exploring relevance circuits in Transformer models.

## Structure

- **backend/**: Python FastAPI server handling model loading, attribution, and circuit logic.
- **frontend/**: HTML/JS/CSS user interface.

## Quick Start

1. **Install Dependencies**
   Ensure you have the required packages:
   ```bash
   pip install fastapi uvicorn
   # Plus torch, transformers, and lxt as used in the notebooks
   ```

2. **Run the Server**
   From the root of the repo (where `attnlrp_circuit` folder is located):
   ```bash
   uvicorn attnlrp_circuit.backend.app:app --host 0.0.0.0 --port 8000
   ```

3. **Access the Interface**
   Open your browser and navigate to:
   [http://localhost:8000/](http://localhost:8000/)

## Features

- **Model Loading**: Supports Qwen3 and generic HF models with optional 4-bit quantization.
- **Logit Analysis**: Compute top-k logits and select contrast/reference tokens.
- **Circuit Visualization**: Visualize token-to-token relevance between any two layers using interactive canvas.
- **Backprop Strategies**:
  - Max Logit
  - Logit Difference (Average Top-K, Demean, Specific Reference Token).
