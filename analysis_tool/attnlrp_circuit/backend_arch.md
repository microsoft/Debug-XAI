# AttnLRP Backend Architecture

This document describes the current architecture of the backend system implemented in `attnlrp_circuit/backend`.

## Core Components

### 1. Model Manager (`backend/models/`)
Responsible for loading and wrapping the model.
-   **Structure**: Uses a Factory pattern via `backend/models/factory.py` and `backend/models/manager.py`.
-   **Model Wrappers (`backend/models/qwen.py`, etc.)**: Implement `LayerDecomposer` interface to handle architectural differences (e.g., finding specific Attention/MLP modules).
-   **Decomposition**: Provides the logic to split a Transformer layer into two parts:
    -   **Part 1**: Post-Norm $\to$ Attention $\to$ Residual Add (Output: `resid_mid`)
    -   **Part 2**: Post-Attn-Norm $\to$ MLP $\to$ Residual Add (Output: `resid_post`)

### 2. Attribution Engine (`backend/core.py`)
the central engine that manages the model state, forward passes, and gradient-based attribution.

#### Key Mechanisms:
-   **Hooking System**:
    -   **`resid_post`**: Standard hook on layer output (`_hook_hidden_activation`).
    -   **`resid_mid` (New)**: Optional hook (`capture_mid=True`) on the `post_attention_layernorm` (or equivalent) to capture the intermediate residual stream state between Attention and MLP blocks (`_hook_mid_activation`).
    -   Enables `retain_grad()` on these outputs to allow gradient flow analysis.

-   **Logit Computation (`compute_logits`)**:
    -   Runs a standard forward pass.
    -   Captures `top-k` logits and allows appending special tokens (BOS).
    -   **Fine-Grained Mode**: Accepts `capture_mid` flag to enable finding the decomposition structure and hooking intermediate acts.

-   **Backward Pass (`run_backward_pass`)**:
    -   Computes a scalar target score based on `backprop_config` (e.g., `max_logit`, `logit_diff`).
    -   Backpropagates from this score to populate gradients in the cached hidden states.

-   **Connection Matrix Computation (`compute_connection_matrix_gen`)**:
    -   Computes the dense interaction matrix (Relevance) between a `source` node and a `target` node.
    -   **Flexible Nodes**: Source/Target can be an integer (Layer End) or a tuple `(layer_idx, 'mid')` (Middle of Layer).
    -   **Partial Forward Paths**: Uses `LayerDecomposer` to execute specific sub-blocks (`part1` or `part2`) depending on the path segment (e.g., `mid_L5` $\to$ `post_L5` runs only the MLP block).
    -   **Algorithm**:
        1.  Start with the output of the source node (resid_post or resid_mid).
        2.  Run partial forward passes (`_forward_part1`/`_forward_part2`) through intermediate steps.
        3.  Compute gradients w.r.t. the source activation.
        4.  Relevance = `(Gradient_at_Source * Activation_at_Source).sum(-1)`.

### 3. Circuit Analyzer (`backend/circuit.py`)
A higher-level orchestration class that:
-   **Auto-Expansion**: Detects if `mid` activations are present. If so, automatically expands a user-requested layer list (e.g., `[5, 6]`) into a fine-grained sequence (`[5_mid, 5_post, 6_mid, 6_post]`).
-   Iterates through node pairs to compute full circuit connectivity.
-   Constructs a `NetworkX` graph from the dense matrices.

## Data Flow

1.  **Initialization**: User loads a model via `ModelManager`.
2.  **Forward**: `AttributionEngine.compute_logits(prompt, capture_mid=True/False)` runs the model.
    -   If `capture_mid=True`, `resid_mid` tensors are stored on the model layers.
3.  **Target Selection**: User selects a target token and strategy.
4.  **Backward**: `AttributionEngine.run_backward_pass()` populates gradients.
5.  **Graph Construction**:
    -   `CircuitAnalyzer` determines the sequence of nodes (coarse or fine-grained).
    -   Calls `engine.compute_connection_matrix_gen` for each transition.
    -   Prunes and returns the graph structure.

## Current Limitations

-   **Memory Usage**: `compute_connection_matrices` can be memory intensive, though mitigated by chunking.
-   **Model Support**: Fine-grained decomposition requires a specific `LayerDecomposer` implementation for each model family (currently supported: Qwen).
