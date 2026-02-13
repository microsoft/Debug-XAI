# Olmo3 Implementation TODO

## Architecture Analysis
The Olmo3 architecture is a decoder-only transformer similar to Llama and Qwen, but with some specific components:

- **Attention**: `Olmo3Attention`
    - Uses `q_proj`, `k_proj`, `v_proj`, `o_proj`.
    - Distinctive feature: **QK-Norm**. It has `q_norm` and `k_norm` (RMSNorm) applied to Query and Key before the dot product. This is critical for training stability but for LRP, since they are normalizations, we can apply the standard identity rule (via `rms_norm_forward`).
- **MLP**: `Olmo3MLP`
    - Uses `gate_proj`, `up_proj`, `down_proj`.
    - Activation: SiLU.
    - Structure: Gated MLP (SwiGLU variant). Matches `LlamaMLP` and `Qwen3MLP`.
- **Normalization**: `Olmo3RMSNorm`
    - Used for LayerNorms and QK-Norms. Matches standard RMSNorm.

## Patching Strategy
We can leverage existing patching rules from `lxt.efficient.patches`:

1.  **Normalization (`Olmo3RMSNorm`)**:
    - Apply `rms_norm_forward` (Identity Rule).
    - This covers `q_norm`, `k_norm`, and layer norms.

2.  **MLP (`Olmo3MLP`)**:
    - Apply `gated_mlp_forward` for AttnLRP (Identity on Act, Uniform on Element-wise Mult).
    - Apply `cp_gated_mlp_forward` for CP-LRP (Gradient blocking on Gate).

3.  **Attention (`modeling_olmo3`)**:
    - Apply `patch_attention` / `patch_cp_attention`.
    - These wrappers modify gradients on Q, K, V *inside* the attention function.
    - Since `q_norm` and `k_norm` happen *before* the attention function (in terms of module hierarchy, though functionally they feed into it), the gradient modification on Q/K by `patch_attention` will propagate back through the norms. Since Norms are Identity rule (stop gradient on variance), the relevance should flow correctly.
    - `modeling_olmo3` has `eager_attention_forward` and `ALL_ATTENTION_FUNCTIONS` just like Llama/Qwen, so `patch_attention` is compatible.

4.  **Dropout**:
    - Apply `dropout_forward` to disable dropout during "training" mode (required for gradient checkpointing if used).

## Implementation Plan
1.  Create `lxt/efficient/models/olmo3.py`.
2.  Import `Olmo3MLP`, `Olmo3RMSNorm` from `transformers.models.olmo3.modeling_olmo3` (if available) or rely on explicit patching sets.
3.  Define `attnLRP` and `cp_LRP` dictionaries.
4.  Expose in `lxt/efficient/models/__init__.py`.
