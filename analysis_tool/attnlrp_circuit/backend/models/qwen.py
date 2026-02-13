import torch.nn as nn
from .base import LayerDecomposer

class QwenDecomposer(LayerDecomposer):
    """
    Decomposer for Qwen2/Qwen3 models.
    Assumes Pre-Norm architecture:
    Part 1: x -> InputNorm -> Attn -> Add (resid_mid)
    Part 2: resid_mid -> PostAttnNorm -> MLP -> Add (resid_post)
    """

    def get_mid_activation_module(self, layer_module):
        # Qwen models usually have 'post_attention_layernorm'
        return getattr(layer_module, "post_attention_layernorm", None)

    def forward_part1(self, layer_module, hidden_states, position_embeddings=None, attention_mask=None):
        norm = getattr(layer_module, "input_layernorm", None)
        # Try both generic 'self_attn' and Qwen specific naming if needed, 
        # though HF implementation usually maps to 'self_attn'
        attn = getattr(layer_module, "self_attn", None)
        
        if not norm or not attn:
            # Fallback for some versions or wrapped modules
            # Check named children
            children = dict(layer_module.named_modules())
            attn = attn or children.get("self_attn")
            
        if not norm or not attn:
             raise AttributeError(f"QwenDecomposer: Layer module {type(layer_module)} missing input_layernorm or self_attn")

        # Norm
        norm_out = norm(hidden_states)
        
        # Attn
        if position_embeddings is not None:
             # Qwen/Llama usually accept position_embeddings
             attn_out = attn(norm_out, position_embeddings=position_embeddings, attention_mask=attention_mask)
        else:
             attn_out = attn(norm_out, attention_mask=attention_mask)
             
        if isinstance(attn_out, tuple): attn_out = attn_out[0]
             
        return hidden_states + attn_out

    def forward_part2(self, layer_module, hidden_states):
        norm = getattr(layer_module, "post_attention_layernorm", None)
        mlp = getattr(layer_module, "mlp", None)
        
        if not norm or not mlp:
             raise AttributeError("QwenDecomposer: Layer module missing post_attention_layernorm or mlp")
             
        norm_out = norm(hidden_states)
        mlp_out = mlp(norm_out)
        
        return hidden_states + mlp_out
