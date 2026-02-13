from .base import LayerDecomposer
from .qwen import QwenDecomposer

# Registry mapping model type string (or class) to Decomposer
DECOMPOSER_REGISTRY = {
    # Keys should match what we expect in model config or name logic
    "qwen2": QwenDecomposer,
    "qwen3": QwenDecomposer,
    "qwen": QwenDecomposer, # Generic fallback
    "llama": QwenDecomposer, # Llama usually identical structure (PreNorm, RMS, MLP)
}

def get_decomposer(model_name_or_obj) -> LayerDecomposer:
    """
    Factory to return appropriate decomposer.
    """
    # Simple logic based on string for now
    name = str(model_name_or_obj).lower()
    
    if "qwen" in name:
        return QwenDecomposer()
    if "llama" in name:
        return QwenDecomposer() # Re-use for now as structure is same
        
    # Default fallback (hope compatibility)
    print(f"Warning: No specific decomposer for {name}. Using Qwen/Llama default.")
    return QwenDecomposer()
