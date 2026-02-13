import torch

# Configuration for Batch Chunk Sizes based on Model Size and Dtype
# Reference: 4B model with bf16 uses batch_chunk_size = 64

# Mapping: (Min_Params_Billions, Max_Params_Billions) -> Recommended Batch Size (BF16/FP16)
BATCH_CHUNK_MAPPING = {
    (0.0, 1.0): 32,   # For 0.6B and similar
    (1.0, 3.0): 128,   # For 1.7B, 2B, 3B
    (3.0, 6.0): 64,    # For 4B, 6B
    (6.0, 12.0): 32,   # For 7B, 8B, 10B
    (12.0, 25.0): 16,  # For 14B, 20B
    (25.0, 1000.0): 8  # For 32B+, 70B
}

def get_batch_chunk_size(model_params_count, model_dtype):
    """
    Determine appropriate batch chunk size based on parameter count and dtype.
    
    Args:
        model_params_count (int): Total number of parameters in the model.
        model_dtype (torch.dtype): The data type used for computation (activations).
        
    Returns:
        int: Recommended batch chunk size.
    """
    # Convert to Billions
    params_billions = model_params_count / 1e9
    
    # Default fallback
    chunk_size = 32
    
    # Lookup in mapping
    for (min_b, max_b), size in BATCH_CHUNK_MAPPING.items():
        if min_b <= params_billions < max_b:
            chunk_size = size
            break
            
    # Scale by Dtype
    # If using float32, activations take 2x memory compared to bf16/fp16
    if model_dtype == torch.float32:
        chunk_size = max(1, chunk_size // 2)
        
    return chunk_size

