import os
import json
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3
from transformers import BitsAndBytesConfig

from lxt.efficient import monkey_patch
from visualization import save_heatmap_matplotlib, create_html_heatmap

monkey_patch(modeling_qwen3, verbose=True)

# ============== Configuration ==============
# Backprop mode options:
#   - "max_logit": backprop on the maximum logit (default)
#   - "logit_diff": backprop on the difference between top logit and a logit at a given rank
# BACKPROP_MODE = "max_logit"  
BACKPROP_MODE = "logit_diff"  
CONTRAST_RANK = 7  # rank of the logit to contrast with (2 = second highest, 3 = third, etc.)

# optional 4bit quantization 
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent overflow in gradients
)


def hook_hidden_activation(module, input, output):
    if isinstance(output, tuple):
        output = output[0]

    # save the activation and make sure the gradient is also saved in the .grad attribute after the backward pass
    module.output = output
    module.output.retain_grad() if module.output.requires_grad else None

caiw_dir = "/mnt/caiw"
fp = r"exp/quick_test/traces/quick_test_1_prompt_idx_1.json" # CONTRAST_RANK = 7
# fp = r"exp/quick_test/traces/quick_test_1_prompt_idx_4.json"
# fp = r"exp/quick_test/traces/quick_test_2.json"
# fp = r"exp/quick_test/traces/quick_test_2_prompt_idx_2.json"
# fp = r"exp/quick_test/traces/quick_test_2_prompt_idx_2_model_2.json" # correct
# fp = r"exp/quick_test/traces/quick_test_3.json"

with open(os.path.join(caiw_dir, fp), "r") as f:
    sample = json.load(f)
prompt = f"{sample['query']} {sample['prediction_origin']}" if sample['query'] else sample['prediction_origin']
# prompt = f"I have 5 cats and 3 dogs. My cats love to play with my"

# path = 'Qwen/Qwen3-0.6B'
# model_name = "qwen3_0.6b"
path = sample['metadata']["model"]
model_name = path.split('/')[-1]

print(f"Loading model from {path}...")

model = modeling_qwen3.Qwen3ForCausalLM.from_pretrained(path, device_map='cuda', torch_dtype=torch.bfloat16, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(path)

# optional gradient checkpointing to save memory (2x forward pass)
model.train()
model.gradient_checkpointing_enable()

# deactive gradients on parameters to save memory
for param in model.parameters():
    param.requires_grad = False

# apply hooks
for layer in model.model.layers:
    layer.register_forward_hook(hook_hidden_activation)

# forward & backward pass
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
last_logits = output_logits[0, -1, :]

# Get sorted logits and indices
sorted_logits, sorted_indices = torch.sort(last_logits, dim=-1, descending=True)

# print top 5 logits and the corresponding tokens
print("\nTop 10 logits:")
for i in range(10):
    token_id = sorted_indices[i].item()
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    logit_val = sorted_logits[i].item()
    print(f"  Token: {token:15} | Logit: {logit_val:.4f}")

if BACKPROP_MODE == "max_logit":
    # Backprop on the maximum logit
    target_logit = sorted_logits[0]
    target_token = tokenizer.convert_ids_to_tokens(sorted_indices[:, 0])
    print(f"Prediction (top-1): {target_token}")
    print(f"Top logit value: {sorted_logits[0].item():.4f}")
    mode_desc = "Max Logit"
    
elif BACKPROP_MODE == "logit_diff":
    # Backprop on the difference between top logit and logit at CONTRAST_RANK
    top_logit = sorted_logits[0]
    contrast_logit = sorted_logits[CONTRAST_RANK - 1]
    target_logit = top_logit - contrast_logit
    
    top_token = tokenizer.convert_ids_to_tokens([sorted_indices[0]])
    contrast_token = tokenizer.convert_ids_to_tokens([sorted_indices[CONTRAST_RANK - 1]])
    
    print(f"Top-1 prediction: {top_token} (logit: {top_logit.item():.4f})")
    print(f"Rank-{CONTRAST_RANK} prediction: {contrast_token} (logit: {contrast_logit.item():.4f})")
    print(f"Logit difference: {target_logit.item():.4f}")
    mode_desc = f"Logit Diff (Top-1 vs Rank-{CONTRAST_RANK})"
else:
    raise ValueError(f"Unknown BACKPROP_MODE: {BACKPROP_MODE}")

target_logit.backward()

# trace relevance through layers
relevance_trace = []
for layer in model.model.layers:
    relevance = (layer.output * layer.output.grad).float().sum(-1).detach().cpu()
    # normalize relevance at each layer between -1 and 1
    relevance = relevance / relevance.abs().max()
    relevance_trace.append(relevance)

relevance_trace = torch.cat(relevance_trace, dim=0)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
relevance_data = relevance_trace.numpy().T

# Generate both visualizations
title = f"Latent Relevance Trace ({mode_desc}) - {model_name}"
save_heatmap_matplotlib(relevance_data, tokens, (20, 10), title, 'latent_rel_trace.png')
create_html_heatmap(relevance_data, tokens, title, 'latent_rel_trace.html')

print(f"\n✓ Visualizations complete!")
print(f"  - PNG: latent_rel_trace.png")
print(f"  - HTML (interactive): latent_rel_trace.html")