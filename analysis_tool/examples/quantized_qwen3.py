import os
import json

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3
from transformers import BitsAndBytesConfig

from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap, clean_tokens

# modify the Qwen3 module to compute LRP in the backward pass
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


# prompt = """Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
# Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""
# prompt = "The capital of France is"
# prompt = "<|im_start|>system\nYou are an knowledge expert, you are supposed to answer the multi-choice question to put your final answer within \\boxed\{\}.<|im_end|>\n<|im_start|>user\nQ: What is the sign of the covenant for Jewish males?\nOptions are:\n(A): Fasting on Yom Kippur\n(B): Lighting Shabbat candles\n(C): The rainbow\n(D): Circumcision\n(E): The Torah\n(F): Bar mitzvah\n(G): Keeping kosher\n(H): Wearing a kippah\n(I): A son\n(J): The Star of David\n\n<|im_end|>\n<|im_start|>thought\n<think>\nOkay, let's see. The question is asking about the sign of the covenant for Jewish males. The options are A through J. Hmm, I need to recall what the covenant means in Judaism.\n\nFirst, I remember that the covenant is a religious agreement between God and the Israelites. The covenant is part of the Torah, right? So the answer might be related to the Torah. Let me check the options again. Option E is \"The Torah.\" That seems relevant because the covenant is based on the Torah. \n\nWait, but let me make sure. The covenant is between God and the people, and the Torah is the source of that covenant. So the sign of the covenant would be the Torah itself. So the answer should be"

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

# path = 'Qwen/Qwen3-0.6B'
# model_name = "qwen3_0.6b"
path = sample['metadata']["model"]
model_name = path.split('/')[-1]

print(f"Loading model from {path}...")

model = modeling_qwen3.Qwen3ForCausalLM.from_pretrained(path, device_map='cuda', torch_dtype=torch.bfloat16, quantization_config=quantization_config)

# optional gradient checkpointing to save memory (2x forward pass)
model.train()
model.gradient_checkpointing_enable()

# deactive gradients on parameters to save memory
for param in model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(path)


# get input embeddings so that we can compute gradients w.r.t. input embeddings
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

# inference and get the maximum logit at the last position (we can also explain other tokens)
output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
last_logits = output_logits[0, -1, :]

# Get sorted logits and indices
sorted_logits, sorted_indices = torch.sort(last_logits, dim=-1, descending=True)

# Display top-k predictions
print("\nTop-10 predictions:")
for rank in range(10):
    tid = sorted_indices[rank].item()
    val = sorted_logits[rank].item()
    token_str = tokenizer.decode([tid])
    print(f"  Top{rank+1}: id={tid:>6}  logit={val:>8.4f}  token={repr(token_str)}")

# Backward pass based on selected mode
if BACKPROP_MODE == "max_logit":
    # Backprop on the maximum logit
    target_logit = sorted_logits[0]
    target_token = tokenizer.decode([sorted_indices[0].item()])
    print(f"\nBackprop mode: Max Logit")
    print(f"Target token: {repr(target_token)} (logit: {target_logit.item():.4f})")
    mode_desc = "Max Logit"
    
elif BACKPROP_MODE == "logit_diff":
    # Backprop on the difference between top logit and logit at CONTRAST_RANK
    top_logit = sorted_logits[0]
    contrast_logit = sorted_logits[CONTRAST_RANK - 1]
    target_logit = top_logit - contrast_logit
    
    top_token = tokenizer.decode([sorted_indices[0].item()])
    contrast_token = tokenizer.decode([sorted_indices[CONTRAST_RANK - 1].item()])
    
    print(f"\nBackprop mode: Logit Difference")
    print(f"Top-1: {repr(top_token)} (logit: {top_logit.item():.4f})")
    print(f"Rank-{CONTRAST_RANK}: {repr(contrast_token)} (logit: {contrast_logit.item():.4f})")
    print(f"Logit difference: {target_logit.item():.4f}")
    mode_desc = f"Logit Diff (Top-1 vs Rank-{CONTRAST_RANK})"
else:
    raise ValueError(f"Unknown BACKPROP_MODE: {BACKPROP_MODE}")

# This initiates the LRP computation through the network
target_logit.backward()

# obtain relevance by computing Input * Gradient
relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0] # cast to float32 before summation for higher precision

# normalize relevance between [-1, 1] for plotting
relevance = relevance / relevance.abs().max()

# remove special characters from token strings and plot the heatmap
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
tokens = clean_tokens(tokens)
print(tokens)

# Use mode description in filenames
mode_suffix = "max" if BACKPROP_MODE == "max_logit" else f"diff_r{CONTRAST_RANK}"
pdf_heatmap(tokens, relevance, path=f'{model_name}_heatmap_{mode_suffix}.pdf', backend='xelatex') # backend='xelatex' supports more characters

# plot again without first token, because it receives large relevance values overshadowing the rest
pdf_heatmap(tokens[1:], relevance[1:] / relevance[1:].max(), path=f'{model_name}_heatmap_{mode_suffix}_wo_first.pdf', backend='xelatex')

print(f"\n✓ Heatmaps saved!")
print(f"  - {model_name}_heatmap_{mode_suffix}.pdf")
print(f"  - {model_name}_heatmap_{mode_suffix}_wo_first.pdf")