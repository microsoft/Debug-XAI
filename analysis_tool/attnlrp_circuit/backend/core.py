import torch
import numpy as np
from .batch_config import get_batch_chunk_size

class AttributionEngine:
    def __init__(self, model_manager):
        self.manager = model_manager
        self.hook_handles = []
        self.outputs = None
        self.input_ids = None
        
    def _parse_node(self, node):
        if isinstance(node, int):
            return node, 'post'
        if isinstance(node, (tuple, list)) and len(node) == 2:
            return node[0], node[1]
        raise ValueError(f"Invalid node format: {node}")

    def _forward_part1(self, layer_module, hidden_states, position_embeddings=None, attention_mask=None):
        """
        Executes: Norm -> Attn -> Residual Add
        Returns: resid_mid
        """
        return self.manager.decomposer.forward_part1(layer_module, hidden_states, position_embeddings, attention_mask)

    def _forward_part2(self, layer_module, hidden_states):
        """
        Executes: Norm -> MLP -> Residual Add
        Returns: resid_post
        """
        return self.manager.decomposer.forward_part2(layer_module, hidden_states)

    def _hook_hidden_activation(self, module, input, output):
        """
        Hook to save activation and enable gradient retention.
        """
        if isinstance(output, tuple):
            output = output[0]
        
        # Save output to the module for later access
        module.output = output
        if module.output.requires_grad:
            module.output.retain_grad()

    def _hook_mid_activation(self, module, input, output):
        """
        Hook to capture resid_mid at the input of post_attention_layernorm.
        """
        # input is a tuple (tensor,)
        val = input[0]
        # Attach to the module (which is the Norm layer)
        module.mid_activation = val
        if module.mid_activation.requires_grad:
            module.mid_activation.retain_grad()

    def register_hooks(self, capture_mid=False):
        """
        Register forward hooks on all model layers.
        """
        self.remove_hooks() # Clear existing
        model = self.manager.get_model()
        if not model:
            raise ValueError("Model not loaded yet.")
            
        # Assuming generic structure model.model.layers (common in HF Qwen, Llama, etc.)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"): # Some archs
            layers = model.layers
        else:
            layers = [] 
            
        for layer in layers:
            # Hook Output (Resid Post)
            handle = layer.register_forward_hook(self._hook_hidden_activation)
            self.hook_handles.append(handle)
            
            # Hook Mid (Resid Mid) - Pre-hook on Post-Attn Norm
            if capture_mid:
                # Use decomposer to find the correct module for mid activation
                mid_module = self.manager.decomposer.get_mid_activation_module(layer)
                
                if mid_module:
                    # Use forward hook to get input? 
                    # register_forward_hook receives (module, input, output)
                    # input is (resid_mid,)
                    handle_mid = mid_module.register_forward_hook(self._hook_mid_activation)
                    self.hook_handles.append(handle_mid)
                else:
                    print(f"Warning: Could not identify mid-activation module for layer {layer}. Skipping mid hook.")

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def reset(self):
        """
        Clears all internal state and specific temporary data from model layers.
        """
        self.remove_hooks()
        self.outputs = None
        self.input_ids = None
        
        # Manually clear output tensors attached to layers to free graph
        model = self.manager.get_model()
        if model:
            layers = []
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layers = model.model.layers
            elif hasattr(model, "layers"):
                layers = model.layers
                
            for layer in layers:
                if hasattr(layer, 'output'):
                    del layer.output
                if hasattr(layer, 'post_attention_layernorm') and hasattr(layer.post_attention_layernorm, 'mid_activation'):
                    del layer.post_attention_layernorm.mid_activation
        
        torch.cuda.empty_cache()


    def compute_logits(self, prompt, is_append_bos=False, topk=10, extra_token_ids=None, extra_token_strs=None, capture_mid=False):
        """
        Section 1: Forward pass to get logits and top-k predictions.
        """
        model = self.manager.get_model()
        tokenizer = self.manager.get_tokenizer()
        
        self.register_hooks(capture_mid=capture_mid)
        
        # Prepare input
        # We tokenize with add_special_tokens=False to manually control the BOS/Start token
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False) 
        input_ids = inputs.input_ids.to(model.device)
        
        if is_append_bos:
            # 1. Try explicit BOS
            bos_id = tokenizer.bos_token_id
            
            # 2. Try CLS (BERT-like)
            if bos_id is None:
                bos_id = tokenizer.cls_token_id

            # 3. Fallback: EOS (Often used as BOS/Separator in Llama/decoder-only models if BOS is missing)
            if bos_id is None:
                bos_id = tokenizer.eos_token_id
                
            if bos_id is not None:
                prefix = torch.tensor([[bos_id]], device=model.device)
                input_ids = torch.cat([prefix, input_ids], dim=1)
                print(f"Appended start token ID: {bos_id}")
            else:
                print("Warning: Append BOS requested but no suitable start token (BOS/CLS/EOS) found.")

        self.input_ids = input_ids
        
        # Embedding with gradients required for LRP base
        # We detach and enable gradients so we can compute attribution w.r.t input embeddings
        # even if the model is frozen/quantized.
        self.input_embeddings = model.get_input_embeddings()(self.input_ids).detach()
        self.input_embeddings.requires_grad_(True)
        
        # Forward pass
        # output_hidden_states=True is crucial for some LRP methods, 
        # though we use hooks for "efficient" method mostly.
        self.outputs = model(
            inputs_embeds=self.input_embeddings, 
            use_cache=False
        )
        
        output_logits = self.outputs.logits
        last_logits = output_logits[0, -1, :]
        
        # Get Top-K
        sorted_logits, sorted_indices = torch.sort(last_logits, dim=-1, descending=True)
        
        # Formatted output
        topk_data = []
        for i in range(topk):
            idx = sorted_indices[i].item()
            token_str = tokenizer.decode([idx])
            logit_val = sorted_logits[i].item()
            topk_data.append({
                "rank": i + 1,
                "token_id": idx,
                "token_str": token_str,
                "logit": logit_val
            })
            
        # Handle Extra Tokens (if requested)
        if extra_token_ids or extra_token_strs:
            # Helper to find rank
            def get_rank(val, sorted_vals):
                # tensor search for rank
                # val is float, sorted_vals is tensor
                # find first index where sorted_vals < val is NOT true?
                # sorted_vals is descending
                # we want count of items > val
                return (sorted_vals > val).sum().item() + 1

            processed_ids = set()
            
            # Process IDs
            if extra_token_ids:
                for tid in extra_token_ids:
                    if tid < 0 or tid >= len(last_logits): continue
                    if tid in processed_ids: continue
                    
                    logit_val = last_logits[tid].item()
                    rank = get_rank(logit_val, sorted_logits)
                    token_str = tokenizer.decode([tid])
                    
                    topk_data.append({
                        "rank": rank,
                        "token_id": tid,
                        "token_str": token_str,
                        "logit": logit_val,
                        "is_extra": True
                    })
                    processed_ids.add(tid)
            
            # Process Strings
            if extra_token_strs:
                print(f"DEBUG: Processing extra strs: {extra_token_strs}")
                for tstr in extra_token_strs:
                    # Encode
                    try:
                        # Ensure we get list of ints
                        encoded = tokenizer.encode(tstr, add_special_tokens=False)
                        print(f"DEBUG: Encoded '{tstr}' -> {encoded} (Type: {type(encoded)})")
                        
                        if hasattr(encoded, 'tolist'): encoded = encoded.tolist()
                        
                        if len(encoded) == 0: 
                            print(f"DEBUG: Empty encoding for '{tstr}'")
                            continue
                            
                        # Take first token
                        tid = encoded[0]
                        print(f"DEBUG: Using TID {tid} for '{tstr}'")
                        
                        if tid in processed_ids: 
                            print(f"DEBUG: TID {tid} already processed")
                            continue
                        
                        logit_val = last_logits[tid].item()
                        rank = get_rank(logit_val, sorted_logits)
                        real_str = tokenizer.decode([tid])
                        
                        print(f"DEBUG: Added extra token: {real_str} (ID: {tid}, Rank: {rank})")
                        
                        topk_data.append({
                            "rank": rank,
                            "token_id": tid,
                            "token_str": real_str,
                            "logit": logit_val,
                            "is_extra": True
                        })
                        processed_ids.add(tid)
                    except Exception as e:
                        print(f"DEBUG: Error processing extra str '{tstr}': {e}")
                        import traceback
                        traceback.print_exc()
        
        # Sort combined data by rank for display consistency?
        # Or keep extras at the bottom? User request: "add more tokens in the existing top-50 table"
        # If we sort, they mix in. If they are rank 1000, they go to bottom.
        # But if they are rank 5 (and we showed top 10), they mix in.
        # Let's sort.
        topk_data.sort(key=lambda x: x['rank'])

        if self.input_ids is None:
             raise ValueError("Input IDs not found. Ensure compute_logits was run.")

        # Get input tokens for visualization
        # Robust token reconstruction ensuring spaces are preserved
        input_tokens = []
        # convert_ids_to_tokens usually preserves the special characters (like Ġ or  )
        raw_tokens = tokenizer.convert_ids_to_tokens(self.input_ids[0])

        for t in raw_tokens:
            # Handle bytes (common in tiktoken-based tokenizers like Qwen)
            if isinstance(t, bytes):
                try:
                    t = t.decode('utf-8')
                except:
                    # Fallback for weird bytes behavior
                    t = str(t)
            
            # If it's a string, it might still have the special whitespace characters
            if isinstance(t, str):
                # Replace SentencePiece underline (U+2581)
                t = t.replace('\u2581', ' ')
                # Replace GPT-2/RoBERTa G-dot (U+0120)
                t = t.replace('\u0120', ' ')
                # Replace Newline char (U+010A)
                t = t.replace('\u010A', '\n')
                # Replace generic replacement char just in case
                t = t.replace('', '') 

            input_tokens.append(t)
            
        return topk_data, last_logits, input_tokens

    def get_target_score(self, backprop_config):
        """
        Calculates the target scalar score (e.g. logit diff) based on config.
        Returns the score tensor (attached to graph).
        """
        if self.outputs is None:
             raise ValueError("Model outputs not computed. Call compute_logits first.")

        mode = backprop_config.get("mode", "max_logit")
        last_logits = self.outputs.logits[0, -1, :]
        sorted_logits, sorted_indices = torch.sort(last_logits, dim=-1, descending=True)
        
        target_token_id = backprop_config.get("target_token_id")
        if target_token_id is not None:
             target_logit = last_logits[target_token_id]
        else:
             target_logit = sorted_logits[0] # Default to Top 1

        if mode == "max_logit":
            return target_logit

        elif mode == "logit_diff":
            strategy = backprop_config.get("strategy", "by_topk_avg")
            top_logit = target_logit 
            
            if strategy == "by_ref_token":
                ref_id = backprop_config.get("ref_token_id")
                if ref_id is None:
                    raise ValueError("ref_token_id required for strategy 'by_ref_token'")
                contrast_logit = last_logits[ref_id]
                target_logit = top_logit - contrast_logit
                
            elif strategy == "demean":
                target_logit = top_logit - last_logits.mean()
                
            elif strategy == "by_topk_avg":
                k = backprop_config.get("k", 10) # default K=10
                k = min(k, len(sorted_logits))
                contrast_logit = sorted_logits[:k].mean()
                target_logit = top_logit - contrast_logit
        
        return target_logit

    def run_backward_pass(self, backprop_config):
        """
        Section 2 Part A: execute backward pass based on configuration.
        """
        target_logit = self.get_target_score(backprop_config)
        
        if target_logit is None:
            raise ValueError(f"Invalid backprop configuration: {backprop_config}")
            
        # Clear previous gradients
        model = self.manager.get_model()
        model.zero_grad()
        
        # Also clear gradients on input embeddings if they exist
        if hasattr(self, 'input_embeddings') and self.input_embeddings is not None:
            if self.input_embeddings.grad is not None:
                self.input_embeddings.grad.zero_()
        
        # Run backward
        # We need to retain grad on hidden states often? 
        # In notebook: h = outputs.hidden_states[-1]; h.retain_grad(); target_logit.backward()
        # But we act on layer.output.grad which is captured by hook + activation

        target_logit.backward(retain_graph=True) # retain_graph needed for interactive exploration where we run backward multiple times
        
    def compute_input_attribution(self, backprop_config):
        """
        Compute input attribution (Input * Gradient).
        """
        # Ensure correct LRP rule is active
        # The forward pass must have been run with the correct rule.
        # If we detect a mismatch, we must force a reload and ask user to re-run forward.
        required_rule = backprop_config.get("lrp_rule", "Attn-LRP") # Default to Attn-LRP
        
        if self.manager.current_lrp_rule and self.manager.current_lrp_rule != required_rule:
             print(f"LRP Rule Mismatch detected (Current: {self.manager.current_lrp_rule}, Requested: {required_rule})")
             print(f"Reloading model {self.manager.current_model_path} with rule={required_rule}...")
             
             old_rule = self.manager.current_lrp_rule
             
             self.manager.load_model(
                model_path=self.manager.current_model_path,
                dtype=self.manager.current_dtype,
                lrp_rule=required_rule
             )
             
             # Since the forward pass graph (self.outputs) was built with the OLD rule, 
             # we cannot proceed. The user must re-run compute_logits.
             raise RuntimeError(
                 f"LRP rule changed from '{old_rule}' to '{required_rule}'. "
                 "The model has been reloaded. You MUST re-run 'compute_logits()' to rebuild the computation graph with the new rule, "
                 "then call 'compute_input_attribution()' again."
             )

        self.run_backward_pass(backprop_config)
        
        # Calculate relevance: (input * grad).sum(-1)
        # self.input_embeddings is [Batch, Seq, Dim]
        if self.input_embeddings.grad is None:
             raise RuntimeError("No gradient found on input embeddings. Ensure compute_logits was run.")
             
        relevance = (self.input_embeddings * self.input_embeddings.grad).float().sum(-1).detach().cpu()[0]
        
        # Return raw relevance
        return relevance.tolist()

    def compute_connection_matrix_gen(self, source, target, node_threshold=None):
        """
        Section 2 Part B: Compute Token-to-Token interaction matrix between two nodes.
        Generator version that yields progress.
        source, target: int (layer idx) or tuple (layer_idx, 'mid'/'post')
        """
        source_layer_idx, source_type = self._parse_node(source)
        target_layer_idx, target_type = self._parse_node(target)
        
        model = self.manager.get_model()
        layers = model.model.layers
        
        target_layer = layers[target_layer_idx]
        
        # 1. Identify Source Tensor
        if source_layer_idx == -1:
            source_tensor = self.input_embeddings
        else:
            layer = layers[source_layer_idx]
            if source_type == 'mid':
                # Use Decomposer to get module
                mid_mod = self.manager.decomposer.get_mid_activation_module(layer)
                if not mid_mod:
                    raise ValueError(f"Decomposer could not identify mid-activation module for layer {source_layer_idx}")
                    
                source_tensor = getattr(mid_mod, 'mid_activation', None)
                if source_tensor is None: 
                    raise ValueError(f"Mid activation for layer {source_layer_idx} not captured. Enable capture_mid in compute_logits.")
            else:
                source_tensor = layer.output

        # 2. Identify Target Tensor and Gradient
        if target_type == 'mid':
            # We need the gradient at the mid point (input to post_attn_norm)
            mid_mod = self.manager.decomposer.get_mid_activation_module(target_layer)
            if not mid_mod:
                 raise ValueError(f"Decomposer could not identify mid-activation module for target {target_layer_idx}")

            target_tensor = getattr(mid_mod, 'mid_activation', None)
            if target_tensor is None:
                raise ValueError(f"Mid activation for target {target_layer_idx} not captured.")
        else:
            target_tensor = target_layer.output
            
        target_grad = target_tensor.grad
        
        # Disable gradient checkpointing temporarily
        was_checkpointing = model.is_gradient_checkpointing
        if was_checkpointing:
            model.gradient_checkpointing_disable()
            
        try:
            # Prepare Input
            target_layer_input = source_tensor.detach()
            batch_size, seq_len, hidden_dim = target_layer_input.shape
            
            # Target Real Relevance
            if target_grad is not None:
                real_target_rel = (target_tensor * target_grad).sum(dim=-1)[0]
            else:
                real_target_rel = torch.zeros(seq_len, device=model.device)
                
            # Filter Indices
            total_params = model.num_parameters()
            if node_threshold is None: node_threshold = 0.01

            if node_threshold > 0:
                indices_to_compute = torch.nonzero(real_target_rel.abs() > node_threshold).squeeze(-1).tolist()
                if isinstance(indices_to_compute, int): indices_to_compute = [indices_to_compute]
                print(f"DEBUG: Node Threshold {node_threshold}. Computing for {len(indices_to_compute)}/{seq_len} nodes.")
            else:
                indices_to_compute = list(range(seq_len))
                
            # Fixed Position IDs (for Rotary)
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
            
            # Construct Operation Sequence
            ops = []
            if source_layer_idx != -1:
                 if source_type == 'mid':
                      ops.append(('part2', layers[source_layer_idx]))
            
            # Intermediate Layers
            for i in range(source_layer_idx + 1, target_layer_idx):
                 ops.append(('part1', layers[i]))
                 ops.append(('part2', layers[i]))
                 
            # Target Layer
            if target_layer_idx > source_layer_idx:
                 ops.append(('part1', layers[target_layer_idx]))
                 if target_type == 'post':
                      ops.append(('part2', layers[target_layer_idx]))
            elif target_layer_idx == source_layer_idx:
                 pass # Already handled or identity
            elif source_layer_idx == -1:
                 # Special case: source is embeddings, target is 0
                 # Range was (0,0) empty.
                 # Need to add target 0 parts
                 ops.append(('part1', layers[target_layer_idx]))
                 if target_type == 'post':
                      ops.append(('part2', layers[target_layer_idx]))

            # Pre-calc Rotary Embedding (using dummy execution or helper)
            # We assume rotary depends only on position_ids and shape
            rotary_emb = None
            if hasattr(model.model, 'rotary_emb'):
                 rotary_emb = model.model.rotary_emb(target_layer_input, position_ids)
            elif hasattr(model.model, 'rotary_embs') and 'full_attention' in model.model.rotary_embs:
                 rotary_emb = model.model.rotary_embs['full_attention'](target_layer_input, position_ids)
            elif hasattr(model.model, 'rotary_embs') and len(model.model.rotary_embs) > 0:
                 rotary_emb = list(model.model.rotary_embs.values())[0](target_layer_input, position_ids)
            
            # Chunk Processing
            current_dtype = target_layer_input.dtype
            BATCH_CHUNK_SIZE = get_batch_chunk_size(total_params, current_dtype)
            token_interaction = torch.zeros(seq_len, seq_len, device=model.device)
            target_grad_full = target_grad # Alias

            total_items = len(indices_to_compute)

            for i in range(0, total_items, BATCH_CHUNK_SIZE):
                yield {"type": "progress", "current": i, "total": total_items}
                
                chunk_indices = indices_to_compute[i : i + BATCH_CHUNK_SIZE]
                current_batch_size = len(chunk_indices)
                
                expanded_input = target_layer_input.expand(current_batch_size, seq_len, hidden_dim).clone().requires_grad_(True)
                
                # Execute Ops
                hidden_states = expanded_input
                for op_type, layer_mod in ops:
                    if op_type == 'part1':
                        hidden_states = self._forward_part1(layer_mod, hidden_states, position_embeddings=rotary_emb)
                    else:
                        hidden_states = self._forward_part2(layer_mod, hidden_states)
                
                reconstructed_output = hidden_states
                
                # Backward
                grad_output_chunk = torch.zeros(current_batch_size, seq_len, hidden_dim, dtype=reconstructed_output.dtype, device=model.device)
                
                for batch_idx, global_idx in enumerate(chunk_indices):
                    if target_grad_full is not None:
                        grad_output_chunk[batch_idx, global_idx, :] = target_grad_full[0, global_idx, :]
                        
                grad_input = torch.autograd.grad(outputs=reconstructed_output, inputs=expanded_input, grad_outputs=grad_output_chunk, retain_graph=False)[0]
                
                chunk_relevance = (grad_input * expanded_input).sum(dim=-1)
                token_interaction[chunk_indices, :] = chunk_relevance.detach().to(token_interaction.dtype)
                
                del expanded_input, hidden_states, reconstructed_output, grad_output_chunk, grad_input, chunk_relevance
                torch.cuda.empty_cache()

            # Source Real Relevance
            if source_layer_idx == -1:
                if self.input_embeddings.grad is not None:
                    real_source_rel = (self.input_embeddings * self.input_embeddings.grad).sum(dim=-1)[0]
                else:
                    real_source_rel = torch.zeros(seq_len, device=model.device)
            else:
                if source_tensor.grad is not None:
                    real_source_rel = (source_tensor * source_tensor.grad).sum(dim=-1)[0]
                else:
                    real_source_rel = torch.zeros(seq_len, device=model.device)

            yield {
                "type": "result", 
                "payload": {
                    "matrix": token_interaction.detach().float().cpu().numpy(),
                    "real_target_rel": real_target_rel.detach().float().cpu().numpy(),
                    "real_source_rel": real_source_rel.detach().float().cpu().numpy()
                }
            }
            
        finally:
            if was_checkpointing:
                model.gradient_checkpointing_enable()

    def compute_connection_matrix(self, source, target):
        for item in self.compute_connection_matrix_gen(source, target):
            if item.get("type") == "result":
                return item["payload"]
        return None


