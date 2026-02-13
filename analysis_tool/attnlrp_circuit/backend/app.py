from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import sys
import torch
import json
import glob
from huggingface_hub import list_models, list_repo_refs
import logging
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Safe handle checking for finite
            f = float(obj)
            return f if np.isfinite(f) else 0.0
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Ensure backend can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models import ModelManager
from backend.core import AttributionEngine
from backend.circuit import CircuitAnalyzer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse

EXP_ROOT = "/mnt/caiw/exp/clean_cases"

app = FastAPI(title="LLM Insider for Failure Debugging")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Frontend (Static Files)
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
if os.path.exists(frontend_path):
    app.mount("/ui", StaticFiles(directory=frontend_path), name="ui")

@app.get("/")
async def read_root():
    return RedirectResponse(url="/ui/index.html")

# Global instances
model_manager = ModelManager()
attribution_engine = None # Initialize after model load

# Caching for connection matrices to speed up slider interactions
CACHED_CONNECTION_DATA = {
    "config_hash": None,
    "data": None
}

def get_config_hash(bp_config, layers):
    try:
        # Create a deterministic hash string
        return json.dumps({
            "bp": bp_config,
            "layers": sorted(layers)
        }, sort_keys=True)
    except:
        return None


# Pydantic models for inputs
class LoadModelRequest(BaseModel):
    model_path: str = "Qwen/Qwen3-0.6B"
    quantization_4bit: bool = True
    dtype: str = "float16" # float16, bfloat16, float32, auto
    revision: Optional[str] = None
    lrp_rule: str = "Attn-LRP" # "Attn-LRP" or "CP-LRP"

class ComputeLogitsRequest(BaseModel):
    prompt: str
    is_append_bos: bool = True
    topk: int = 10
    extra_token_ids: Optional[List[int]] = None
    extra_token_strs: Optional[List[str]] = None
    capture_mid: bool = False  # Fine-grained attribution separation

class BackpropConfig(BaseModel):
    mode: str = "max_logit" # "max_logit" or "logit_diff"
    strategy: Optional[str] = "by_topk_avg" # "demean", "by_topk_avg", "by_ref_token"
    ref_token_id: Optional[int] = None
    contrast_rank: Optional[int] = 2
    k: Optional[int] = 10
    node_threshold: Optional[float] = 0.01  # Threshold for computing node inter-connections

class ComputeCircuitRequest(BaseModel):
    # Configurations for backprop
    backprop_config: BackpropConfig
    
    # New Multi-Layer Field
    layers: List[int]
    
    # Pruning Params
    pruning_mode: str = "by_per_layer_cum_mass_percentile"
    top_p: float = 0.9
    edge_threshold: float = 0.01 # Used if by_global_threshold


class ComputeInputAttributionRequest(BaseModel):
    target_token_id: int
    contrast_token_id: Optional[int] = None
    backprop_config: BackpropConfig

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30
    append_token_id: Optional[int] = None

@app.get("/api/list_hf_models")
async def list_hf_models(series: str = "Qwen2"):
    """
    List models from HuggingFace Hub filtered by series/author.
    """
    try:
        if series.lower() == "qwen2":
            # Search Qwen Organization for Qwen2
            # Use search="Qwen2" effectively
            models = list(list_models(author="Qwen", search="Qwen2", filter="text-generation", sort="downloads", direction=-1, limit=50))
            return {"models": [m.id for m in models]}
            
        elif series.lower() == "qwen3":
            # Search Qwen Organization for Qwen3
            models = list(list_models(author="Qwen", search="Qwen3", filter="text-generation", sort="downloads", direction=-1, limit=50))
            return {"models": [m.id for m in models]}
            
        elif series.lower() == "olmo3":
            # Search AllenAI for OLMo-3
            # Use search="Olmo-3" based on typical naming "allenai/Olmo-3-7B-Think"
            # Some might capture OLMo-3.1 etc
            models = list(list_models(author="allenai", search="Olmo-3", filter="text-generation", sort="downloads", direction=-1, limit=50))
            return {"models": [m.id for m in models]}

        elif series.lower() == "olmo":
            # Search AllenAI for OLMo
            models = list(list_models(author="allenai", search="OLMo", filter="text-generation", sort="downloads", direction=-1, limit=50))
            return {"models": [m.id for m in models]}
        
        elif series.lower() == "qwen":
            # Legacy/All Qwen
            models = list(list_models(author="Qwen", filter="text-generation", sort="downloads", direction=-1, limit=50))
            return {"models": [m.id for m in models]}

        # Generic fallback
        models = list(list_models(search=series, filter="text-generation", sort="downloads", direction=-1, limit=20))
        return {"models": [m.id for m in models]}
        
    except Exception as e:
        print(f"Error listing models: {e}")
        # Return fallback/hardcoded list if offline
        if series.lower() == "qwen2":
             return {"models": ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2-0.5B", "Qwen/Qwen2-1.5B", "Qwen/Qwen2-7B"]}
        elif series.lower() == "qwen3":
             return {"models": ["Qwen/Qwen3-0.6B"]}
        elif series.lower() == "qwen":
            return {"models": ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"]}
        elif series.lower() == "olmo3":
            return {"models": ["allenai/Olmo-3-7B-Think"]}
        elif series.lower() == "olmo":
            return {"models": ["allenai/OLMo-7B", "allenai/OLMo-1B-0724", "allenai/Olmo-3-7B-Think"]}
        return {"models": [], "error": str(e)}

@app.get("/api/list_model_revisions")
async def list_model_revisions(model_id: str):
    """
    List git branches/refs for a model.
    """
    try:
        refs = list_repo_refs(model_id)
        branches = [b.name for b in refs.branches]
        tags = [t.name for t in refs.tags]
        return {"branches": branches, "tags": tags}
    except Exception as e:
        print(f"Error listing revisions for {model_id}: {e}")
        return {"branches": [], "tags": [], "error": str(e)}

@app.post("/api/cleanup")
async def cleanup_memory():
    global attribution_engine
    if attribution_engine:
        attribution_engine.reset()
    else:
        # even if no engine, try to clear cache
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()
    
    return {"status": "success", "message": "Memory cleanup complete"}

@app.post("/api/generate")
async def generate_continuation(request: GenerateRequest):
    if not model_manager.model:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    tokenizer = model_manager.tokenizer
    model = model_manager.model
    device = model_manager.device

    try:
        # Switch to eval for generation
        was_training = model.training
        model.eval()
        
        # Encode prompt
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)
        
        # Append token if requested
        if request.append_token_id is not None:
            token_tensor = torch.tensor([[request.append_token_id]], device=device)
            input_ids = torch.cat([input_ids, token_tensor], dim=1)
            
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        new_token_ids = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        
        # Restore training mode
        if was_training:
            model.train()
            
        return {"generated_text": generated_text}
        
    except Exception as e:
        if model_manager.model and was_training:
            model_manager.model.train()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load_model")
async def load_model(request: LoadModelRequest):
    global attribution_engine
    try:
        # Pass dtype to load_model
        model_name = model_manager.load_model(
            request.model_path, 
            request.quantization_4bit,
            dtype=request.dtype,
            revision=request.revision,
            lrp_rule=request.lrp_rule
        )
        attribution_engine = AttributionEngine(model_manager)
        
        # Get Num Layers
        n_layers = 28 # Default for Qwen 0.5B
        try:
             # Try access config
             if hasattr(model_manager.model, 'config'):
                 n_layers = getattr(model_manager.model.config, 'num_hidden_layers', 28)
        except:
             pass
             
        return {"status": "success", "message": f"Model {model_name} loaded successfully", "n_layers": n_layers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compute_logits")
async def compute_logits(request: ComputeLogitsRequest):
    global attribution_engine
    if not attribution_engine:
        raise HTTPException(status_code=400, detail="Model not loaded. Please call /api/load_model first.")
    
    try:
        topk_data, _, input_tokens = attribution_engine.compute_logits(
            prompt=request.prompt,
            is_append_bos=request.is_append_bos,
            topk=request.topk,
            extra_token_ids=request.extra_token_ids,
            extra_token_strs=request.extra_token_strs,
            capture_mid=request.capture_mid
        )
        
        # Convert simple string list input_tokens to list of objects for frontend consistency
        # Assuming sequential IDs matching index for now, or just use strings.
        # Frontend expects: tokens[i].token_str
        token_objs = [{"token_str": t, "token_id": i} for i, t in enumerate(input_tokens)]
        
        return {"data": topk_data, "tokens": token_objs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compute_input_attribution")
async def compute_input_attribution_endpoint(request: ComputeInputAttributionRequest):
    global attribution_engine
    if not attribution_engine:
        raise HTTPException(status_code=400, detail="Model not loaded.")
    
    try:
        if attribution_engine.outputs is None:
             raise HTTPException(status_code=400, detail="No forward pass found. Run compute_logits first.")
             
        # Inject target token ID into backprop config
        bp_config = request.backprop_config.dict()
        bp_config["target_token_id"] = request.target_token_id
        
        relevance = attribution_engine.compute_input_attribution(bp_config)
        return {"relevance": relevance}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compute_circuit")
async def compute_circuit(request: ComputeCircuitRequest):
    global attribution_engine
    if not attribution_engine:
        raise HTTPException(status_code=400, detail="Model not loaded.")
        
    if attribution_engine.outputs is None:
         raise HTTPException(status_code=400, detail="No forward pass found. Run compute_logits first.")

    async def generate_response():
        try:
            # Step 1: Run Backward Pass
            yield json.dumps({"type": "progress", "msg": "Initiating Backward Pass...", "percent": 0}) + "\n"
            
            # Use CircuitAnalyzer
            analyzer = CircuitAnalyzer(attribution_engine)
            
            bp_config = request.backprop_config.dict()
            
            # We explicitly run backward pass first (though build_graph does it, we want to emit progress)
            # Check Cache
            current_hash = get_config_hash(bp_config, request.layers)
            connection_data = None
            
            if CACHED_CONNECTION_DATA["config_hash"] == current_hash and CACHED_CONNECTION_DATA["data"] is not None:
                 yield json.dumps({"type": "progress", "msg": "Using Cached Matrices (Fast)...", "percent": 50}) + "\n"
                 connection_data = CACHED_CONNECTION_DATA["data"]
            else:
                 yield json.dumps({"type": "progress", "msg": "Computing Circuit (This may take a moment)...", "percent": 20}) + "\n"
                 # Run the heavy lifting
                 connection_data = analyzer.compute_connection_matrices(bp_config, sorted(request.layers))
                 
                 # Update Cache
                 CACHED_CONNECTION_DATA["config_hash"] = current_hash
                 CACHED_CONNECTION_DATA["data"] = connection_data

            yield json.dumps({"type": "progress", "msg": "Pruning & Building Graph...", "percent": 80}) + "\n"

            G, pruning_details = analyzer.build_graph_from_matrices(
                connection_data,
                edge_rel_threshold=request.edge_threshold,
                pruning_mode=request.pruning_mode,
                top_p=request.top_p
            )
            
            yield json.dumps({"type": "progress", "msg": "Graph Constructed. Serializing...", "percent": 90}) + "\n"
            
            # Serialize Graph
            # nx.node_link_data returns dict with 'nodes' and 'links'
            # Nodes have 'id' which are tuples (layer, token). JSON converts tuple to list.
            graph_data = nx.node_link_data(G)
            
            yield json.dumps({
                "type": "graph_data", 
                "graph": graph_data, 
                "pruning_details": pruning_details
            }, cls=NumpyEncoder) + "\n"
            
            yield json.dumps({"type": "progress", "msg": "Complete!", "percent": 100}) + "\n"
            yield json.dumps({"type": "complete"}) + "\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"type": "error", "msg": str(e)}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")

# Mappings for dataset folder names to absolute paths
DATASET_PATHS = {
    "IFEval": "/mnt/caiw/exp/IFEval/data/without_think",
    "HumanEval": "/mnt/caiw/exp/evalplus/data/traces", 
    "MATH": "/mnt/caiw/exp/MATH/qwen3-0.6b/traces",
    "MMLU_Pro": "/mnt/caiw/exp/MMLU_Pro/data/test_data/traces",
    "GAIA2_adaptability": "/mnt/caiw/exp/GAIA2/data/adaptability"
}

@app.get("/api/datasets")
async def get_datasets():
    # Return available datasets (checking existence not strictly required but good practice)
    available = []
    for ui_name, path in DATASET_PATHS.items():
        if os.path.exists(path):
            available.append(ui_name)
    return {"datasets": available}

@app.get("/api/traces/{dataset_name}")
async def get_traces(dataset_name: str):
    trace_dir = DATASET_PATHS.get(dataset_name)
    if not trace_dir:
        raise HTTPException(status_code=404, detail="Unknown dataset")
    
    path = os.path.join(trace_dir, "*.json")
    
    files = glob.glob(path)
    # Extract basenames
    trace_ids = sorted([os.path.basename(f) for f in files])
    return {"traces": trace_ids}

@app.get("/api/trace_details/{dataset_name}/{trace_id}")
async def get_trace_details(dataset_name: str, trace_id: str):
    trace_dir = DATASET_PATHS.get(dataset_name)
    if not trace_dir:
        raise HTTPException(status_code=404, detail="Unknown dataset")
        
    file_path = os.path.join(trace_dir, trace_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Trace file not found")
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        metadata = data.get("metadata", {})
        model_path = metadata.get("model", "")
        
        prompt_part = data.get("prompt", "")
        completion_part = data.get("completion_before_err", "")
        # Get raw completion for comparison
        completion_orig = data.get("completion", "")
        
        full_prompt = prompt_part + completion_part
        
        return {
            "model_path": model_path,
            "prompt": full_prompt,
            "raw_prompt": prompt_part,
            "completion": completion_orig,
            # Force defaults as requested
            "quantization": False,
            "dtype": "bfloat16",
            "topk_token_explore": data.get("topk_token_explore", []),
            "other_candidates": {k: v for k, v in data.items() if k.endswith("_topk_token_explore") and k != "topk_token_explore"},
            "topk_token_explore_4b": data.get("4b_topk_token_explore", []) # Legacy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
