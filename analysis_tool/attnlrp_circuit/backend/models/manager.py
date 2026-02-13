import torch
import importlib
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.qwen3 import modeling_qwen3
# Import other models as needed via conditional imports or a mapping
from lxt.efficient import monkey_patch
import gc
from .factory import get_decomposer

class ModelManager:
    """
    Manages model loading, quantization, and patching.
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.decomposer = None
        
        # Track active configuration for reloading
        self.current_model_path = None
        self.current_dtype = None
        self.current_lrp_rule = None

    def load_model(self, model_path="Qwen/Qwen3-0.6B", quantization_4bit=False, dtype="auto", revision=None, lrp_rule="Attn-LRP"):
        """
        Loads the model and tokenizer, applies monkey patches for LRP.
        lrp_rule: "Attn-LRP" (default) or "CP-LRP" (Conservative Propagation)
        """
        if revision == "" or revision == "null":
            revision = None
            
        print(f"Loading model from {model_path} with revision={revision} and rule={lrp_rule}...")
        
        # Store active configuration
        self.current_model_path = model_path
        self.current_dtype = dtype
        self.current_lrp_rule = lrp_rule
        
        # Free up memory if reloading
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        self.model_name = model_path.split('/')[-1]
        
        # Initialize Decomposer
        self.decomposer = get_decomposer(self.model_name)
        
        # Apply Monkey Patch for Efficient LRP
        target_module = None
        patch_map = None
        
        lower_path = model_path.lower()
        if "qwen3" in lower_path:
            importlib.reload(modeling_qwen3) # Reset to original classes to remove previous patches
            target_module = modeling_qwen3
            try:
                import lxt.efficient.models.qwen3 as lxt_qwen3
                importlib.reload(lxt_qwen3) # Reload to update class references from new modeling_qwen3
                patch_map = lxt_qwen3.cp_LRP if lrp_rule == "CP-LRP" else lxt_qwen3.attnLRP
            except ImportError as e:
                print(f"Warning: Could not import lxt.efficient.models.qwen3: {e}")

        elif "olmo" in lower_path:
            try:
                from transformers.models.olmo3 import modeling_olmo3
                importlib.reload(modeling_olmo3)
                target_module = modeling_olmo3
                import lxt.efficient.models.olmo3 as lxt_olmo3
                importlib.reload(lxt_olmo3)
                patch_map = lxt_olmo3.cp_LRP if lrp_rule == "CP-LRP" else lxt_olmo3.attnLRP
            except ImportError as e:
                print(f"Warning: Could not import modeling_olmo3 or lxt module. LRP might fail. Error: {e}")
                
        elif "qwen2" in lower_path:
             try:
                 from transformers.models.qwen2 import modeling_qwen2
                 importlib.reload(modeling_qwen2)
                 target_module = modeling_qwen2
                 import lxt.efficient.models.qwen2 as lxt_qwen2
                 importlib.reload(lxt_qwen2)
                 patch_map = lxt_qwen2.cp_LRP if lrp_rule == "CP-LRP" else lxt_qwen2.attnLRP
             except ImportError as e:
                 print(f"Warning: Could not import qwen2 or lxt: {e}")
        
        if target_module:
            if patch_map:
                monkey_patch(target_module, patch_map=patch_map, verbose=True)
            else:
                monkey_patch(target_module, verbose=True) # Fallback to default

        # Add else if for other models supported by lxt
        
        # Map string dtype to torch dtype
        torch_dtype = "auto"
        bnb_dtype = torch.bfloat16 # Default for 4bit compute
        
        if dtype == "float16":
            torch_dtype = torch.float16
            bnb_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
            bnb_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
            bnb_dtype = torch.float32

        # Quantization Config
        quantization_config = None
        if quantization_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_dtype, 
            )

        # Load Model
        if "qwen3" in model_path.lower():
            self.model = modeling_qwen3.Qwen3ForCausalLM.from_pretrained(
                model_path, 
                device_map=self.device, 
                torch_dtype=torch_dtype, 
                max_memory={1: "30GiB"},
                quantization_config=quantization_config,
                revision=revision
            )
        else:
             # Fallback for generic loading if specific class fails
             self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map=self.device, 
                torch_dtype=torch_dtype, 
                max_memory={1: "30GiB"},
                quantization_config=quantization_config,
                revision=revision
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        
        # Prepare model for LRP
        self.model.eval() # Use eval usually, but test.ipynb uses train() + gradients
        # test.ipynb: model.train(), gradient_checkpointing_enable(), requires_grad=False
        
        # "model.train()" is often needed for Gradient Checkpointing to work in HF
        self.model.train() 
        self.model.gradient_checkpointing_enable()
        
        # Deactivate gradients on parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"Model {self.model_name} loaded successfully on {self.device}")
        return self.model_name

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
