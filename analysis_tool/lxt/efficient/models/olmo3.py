from functools import partial
from torch.nn import Dropout
from transformers.models.olmo3 import modeling_olmo3
from transformers.models.olmo3.modeling_olmo3 import Olmo3MLP, Olmo3RMSNorm

from lxt.efficient.patches import patch_method, patch_attention, patch_cp_attention
from lxt.efficient.patches import rms_norm_forward, gated_mlp_forward, cp_gated_mlp_forward, dropout_forward

attnLRP = {
    Olmo3MLP: partial(patch_method, gated_mlp_forward),
    Olmo3RMSNorm: partial(patch_method, rms_norm_forward), 
    Dropout: partial(patch_method, dropout_forward),
    modeling_olmo3: patch_attention,
}

cp_LRP = {
    Olmo3MLP: partial(patch_method, cp_gated_mlp_forward),
    Olmo3RMSNorm: partial(patch_method, rms_norm_forward), 
    Dropout: partial(patch_method, dropout_forward),
    modeling_olmo3: patch_cp_attention,
}
