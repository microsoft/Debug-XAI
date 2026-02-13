from abc import ABC, abstractmethod
import torch.nn as nn

class LayerDecomposer(ABC):
    """
    Abstract base class for decomposing Transformer layers into 
    Attention (Part 1) and MLP (Part 2) components.
    """
    
    @abstractmethod
    def get_mid_activation_module(self, layer_module):
        """
        Returns the module whose input corresponds to 'resid_mid'.
        This is typically the Post-Attention LayerNorm.
        """
        pass

    @abstractmethod
    def forward_part1(self, layer_module, hidden_states, position_embeddings=None, attention_mask=None):
        """
        Executes: Norm -> Attn -> Residual Add
        Returns: resid_mid
        """
        pass

    @abstractmethod
    def forward_part2(self, layer_module, hidden_states):
        """
        Executes: Norm -> MLP -> Residual Add
        Returns: resid_post
        """
        pass
