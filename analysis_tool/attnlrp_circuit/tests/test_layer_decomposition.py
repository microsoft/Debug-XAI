import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from attnlrp_circuit.backend.models.manager import ModelManager
from attnlrp_circuit.backend.models.factory import get_decomposer

class TestLayerDecomposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Loading Qwen/Qwen3-0.6B for testing... this might take a minute.")
        cls.manager = ModelManager()
        # Loading in bfloat16 to match typical usage, or float32 for precision in test
        # float32 is safer for equality checks
        cls.manager.load_model("Qwen/Qwen3-0.6B", quantization_4bit=False, dtype="float32")
        cls.model = cls.manager.get_model()
        cls.decomposer = cls.manager.decomposer
        
    def setUp(self):
        self.layer = self.model.model.layers[0]
        hidden_dim = self.model.config.hidden_size
        # Create random input: [Batch=1, Seq=10, Dim]
        self.x = torch.randn(1, 10, hidden_dim, device=self.model.device, dtype=self.model.dtype)
        
        # Position embeddings and mask are needed for real models usually
        seq_len = 10
        self.position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.model.device).unsqueeze(0)
        
        # Calculate rotary embeddings if available
        self.rotary_emb = None
        if hasattr(self.model.model, 'rotary_emb'):
             self.rotary_emb = self.model.model.rotary_emb(self.x, self.position_ids)

        # Simple attention mask (all ones)
        self.attention_mask = torch.ones((1, 1, seq_len, seq_len), device=self.model.device, dtype=self.model.dtype)
        # HF standard mask - usually [Batch, 1, tgt_len, src_len] or boolean
        # For Qwen, let's try to infer or just pass None if causal mask is handled internally?
        # Qwen3 layers usually handle mask generation if None, or expect 4D. 
        # Using None for simplicity if possible, or construct proper causal mask. 
        # Actually, let's just use None and rely on layer internal behavior for causal inference if applicable,
        # or passing the explicit rotary is key.

    def test_part_reconstruction(self):
        """Test that Part1 + Part2 equals full Forward pass"""
        
        print(f"\nTesting decomposition on Layer {self.layer.self_attn.__class__.__name__}")

        # 1. Full Forward
        # Qwen layer forward signature: (hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs)
        # But we need to match how decomposer calls it. 
        # decomposer.forward_part1 calls: attn(norm(x), position_embeddings=..., attention_mask=...)
        
        with torch.no_grad():
            # Run the layer as the model would
            # Note: We need to match arguments exactly.
            # Real layer forward() handles Norms internally? Yes.
            # layer(hidden_states, attention_mask=..., position_embeddings=...)
            
            # Note: Qwen2/3 implementation details:
            # forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, ...)
            # Rotary is usually computed inside or passed. 
            # In our decomposer we pass `position_embeddings`.
             
            # Let's check how we implemented decomposer calls:
            # attn_out = attn(norm_out, position_embeddings=position_embeddings, attention_mask=attention_mask)
            
            # The standard layer.forward() typically does:
            # residual = hidden_states
            # hidden_states = self.input_layernorm(hidden_states)
            # hidden_states, ... = self.self_attn(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings, ...)
            # hidden_states = residual + hidden_states
            # ...
            
            # So if we call layer(), it should match our manual composition.
            
            # Passing rotary_emb as kwargs/position_embeddings depends on specific implementation version.
            # transformers Qwen2: layer(..., position_embeddings=cos_sin, ...)
            
            expected_out = self.layer(
                self.x, 
                attention_mask=self.attention_mask, 
                position_embeddings=self.rotary_emb
            )
            if isinstance(expected_out, tuple): expected_out = expected_out[0]
        
        # 2. Decomposed
        with torch.no_grad():
            mid = self.decomposer.forward_part1(
                self.layer, 
                self.x, 
                position_embeddings=self.rotary_emb,
                attention_mask=self.attention_mask
            )
            actual_out = self.decomposer.forward_part2(self.layer, mid)
            
        # Check
        # Using slightly looser tolerance for float operations
        self.assertTrue(torch.allclose(expected_out, actual_out, atol=1e-5), 
                        f"Decomposed forward pass mismatch. Max diff: {(expected_out - actual_out).abs().max().item()}")
        
    def test_mid_activation_module(self):
        """Test module retrieval"""
        mod = self.decomposer.get_mid_activation_module(self.layer)
        # Qwen has post_attention_layernorm
        self.assertEqual(mod, self.layer.post_attention_layernorm)

if __name__ == '__main__':
    unittest.main()
