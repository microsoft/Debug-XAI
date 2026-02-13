import unittest
import torch
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from attnlrp_circuit.backend.models.manager import ModelManager
from attnlrp_circuit.backend.core import AttributionEngine
from attnlrp_circuit.backend.circuit import CircuitAnalyzer

class TestBackendFull(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Setting up Model for Backend Tests ---")
        cls.manager = ModelManager()
        # Use float32 for testing to avoid precision issues in checks, 
        # though bfloat16 is standard for this model.
        cls.manager.load_model("Qwen/Qwen3-0.6B", quantization_4bit=False, dtype="float32")
        cls.engine = AttributionEngine(cls.manager)
        cls.analyzer = CircuitAnalyzer(cls.engine)
        cls.device = cls.manager.device
        
        cls.prompt = "The capital of France is"
        
    def setUp(self):
        # Reset engine state before each test
        self.engine.reset()

    def test_01_standard_flow(self):
        """Test the standard (coarse) attribution flow."""
        print("\nTest 01: Standard Flow (Layer-to-Layer)")
        
        # 1. Compute Logits
        topk, last_logits, input_tokens = self.engine.compute_logits(self.prompt, capture_mid=False)
        self.assertTrue(len(topk) > 0)
        self.assertIsNotNone(self.engine.input_ids)
        self.assertIsNotNone(self.engine.outputs)
        
        # 2. Backprop
        target_token_id = topk[0]['token_id']
        bp_config = {
            "mode": "max_logit",
            "target_token_id": target_token_id
        }
        self.engine.run_backward_pass(bp_config)
        
        # Check gradients exist on last layer output
        last_layer = self.manager.model.model.layers[-1]
        self.assertIsNotNone(last_layer.output.grad, "Gradient not found on last layer output")

        # 3. Connection Matrix (Layer 0 -> Layer 1)
        # Just compute one transition to save time
        print("Computing connection 0 -> 1")
        matrix_data = self.engine.compute_connection_matrix(source=0, target=1)
        
        self.assertIsNotNone(matrix_data)
        self.assertTrue('matrix' in matrix_data)
        self.assertTrue(isinstance(matrix_data['matrix'], np.ndarray))
        
        seq_len = len(input_tokens)
        self.assertEqual(matrix_data['matrix'].shape, (seq_len, seq_len))

    def test_02_fine_grained_flow(self):
        """Test the fine-grained (MLP/Attn separated) flow."""
        print("\nTest 02: Fine-Grained Flow (Attn/MLP Separation)")
        
        # 1. Compute Logits with Capture Mid
        topk, _, input_tokens = self.engine.compute_logits(self.prompt, capture_mid=True)
        
        # Verify Mid Capture
        layer0 = self.manager.model.model.layers[0]
        # Decomposer should have found post_attention_layernorm
        mid_mod = self.manager.decomposer.get_mid_activation_module(layer0)
        self.assertTrue(hasattr(mid_mod, 'mid_activation'), "Mid activation not captured on Layer 0")
        self.assertIsNotNone(mid_mod.mid_activation)
        
        # 2. Backprop
        target_token_id = topk[0]['token_id']
        bp_config = {
            "mode": "max_logit",
            "target_token_id": target_token_id
        }
        self.engine.run_backward_pass(bp_config) # Should populate mid gradients
        
        self.assertIsNotNone(mid_mod.mid_activation.grad, "Gradient not found on mid activation of Layer 0")

        # 3. Compute Transitions
        # Case A: Input -> (0, 'mid')  [Part 1 of Layer 0]
        print("Computing connection Input -> (0, 'mid')")
        data_a = self.engine.compute_connection_matrix(source=-1, target=(0, 'mid'))
        self.assertIsNotNone(data_a)
        
        # Case B: (0, 'mid') -> 0      [Part 2 of Layer 0]
        print("Computing connection (0, 'mid') -> 0")
        data_b = self.engine.compute_connection_matrix(source=(0, 'mid'), target=0)
        self.assertIsNotNone(data_b)
        
        # Case C: 0 -> (1, 'mid')      [Part 1 of Layer 1]
        print("Computing connection 0 -> (1, 'mid')")
        data_c = self.engine.compute_connection_matrix(source=0, target=(1, 'mid'))
        self.assertIsNotNone(data_c)

    def test_03_circuit_analyzer_integration(self):
        """Test CircuitAnalyzer auto-detection and loop."""
        print("\nTest 03: Circuit Analyzer Integration")
        
        # Setup fine-grained state
        self.engine.compute_logits(self.prompt, capture_mid=True)
        bp_config = {"mode": "max_logit", "target_token_id": 1234} # Arbitrary
        
        # Run Analyzer for just the first layer to save time
        # Explicitly pass layers including mids to verify it handles them
        # Note: If we pass layers=None, it tries all layers (too slow for unit test)
        
        test_nodes = [-1, (0, 'mid'), 0] # Input -> Mid0 -> Post0
        
        connections = self.analyzer.compute_connection_matrices(bp_config, layers=test_nodes)
        
        self.assertEqual(len(connections), 2, "Should have 2 transitions: Input->Mid, Mid->Post")
        
        c1 = connections[0]
        self.assertEqual(c1['src_layer'], -1)
        self.assertEqual(c1['tgt_layer'], (0, 'mid'))
        
        c2 = connections[1]
        self.assertEqual(c2['src_layer'], (0, 'mid'))
        self.assertEqual(c2['tgt_layer'], 0)

if __name__ == '__main__':
    unittest.main()
