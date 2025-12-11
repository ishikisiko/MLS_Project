import unittest
import torch
import sys
import os

# Put project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.privacy import DPEngine

class TestDPEngine(unittest.TestCase):
    def setUp(self):
        self.dp_engine = DPEngine(max_norm=1.0, noise_multiplier=0.1)

    def test_clipping(self):
        # Create a gradient with norm > 1.0
        g1 = torch.tensor([2.0, 0.0]) # Norm is 2.0
        grads = [g1]
        
        clipped = self.dp_engine.per_sample_clip(grads)
        
        # Norm of result should be <= 1.0
        clipped_norm = torch.norm(clipped[0])
        self.assertTrue(clipped_norm <= 1.0 + 1e-5, f"Norm {clipped_norm} should be <= 1.0")
        # Check scaling: 2.0 -> 1.0, so should be [1.0, 0.0]
        self.assertTrue(torch.allclose(clipped[0], torch.tensor([1.0, 0.0]), atol=1e-5))

    def test_noise_addition(self):
        # Test noise properties (stochastic, so probabilistic check or check non-equality)
        g1 = torch.zeros(1000)
        grads = [g1]
        
        noisy = self.dp_engine.add_noise(grads)
        
        # Mean should be close to 0, Std should be close to 0.1 * 1.0 = 0.1
        noisy_tensor = noisy[0]
        mean = torch.mean(noisy_tensor).item()
        std = torch.std(noisy_tensor).item()
        
        # These bounds are loose for stochastic tests but checking range is good sanity check
        self.assertTrue(abs(mean) < 0.05, f"Mean {mean} too far from 0")
        self.assertTrue(abs(std - 0.1) < 0.05, f"Std {std} too far from 0.1")
        
    def test_step(self):
        # Integration test of step()
        g1 = torch.tensor([10.0]) # Will be clipped to 1.0
        grads = [g1]
        
        output = self.dp_engine.step(grads)
        # Should be roughly 1.0 + noise
        # Since noise is small (0.1), value should be around 1.0
        # It's unlikely to be 10.0
        
        self.assertTrue(abs(output[0].item()) < 2.0, "Result should be clipped around 1.0")

if __name__ == '__main__':
    unittest.main()
