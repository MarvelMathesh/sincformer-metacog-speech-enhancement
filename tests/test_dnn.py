"""
Unit tests for DNN model.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dnn import SpeechEnhancementDNN, create_dnn


class TestDNN:
    def test_forward_shape(self):
        model = SpeechEnhancementDNN(input_dim=594, hidden_dim=256,
                                      output_dim=64, num_hidden_layers=3)
        x = torch.randn(16, 594)
        y = model(x)
        assert y.shape == (16, 64)

    def test_output_range(self):
        """Sigmoid output should be in [0, 1]."""
        model = SpeechEnhancementDNN(input_dim=100, hidden_dim=64,
                                      output_dim=32, num_hidden_layers=2)
        x = torch.randn(8, 100)
        y = model(x)
        assert torch.all(y >= 0) and torch.all(y <= 1)

    def test_gradient_flow(self):
        """Gradients should flow from loss to all parameters."""
        model = SpeechEnhancementDNN(input_dim=50, hidden_dim=32,
                                      output_dim=16, num_hidden_layers=2)
        x = torch.randn(4, 50)
        target = torch.rand(4, 16)
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.any(param.grad != 0)

    def test_create_dnn_factory(self):
        model = create_dnn(feature_dim=594, mask_dim=64)
        assert model.input_dim == 594
        assert model.output_dim == 64

    def test_count_parameters(self):
        model = SpeechEnhancementDNN(input_dim=100, hidden_dim=64,
                                      output_dim=32, num_hidden_layers=2)
        count = model.count_parameters()
        assert count > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
