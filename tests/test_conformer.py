"""
Unit tests for Complex Conformer.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.conformer import ComplexConformer


class TestComplexConformer:
    @pytest.fixture
    def conformer(self):
        return ComplexConformer(
            n_freq=32, d_model=64, num_blocks=2,
            num_heads=4, d_ff=128, kernel_size=7, dropout=0.0
        )

    def test_forward_shape(self, conformer):
        batch, time, n_freq = 2, 20, 32
        stft_real = torch.randn(batch, time, n_freq)
        stft_imag = torch.randn(batch, time, n_freq)

        mask_real, mask_imag = conformer(stft_real, stft_imag)
        assert mask_real.shape == (batch, time, n_freq)
        assert mask_imag.shape == (batch, time, n_freq)

    def test_complex_mask_application(self, conformer):
        batch, time, n_freq = 2, 10, 32
        stft_r = torch.randn(batch, time, n_freq)
        stft_i = torch.randn(batch, time, n_freq)

        mask_r, mask_i = conformer(stft_r, stft_i)
        enh_r, enh_i = conformer.apply_mask(stft_r, stft_i, mask_r, mask_i)

        assert enh_r.shape == stft_r.shape
        assert enh_i.shape == stft_i.shape

    def test_gradient_flow(self, conformer):
        stft_r = torch.randn(1, 10, 32, requires_grad=True)
        stft_i = torch.randn(1, 10, 32, requires_grad=True)

        mask_r, mask_i = conformer(stft_r, stft_i)
        loss = mask_r.sum() + mask_i.sum()
        loss.backward()

        assert stft_r.grad is not None

    def test_parameter_count(self, conformer):
        count = conformer.count_parameters()
        assert count > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
