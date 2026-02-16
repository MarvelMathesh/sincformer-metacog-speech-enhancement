"""
Unit tests for IRM, PCIRM, and OPT-PCIRM masks.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from masks.irm import compute_irm, apply_irm
from masks.pcirm import (compute_pcirm, compute_correlation_coefficients,
                           compute_phase_differences)
from masks.opt_pcirm import compute_snr_boundaries, quantize_pcirm


class TestIRM:
    def test_output_range(self):
        clean = np.random.rand(64, 100) + 0.01
        noise = np.random.rand(64, 100) + 0.01
        irm = compute_irm(clean, noise)
        assert np.all(irm >= 0) and np.all(irm <= 1)

    def test_pure_speech(self):
        """When noise is zero, IRM should be ~1."""
        clean = np.ones((64, 50))
        noise = np.zeros((64, 50))
        irm = compute_irm(clean, noise)
        assert np.all(irm > 0.99)

    def test_pure_noise(self):
        """When speech is zero, IRM should be ~0."""
        clean = np.zeros((64, 50))
        noise = np.ones((64, 50))
        irm = compute_irm(clean, noise)
        assert np.all(irm < 0.01)

    def test_apply_irm(self):
        noisy = np.random.rand(64, 50)
        irm = np.ones((64, 50)) * 0.5
        enhanced = apply_irm(noisy, irm)
        np.testing.assert_allclose(enhanced, noisy * 0.5)


class TestCorrelation:
    def test_correlation_range(self):
        noisy = np.random.rand(64, 100) + 0.1
        clean = np.random.rand(64, 100) + 0.1
        noise = np.random.rand(64, 100) + 0.1
        rho_s, rho_n = compute_correlation_coefficients(noisy, clean, noise)
        assert np.all(rho_s >= 0) and np.all(rho_s <= 1)
        assert np.all(rho_n >= 0) and np.all(rho_n <= 1)


class TestPCIRM:
    def test_output_range(self):
        clean_mag = np.random.rand(64, 50) + 0.01
        noise_mag = np.random.rand(64, 50) + 0.01
        rho_s = np.random.rand(64, 50)
        rho_n = np.random.rand(64, 50)
        phi1 = np.random.rand(64, 50) * np.pi
        phi2 = np.random.rand(64, 50) * np.pi
        pcirm = compute_pcirm(clean_mag, noise_mag, rho_s, rho_n, phi1, phi2)
        assert np.all(pcirm >= 0) and np.all(pcirm <= 1)

    def test_shape_preserved(self):
        shape = (32, 80)
        clean_mag = np.random.rand(*shape) + 0.01
        noise_mag = np.random.rand(*shape) + 0.01
        rho_s = np.ones(shape)
        rho_n = np.ones(shape) * 0.5
        phi1 = np.zeros(shape)
        phi2 = np.zeros(shape)
        pcirm = compute_pcirm(clean_mag, noise_mag, rho_s, rho_n, phi1, phi2)
        assert pcirm.shape == shape


class TestOPTPCIRM:
    def test_snr_boundaries(self):
        step_values, exponent = compute_snr_boundaries()
        assert len(step_values) == 3  # M=3
        assert step_values[0] == 0.0  # First step is 0
        assert exponent > 0

    def test_quantize_produces_discrete(self):
        pcirm = np.random.rand(64, 50)
        step_values, _ = compute_snr_boundaries()
        opt = quantize_pcirm(pcirm, step_values)
        unique_vals = np.unique(opt)
        assert len(unique_vals) <= 3  # At most M=3 values

    def test_quantize_range(self):
        pcirm = np.random.rand(64, 50)
        step_values, _ = compute_snr_boundaries()
        opt = quantize_pcirm(pcirm, step_values)
        assert np.all(opt >= 0) and np.all(opt <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
