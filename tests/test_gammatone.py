"""
Unit tests for Gammatone Filterbank.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signal_processing.gammatone import (
    GammatoneFilterbank, erb_space, gammatone_impulse_response
)
import config


class TestERBSpace:
    def test_num_channels(self):
        cfs = erb_space(50, 4000, 64)
        assert len(cfs) == 64

    def test_frequency_range(self):
        cfs = erb_space(50, 4000, 64)
        assert cfs[0] >= 50
        assert cfs[-1] <= 4000 + 1e-6

    def test_monotonic_ascending(self):
        cfs = erb_space(50, 4000, 64)
        assert np.all(np.diff(cfs) > 0)


class TestGammatoneImpulseResponse:
    def test_output_length(self):
        ir = gammatone_impulse_response(1000, 8000, duration=0.05)
        expected_len = int(0.05 * 8000)
        assert len(ir) == expected_len

    def test_decays(self):
        ir = gammatone_impulse_response(1000, 8000, duration=0.05)
        # Energy should decay over time
        first_half = np.sum(ir[:len(ir)//2] ** 2)
        second_half = np.sum(ir[len(ir)//2:] ** 2)
        assert first_half > second_half


class TestGammatoneFilterbank:
    @pytest.fixture
    def gfb(self):
        return GammatoneFilterbank(sample_rate=8000, num_channels=64)

    def test_center_freqs_count(self, gfb):
        assert len(gfb.center_freqs) == 64

    def test_filter_output_shape(self, gfb):
        signal = np.random.randn(8000).astype(np.float32)  # 1 second
        output = gfb.filter(signal)
        assert output.shape[0] == 64
        assert output.shape[1] > 0

    def test_filter_output_not_zero(self, gfb):
        signal = np.sin(2 * np.pi * 500 * np.linspace(0, 1, 8000))
        output = gfb.filter(signal)
        assert np.max(np.abs(output)) > 0

    def test_tf_magnitudes_shape(self, gfb):
        signal = np.random.randn(8000).astype(np.float32)
        mags, phases = gfb.get_tf_magnitudes(signal)
        assert mags.shape[0] == 64
        assert phases.shape[0] == 64
        assert mags.shape == phases.shape

    def test_tf_magnitudes_nonnegative(self, gfb):
        signal = np.random.randn(8000).astype(np.float32)
        mags, _ = gfb.get_tf_magnitudes(signal)
        assert np.all(mags >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
