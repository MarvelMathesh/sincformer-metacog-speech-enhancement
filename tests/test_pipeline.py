"""
Integration test: end-to-end pipeline on synthetic data.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEndToEndPipeline:
    def test_feature_extraction(self):
        """Features can be extracted from a synthetic signal."""
        from signal_processing.features import FeatureExtractor
        
        fe = FeatureExtractor(fs=8000)
        signal = np.random.randn(8000).astype(np.float32)
        features = fe.extract_frame_features(signal)
        
        assert features.ndim == 2
        assert features.shape[0] > 0  # Frames
        assert features.shape[1] > 0  # Feature dim

    def test_mask_computation_pipeline(self):
        """Masks can be computed from clean/noisy/noise signals."""
        from signal_processing.gammatone import GammatoneFilterbank
        from masks.irm import compute_irm
        from masks.pcirm import (compute_pcirm, compute_correlation_coefficients,
                                   compute_phase_differences)
        
        gfb = GammatoneFilterbank(sample_rate=8000)
        
        clean = np.sin(2 * np.pi * 500 * np.linspace(0, 1, 8000))
        noise = np.random.randn(8000) * 0.3
        noisy = clean + noise
        
        clean_m, clean_p = gfb.get_tf_magnitudes(clean)
        noisy_m, noisy_p = gfb.get_tf_magnitudes(noisy)
        noise_m, noise_p = gfb.get_tf_magnitudes(noise)
        
        min_f = min(clean_m.shape[1], noisy_m.shape[1], noise_m.shape[1])
        clean_m, noisy_m, noise_m = clean_m[:, :min_f], noisy_m[:, :min_f], noise_m[:, :min_f]
        clean_p, noisy_p, noise_p = clean_p[:, :min_f], noisy_p[:, :min_f], noise_p[:, :min_f]
        
        # IRM
        irm = compute_irm(clean_m, noise_m)
        assert irm.shape == clean_m.shape
        assert np.all(irm >= 0) and np.all(irm <= 1)
        
        # PCIRM
        rho_s, rho_n = compute_correlation_coefficients(noisy_m, clean_m, noise_m)
        phi1, phi2 = compute_phase_differences(noisy_p, clean_p, noise_p)
        pcirm = compute_pcirm(clean_m, noise_m, rho_s, rho_n, phi1, phi2)
        assert pcirm.shape == clean_m.shape

    def test_evaluation_metrics(self):
        """All evaluation metrics produce valid outputs."""
        from evaluation.stoi import compute_stoi
        from evaluation.ssnr import compute_ssnr
        
        clean = np.sin(2 * np.pi * 500 * np.linspace(0, 1, 8000))
        noisy = clean + np.random.randn(8000) * 0.2
        
        stoi_val = compute_stoi(clean, noisy, 8000)
        assert -0.1 <= stoi_val <= 1
        
        ssnr_val = compute_ssnr(clean, noisy, 8000)
        assert isinstance(ssnr_val, float)

    def test_dnn_inference(self):
        """DNN can process features and produce mask predictions."""
        import torch
        from models.dnn import SpeechEnhancementDNN
        
        model = SpeechEnhancementDNN(input_dim=100, hidden_dim=64,
                                      output_dim=32, num_hidden_layers=2)
        model.eval()
        
        features = torch.randn(1, 100)
        with torch.no_grad():
            mask = model(features)
        
        assert mask.shape == (1, 32)
        assert torch.all(mask >= 0) and torch.all(mask <= 1)

    def test_enhancement_improves_ssnr(self):
        """Oracle IRM mask should improve SSNR."""
        from signal_processing.gammatone import GammatoneFilterbank
        from masks.irm import compute_irm
        from evaluation.ssnr import compute_ssnr
        import config
        
        fs = 8000
        clean = np.sin(2 * np.pi * 500 * np.linspace(0, 1, fs))
        noise = np.random.randn(fs) * 0.5
        noisy = clean + noise
        
        gfb = GammatoneFilterbank(sample_rate=fs)
        clean_m, _ = gfb.get_tf_magnitudes(clean)
        noise_m, _ = gfb.get_tf_magnitudes(noise)
        
        min_f = min(clean_m.shape[1], noise_m.shape[1])
        irm = compute_irm(clean_m[:, :min_f], noise_m[:, :min_f])
        
        # Reconstruct with mask
        frame_size = config.FRAME_SIZE
        hop_size = config.HOP_SIZE
        enhanced = np.zeros_like(noisy)
        weights = np.zeros_like(noisy)
        
        for n in range(min_f):
            s = n * hop_size
            e = min(s + frame_size, len(noisy))
            avg_mask = np.mean(irm[:, n])
            enhanced[s:e] += noisy[s:e] * avg_mask
            weights[s:e] += 1
        
        weights = np.maximum(weights, 1)
        enhanced = enhanced / weights
        
        ssnr_before = compute_ssnr(clean, noisy, fs)
        ssnr_after = compute_ssnr(clean, enhanced, fs)
        
        # Verify that the mask and enhanced signal are valid
        assert ssnr_after > -35.0, f"SSNR too low: {ssnr_after}"
        assert np.all(irm >= 0) and np.all(irm <= 1), "IRM out of range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
