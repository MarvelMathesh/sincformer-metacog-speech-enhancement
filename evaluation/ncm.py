"""
Normalized Covariance Metric (NCM).

From the paper (Section 4.2.3):

    The covariance between the input and output envelope signals is measured
    using NCM. It is usually calculated for each channel in the gammatone 
    filterbank using the Hilbert transform to generate the speech 
    temporal envelopes.

Based on: Holube & Kollmeier (1996) "Speech intelligibility prediction 
in hearing impaired listeners based on a psychoacoustically motivated 
perception model"
"""

import numpy as np
from scipy.signal import hilbert

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from signal_processing.gammatone import GammatoneFilterbank


def _compute_envelope(signal):
    """Compute temporal envelope using Hilbert transform.
    
    The analytic signal's magnitude gives the instantaneous envelope.
    """
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    return envelope


def _normalized_covariance(env_clean, env_enhanced):
    """Compute normalized covariance between two envelopes.
    
    NCC = cov(x, y) / sqrt(var(x) · var(y))
    
    Range: [-1, 1], where 1 = perfect correlation.
    """
    # Remove means
    x = env_clean - np.mean(env_clean)
    y = env_enhanced - np.mean(env_enhanced)
    
    cov_xy = np.mean(x * y)
    var_x = np.mean(x ** 2)
    var_y = np.mean(y ** 2)
    
    denom = np.sqrt(var_x * var_y)
    if denom < 1e-10:
        return 0.0
    
    ncc = cov_xy / denom
    return np.clip(ncc, -1, 1)


def compute_ncm(clean_signal, enhanced_signal, fs=None, gfb=None):
    """Compute Normalized Covariance Metric.
    
    Process for each gammatone channel:
      1. Filter clean and enhanced signals through gammatone filterbank
      2. Extract temporal envelope via Hilbert transform
      3. Compute normalized covariance between envelopes
      4. Average across channels with importance weighting
    
    Args:
        clean_signal: Clean speech (1D numpy array).
        enhanced_signal: Enhanced speech (1D numpy array).
        fs: Sampling rate.
        gfb: Pre-initialized GammatoneFilterbank (optional).
    
    Returns:
        NCM score. Range: approximately [0, 1] for speech.
    """
    fs = fs or config.SAMPLE_RATE
    
    # Ensure same length
    min_len = min(len(clean_signal), len(enhanced_signal))
    clean = clean_signal[:min_len].astype(np.float64)
    enhanced = enhanced_signal[:min_len].astype(np.float64)
    
    if min_len < 64:
        return 0.0
    
    # Apply gammatone filterbank
    if gfb is None:
        gfb = GammatoneFilterbank(sample_rate=fs)
    
    clean_filtered = gfb.filter(clean)
    enhanced_filtered = gfb.filter(enhanced)
    
    num_channels = clean_filtered.shape[0]
    
    # Compute normalized covariance per channel
    channel_ncm = np.zeros(num_channels)
    
    for i in range(num_channels):
        # Extract temporal envelopes via Hilbert transform
        env_clean = _compute_envelope(clean_filtered[i, :])
        env_enhanced = _compute_envelope(enhanced_filtered[i, :])
        
        # Normalized covariance
        channel_ncm[i] = _normalized_covariance(env_clean, env_enhanced)
    
    # Importance weighting (frequency-dependent)
    # Higher weight for speech-dominant frequency bands (300-3400 Hz)
    weights = np.ones(num_channels)
    for i, cf in enumerate(gfb.center_freqs):
        if cf < 300:
            weights[i] = 0.3
        elif cf < 1000:
            weights[i] = 0.8
        elif cf < 3400:
            weights[i] = 1.0
        else:
            weights[i] = 0.5
    
    weights /= np.sum(weights)
    
    # Weighted average NCM
    ncm = np.sum(weights * np.maximum(channel_ncm, 0))
    
    return np.clip(ncm, 0.0, 1.0)
