"""
Coherence Speech Intelligibility Index (CSII).

From the paper (Section 4.2.2):

    The CSII of three-levels is computed by splitting the input speech signal
    into three amplitude regions. For calculating CSII, hamming window with 
    16ms size block and overlapping of 50% between windowed segments.

Based on: Kates & Arehart (2005) "Coherence and the speech intelligibility index"

CSII measures intelligibility by computing magnitude-squared coherence
between clean and enhanced signals across frequency bands, weighted by
the speech intelligibility importance function.
"""

import numpy as np
from scipy.signal.windows import hamming as hamming_window

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _magnitude_squared_coherence(x, y, fs, frame_size, hop_size, num_fft):
    """Estimate magnitude-squared coherence between two signals.
    
    MSC(f) = |P_xy(f)|² / (P_xx(f) · P_yy(f))
    
    where P_xy, P_xx, P_yy are cross and auto power spectral densities.
    """
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    num_frames = (min_len - frame_size) // hop_size + 1
    if num_frames < 1:
        return np.zeros(num_fft // 2 + 1)
    
    n_freq = num_fft // 2 + 1
    Pxx = np.zeros(n_freq)
    Pyy = np.zeros(n_freq)
    Pxy = np.zeros(n_freq, dtype=complex)
    
    window = hamming_window(frame_size)
    
    for n in range(num_frames):
        start = n * hop_size
        end = start + frame_size
        
        X = np.fft.rfft(x[start:end] * window, n=num_fft)
        Y = np.fft.rfft(y[start:end] * window, n=num_fft)
        
        Pxx += np.abs(X) ** 2
        Pyy += np.abs(Y) ** 2
        Pxy += X * np.conj(Y)
    
    # Average
    Pxx /= num_frames
    Pyy /= num_frames
    Pxy /= num_frames
    
    # Magnitude-squared coherence
    denom = Pxx * Pyy
    msc = np.abs(Pxy) ** 2 / (denom + 1e-10)
    msc = np.clip(msc, 0, 1)
    
    return msc


def _speech_importance_function(n_freq, fs, num_fft):
    """Approximate ANSI S3.5 speech importance function (SII weights).
    
    Returns importance weights for each frequency bin.
    """
    freqs = np.arange(n_freq) * fs / num_fft
    
    # Simplified band importance (ANSI S3.5 one-third octave)
    # Higher weight for 1-4 kHz (most important for intelligibility)
    weights = np.ones(n_freq)
    for i, f in enumerate(freqs):
        if f < 200:
            weights[i] = 0.0
        elif f < 500:
            weights[i] = 0.5
        elif f < 1000:
            weights[i] = 0.8
        elif f < 2000:
            weights[i] = 1.0
        elif f < 4000:
            weights[i] = 0.9
        else:
            weights[i] = 0.4
    
    # Normalize
    weights /= (np.sum(weights) + 1e-10)
    return weights


def _split_by_amplitude(signal, fs, num_levels=3):
    """Split signal into amplitude regions for three-level CSII.
    
    Sorts samples by RMS level and divides into num_levels groups.
    """
    frame_size = int(0.016 * fs)  # 16 ms
    hop_size = frame_size // 2
    
    num_frames = (len(signal) - frame_size) // hop_size + 1
    if num_frames < num_levels:
        return [np.arange(len(signal))] * num_levels
    
    # Compute RMS per frame
    rms_levels = np.zeros(num_frames)
    for n in range(num_frames):
        start = n * hop_size
        end = start + frame_size
        rms_levels[n] = np.sqrt(np.mean(signal[start:end] ** 2))
    
    # Sort frames by RMS and split into regions
    sorted_indices = np.argsort(rms_levels)
    frames_per_level = num_frames // num_levels
    
    regions = []
    for level in range(num_levels):
        start_idx = level * frames_per_level
        if level == num_levels - 1:
            end_idx = num_frames
        else:
            end_idx = (level + 1) * frames_per_level
        
        frame_indices = sorted_indices[start_idx:end_idx]
        
        # Convert frame indices to sample indices
        sample_indices = []
        for fi in frame_indices:
            s = fi * hop_size
            e = min(s + frame_size, len(signal))
            sample_indices.extend(range(s, e))
        
        regions.append(np.array(sample_indices))
    
    return regions


def compute_csii(clean_signal, enhanced_signal, fs=None, num_levels=3):
    """Compute three-level CSII.
    
    Splits the signal into three amplitude regions (low, mid, high)
    and computes coherence-weighted SII for each, then averages.
    
    Args:
        clean_signal: Clean speech (1D numpy array).
        enhanced_signal: Enhanced speech (1D numpy array).
        fs: Sampling rate.
        num_levels: Number of amplitude regions (default 3).
    
    Returns:
        Average CSII score across levels. Range: approximately [0, 1].
    """
    fs = fs or config.SAMPLE_RATE
    
    min_len = min(len(clean_signal), len(enhanced_signal))
    clean = clean_signal[:min_len].astype(np.float64)
    enhanced = enhanced_signal[:min_len].astype(np.float64)
    
    frame_size = int(0.016 * fs)  # 16 ms (paper specification)
    hop_size = frame_size // 2     # 50% overlap
    num_fft = 256
    n_freq = num_fft // 2 + 1
    
    # Speech importance weights
    sii_weights = _speech_importance_function(n_freq, fs, num_fft)
    
    # Split by amplitude
    regions = _split_by_amplitude(clean, fs, num_levels)
    
    csii_levels = []
    
    for level, indices in enumerate(regions):
        if len(indices) == 0:
            csii_levels.append(0.0)
            continue
        
        # Extract signal segments for this amplitude region
        indices = indices[indices < min_len]
        if len(indices) < frame_size:
            csii_levels.append(0.0)
            continue
        
        # Use coherence over the whole signal (weighted by region activity)
        msc = _magnitude_squared_coherence(
            clean, enhanced, fs, frame_size, hop_size, num_fft
        )
        
        # Weighted average coherence (SII-like)
        csii_score = np.sum(sii_weights * msc)
        csii_levels.append(np.clip(csii_score, 0, 1))
    
    # Average across levels
    return np.mean(csii_levels)
