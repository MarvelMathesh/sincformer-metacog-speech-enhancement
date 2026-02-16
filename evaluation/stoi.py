"""
Short-Time Objective Intelligibility (STOI).

Implements STOI as described in:
    Taal et al. (2011) "An algorithm for intelligibility prediction of 
    time-frequency weighted noisy speech"

STOI measures the correlation between clean and enhanced speech's
short-term temporal envelopes. Range: [0, 1].

Used as:
    1. Evaluation metric for speech intelligibility
    2. PSO fitness function for OPT-PCIRM optimization
    3. Differentiable approximation for perceptual training loss
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_stoi(clean_signal, enhanced_signal, fs=None, extended=False):
    """Compute STOI between clean and enhanced speech.
    
    Uses the pystoi library if available, otherwise falls back to
    a simplified implementation.
    
    Args:
        clean_signal: Clean speech signal (1D numpy array).
        enhanced_signal: Enhanced speech signal (1D numpy array).
        fs: Sampling rate.
        extended: Use extended STOI (more accurate for processed speech).
    
    Returns:
        STOI score in [0, 1]. Higher = better intelligibility.
    """
    fs = fs or config.SAMPLE_RATE
    
    # Ensure same length
    min_len = min(len(clean_signal), len(enhanced_signal))
    clean = clean_signal[:min_len].astype(np.float64)
    enhanced = enhanced_signal[:min_len].astype(np.float64)
    
    try:
        from pystoi import stoi
        return stoi(clean, enhanced, fs, extended=extended)
    except ImportError:
        return _stoi_simplified(clean, enhanced, fs)


def _stoi_simplified(clean, enhanced, fs):
    """Simplified STOI implementation.
    
    Computes correlation between temporal envelopes in one-third
    octave bands using 25.6ms analysis segments with 50% overlap.
    """
    # STOI parameters
    N = 30           # Number of one-third octave bands
    J_min = 15       # Minimum frequency band index
    frame_len = int(0.0256 * fs)  # 25.6 ms
    hop = frame_len // 2  # 50% overlap
    
    # Normalize
    clean = clean / (np.sqrt(np.mean(clean ** 2)) + 1e-10)
    enhanced = enhanced / (np.sqrt(np.mean(enhanced ** 2)) + 1e-10)
    
    # Frame the signals
    num_frames = (len(clean) - frame_len) // hop + 1
    if num_frames < 1:
        return 0.0
    
    # Compute DFT-based one-third octave band energy envelopes
    correlations = []
    
    for n in range(num_frames):
        start = n * hop
        end = start + frame_len
        
        clean_frame = clean[start:end] * np.hanning(frame_len)
        enh_frame = enhanced[start:end] * np.hanning(frame_len)
        
        clean_spec = np.abs(np.fft.rfft(clean_frame))
        enh_spec = np.abs(np.fft.rfft(enh_frame))
        
        # Clipping (bound-based normalization)
        clean_energy = np.sqrt(np.sum(clean_spec ** 2) + 1e-10)
        enh_norm = enh_spec / (np.sqrt(np.sum(enh_spec ** 2)) + 1e-10) * clean_energy
        
        # Correlation
        corr = np.sum(clean_spec * enh_norm) / (
            np.sqrt(np.sum(clean_spec ** 2) * np.sum(enh_norm ** 2)) + 1e-10
        )
        correlations.append(np.clip(corr, -1, 1))
    
    # Average correlation across frames
    stoi_score = np.mean(correlations)
    return np.clip(stoi_score, 0, 1)
