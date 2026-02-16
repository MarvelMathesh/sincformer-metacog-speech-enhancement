"""
Segmental Signal-to-Noise Ratio (SSNR).

Implements Eq. 15 from the paper:

    SSNR = (1/F) * Σ 10·log10( Σ cs²(k) / Σ [cs(k) - ê_s(k)]² )

where:
    F   — number of frames
    L   — frame length  
    k   — sample index within frame
    cs  — clean speech
    ê_s — enhanced speech

SSNR measures noise reduction quality by computing per-frame SNR
and averaging. Higher SSNR = better noise suppression.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_ssnr(clean_signal, enhanced_signal, fs=None,
                  frame_size=None, hop_size=None,
                  upper_bound=35.0, lower_bound=-10.0):
    """Compute Segmental Signal-to-Noise Ratio.
    
    Args:
        clean_signal: Clean speech (1D numpy array).
        enhanced_signal: Enhanced speech (1D numpy array).
        fs: Sampling rate.
        frame_size: Samples per frame (L).
        hop_size: Hop between frames.
        upper_bound: Maximum per-frame SNR (to avoid inf in silent frames).
        lower_bound: Minimum per-frame SNR.
    
    Returns:
        SSNR value in dB. Higher = better noise suppression.
    """
    fs = fs or config.SAMPLE_RATE
    frame_size = frame_size or config.FRAME_SIZE
    hop_size = hop_size or config.HOP_SIZE
    
    # Ensure same length
    min_len = min(len(clean_signal), len(enhanced_signal))
    clean = clean_signal[:min_len].astype(np.float64)
    enhanced = enhanced_signal[:min_len].astype(np.float64)
    
    # Frame the signals
    num_frames = (min_len - frame_size) // hop_size + 1
    
    if num_frames < 1:
        return 0.0
    
    frame_snrs = []
    
    for n in range(num_frames):
        start = n * hop_size
        end = start + frame_size
        
        clean_frame = clean[start:end]
        enhanced_frame = enhanced[start:end]
        
        # Speech power
        speech_power = np.sum(clean_frame ** 2)
        
        # Error power (difference between clean and enhanced)
        error = clean_frame - enhanced_frame
        error_power = np.sum(error ** 2)
        
        # Skip silence frames (very low energy)
        if speech_power < 1e-10:
            continue
        
        if error_power < 1e-10:
            snr = upper_bound
        else:
            snr = 10 * np.log10(speech_power / error_power)
        
        # Clip to bounds (Eq. 15 doesn't specify bounds, but standard practice)
        snr = np.clip(snr, lower_bound, upper_bound)
        frame_snrs.append(snr)
    
    if len(frame_snrs) == 0:
        return 0.0
    
    # Average across frames (Eq. 15)
    ssnr = np.mean(frame_snrs)
    return ssnr


def compute_ssnr_improvement(clean_signal, noisy_signal, enhanced_signal,
                              fs=None):
    """Compute SSNR improvement (output SSNR - input SSNR).
    
    Args:
        clean_signal: Clean speech.
        noisy_signal: Noisy speech (input to enhancement system).
        enhanced_signal: Enhanced speech (output of enhancement system).
        fs: Sampling rate.
    
    Returns:
        SSNR improvement in dB.
    """
    input_ssnr = compute_ssnr(clean_signal, noisy_signal, fs)
    output_ssnr = compute_ssnr(clean_signal, enhanced_signal, fs)
    
    return output_ssnr - input_ssnr
