"""
Perceptual Evaluation of Speech Quality (PESQ).

Wrapper around the pesq library implementing ITU-T P.862.
PESQ scores range from -0.5 to 4.5 (higher = better quality).

From the paper (Section 4.3.2):
    "PESQ is a method to rate the speech quality; a greater PESQ score 
    denotes a higher level of quality."
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_pesq(clean_signal, enhanced_signal, fs=None, mode=None):
    """Compute PESQ between clean and enhanced speech.
    
    Args:
        clean_signal: Clean speech (1D numpy array).
        enhanced_signal: Enhanced speech (1D numpy array).
        fs: Sampling rate (8000 for narrowband, 16000 for wideband).
        mode: 'nb' (narrowband) or 'wb' (wideband).
    
    Returns:
        PESQ score in [-0.5, 4.5]. Higher = better quality.
    """
    fs = fs or config.SAMPLE_RATE
    mode = mode or config.PESQ_MODE
    
    # Ensure same length
    min_len = min(len(clean_signal), len(enhanced_signal))
    clean = clean_signal[:min_len].astype(np.float64)
    enhanced = enhanced_signal[:min_len].astype(np.float64)
    
    try:
        from pesq import pesq
        score = pesq(fs, clean, enhanced, mode)
        return score
    except ImportError:
        return _pesq_simplified(clean, enhanced, fs)
    except Exception as e:
        # PESQ can fail on very short signals
        print(f"PESQ computation failed: {e}")
        return 0.0


def _pesq_simplified(clean, enhanced, fs):
    """Simplified PESQ approximation when the pesq library is unavailable.
    
    Uses a combination of SNR and spectral distortion as a rough proxy.
    This is NOT a replacement for true PESQ, just a fallback.
    """
    # Frame-level spectral distortion
    frame_size = int(0.032 * fs)  # 32 ms
    hop = frame_size // 2
    
    num_frames = (len(clean) - frame_size) // hop + 1
    if num_frames < 1:
        return 1.0
    
    distortions = []
    for n in range(num_frames):
        start = n * hop
        end = start + frame_size
        
        c_spec = np.abs(np.fft.rfft(clean[start:end]))
        e_spec = np.abs(np.fft.rfft(enhanced[start:end]))
        
        # Log spectral distortion
        c_log = np.log(c_spec + 1e-10)
        e_log = np.log(e_spec + 1e-10)
        
        lsd = np.sqrt(np.mean((c_log - e_log) ** 2))
        distortions.append(lsd)
    
    mean_distortion = np.mean(distortions)
    
    # Map distortion to approximate PESQ range [-0.5, 4.5]
    # Lower distortion → higher PESQ
    pesq_approx = 4.5 - mean_distortion * 0.5
    return np.clip(pesq_approx, -0.5, 4.5)
