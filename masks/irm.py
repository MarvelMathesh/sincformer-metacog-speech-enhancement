"""
Ideal Ratio Mask (IRM) — Baseline mask for speech enhancement.

Implements Eq. 4 from the paper:

    Z_IRM(i, n) = (|Cs(i,n)|² / (|Cs(i,n)|² + |Zn(i,n)|²))^p

where p=0.5 is the tunable parameter.

The IRM operates only in the magnitude domain (no phase, no correlation).
Values range from 0 (noise-dominant) to 1 (speech-dominant).
"""

import numpy as np


def compute_irm(clean_mag, noise_mag, p=0.5, eps=1e-10):
    """Compute the Ideal Ratio Mask.
    
    Args:
        clean_mag: |Cs(i,n)|² — clean speech power per T-F unit.
                   Shape: (num_channels, num_frames) or (num_freq_bins, num_frames).
        noise_mag: |Zn(i,n)|² — noise power per T-F unit.
                   Same shape as clean_mag.
        p:  Tunable exponent parameter (default 0.5).
        eps: Small constant for numerical stability.
    
    Returns:
        IRM values in [0, 1], same shape as input.
    """
    clean_power = np.abs(clean_mag) ** 2 if clean_mag.dtype != complex else clean_mag
    noise_power = np.abs(noise_mag) ** 2 if noise_mag.dtype != complex else noise_mag
    
    ratio = clean_power / (clean_power + noise_power + eps)
    irm = ratio ** p
    
    return np.clip(irm, 0.0, 1.0)


def apply_irm(noisy_tf, irm):
    """Apply IRM to noisy T-F representation to recover clean speech estimate.
    
    Enhanced = IRM ⊙ Noisy
    
    Args:
        noisy_tf: Noisy speech in T-F domain.
        irm: Ideal Ratio Mask values.
    
    Returns:
        Enhanced T-F representation.
    """
    return noisy_tf * irm
