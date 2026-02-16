"""
Phase Correlation Ideal Ratio Mask (PCIRM) — Novel soft mask.

Implements Eq. 5-7 from the paper:

    Z_PCIRM(i,n) = ρs·|Cs·cos(φ1)|² / (ρs·|Cs·cos(φ1)|² + ρn·|Zn·cos(φ2)|²)

where:
    ρs(i,n) = ns(i,n)^T · cs(i,n) / sqrt(||ns||² · ||cs||²)   (Eq. 7)
    ρn(i,n) = ns(i,n)^T · zn(i,n) / sqrt(||ns||² · ||zn||²)   (Eq. 6)
    φ1 = phase difference between Cs and Ns
    φ2 = phase difference between noise and Ns

PCIRM incorporates magnitude domain, inter-channel correlation, AND
phase difference — addressing the limitations of both BIRMP (no correlation)
and pRM (no phase).
"""

import numpy as np


def compute_correlation_coefficients(noisy_frames, clean_frames, noise_frames,
                                      eps=1e-10):
    """Compute normalized cross-correlation coefficients ρs and ρn.
    
    ρs(i,n) measures correlation between noisy speech and clean speech.
    ρn(i,n) measures correlation between noisy speech and noise.
    
    When ρs is high → T-F unit is speech-dominant.
    When ρn is high → T-F unit is noise-dominant.
    
    Args:
        noisy_frames: ns(i,n) — noisy speech magnitude per T-F unit.
                      Shape: (num_channels, num_frames).
        clean_frames: cs(i,n) — clean speech magnitude per T-F unit.
        noise_frames: zn(i,n) — noise magnitude per T-F unit.
        eps: Numerical stability constant.
    
    Returns:
        Tuple (rho_s, rho_n):
            rho_s: (num_channels, num_frames) — speech correlation
            rho_n: (num_channels, num_frames) — noise correlation
    """
    # Eq. 7: ρs(i,n) = ns^T · cs / sqrt(||ns||² · ||cs||²)
    ns_norm = np.sqrt(np.sum(noisy_frames ** 2, axis=-1, keepdims=True) + eps)
    cs_norm = np.sqrt(np.sum(clean_frames ** 2, axis=-1, keepdims=True) + eps)
    
    if noisy_frames.ndim == 2:
        # Frame-level: each element is already a scalar per (i,n)
        # Use element-wise normalized product
        rho_s = (noisy_frames * clean_frames) / (
            np.sqrt(noisy_frames ** 2 + eps) * np.sqrt(clean_frames ** 2 + eps)
        )
        rho_n = (noisy_frames * noise_frames) / (
            np.sqrt(noisy_frames ** 2 + eps) * np.sqrt(noise_frames ** 2 + eps)
        )
    else:
        # 3D case: (channels, frames, frame_samples)
        inner_s = np.sum(noisy_frames * clean_frames, axis=-1)
        norm_ns = np.sqrt(np.sum(noisy_frames ** 2, axis=-1) + eps)
        norm_cs = np.sqrt(np.sum(clean_frames ** 2, axis=-1) + eps)
        rho_s = inner_s / (norm_ns * norm_cs)
        
        inner_n = np.sum(noisy_frames * noise_frames, axis=-1)
        norm_zn = np.sqrt(np.sum(noise_frames ** 2, axis=-1) + eps)
        rho_n = inner_n / (norm_ns * norm_zn)
    
    # Clip to [0, 1] (correlation coefficients are non-negative in this context)
    rho_s = np.clip(np.abs(rho_s), 0, 1)
    rho_n = np.clip(np.abs(rho_n), 0, 1)
    
    return rho_s, rho_n


def compute_phase_differences(noisy_phase, clean_phase, noise_phase):
    """Compute phase differences φ1 and φ2.
    
    φ1 = phase(Cs) - phase(Ns) — PD between clean and noisy
    φ2 = phase(Zn) - phase(Ns) — PD between noise and noisy
    
    Args:
        noisy_phase: Phase of noisy speech per T-F unit. Shape: (C, F).
        clean_phase: Phase of clean speech per T-F unit.
        noise_phase: Phase of noise per T-F unit.
    
    Returns:
        Tuple (phi1, phi2) — phase differences.
    """
    phi1 = clean_phase - noisy_phase
    phi2 = noise_phase - noisy_phase
    
    return phi1, phi2


def compute_pcirm(clean_mag, noise_mag, rho_s, rho_n, phi1, phi2, eps=1e-10):
    """Compute the Phase Correlation Ideal Ratio Mask (PCIRM).
    
    Eq. 5:
    Z_PCIRM(i,n) = ρs·|Cs·cos(φ1)|² / (ρs·|Cs·cos(φ1)|² + ρn·|Zn·cos(φ2)|²)
    
    The PCIRM combines three domains:
      - Magnitude: |Cs|² and |Zn|²
      - Correlation: ρs and ρn (which domain dominates the noisy mix)
      - Phase: cos(φ1) and cos(φ2) (phase alignment between signals)
    
    Args:
        clean_mag: |Cs(i,n)| — clean speech magnitude. Shape: (C, F).
        noise_mag: |Zn(i,n)| — noise magnitude.
        rho_s: ρs(i,n) — speech-noisy correlation coefficient.
        rho_n: ρn(i,n) — noise-noisy correlation coefficient.
        phi1: φ1 — phase difference between clean speech and noisy speech.
        phi2: φ2 — phase difference between noise and noisy speech.
        eps: Numerical stability constant.
    
    Returns:
        PCIRM values in [0, 1], same shape as input.
    """
    # Speech component: ρs · |Cs · cos(φ1)|²
    speech_component = rho_s * (np.abs(clean_mag) * np.abs(np.cos(phi1))) ** 2
    
    # Noise component: ρn · |Zn · cos(φ2)|²
    noise_component = rho_n * (np.abs(noise_mag) * np.abs(np.cos(phi2))) ** 2
    
    # PCIRM (Eq. 5)
    pcirm = speech_component / (speech_component + noise_component + eps)
    
    return np.clip(pcirm, 0.0, 1.0)


def compute_pcirm_from_signals(noisy_frames, clean_frames, noise_frames,
                                noisy_phase, clean_phase, noise_phase,
                                clean_mag, noise_mag, eps=1e-10):
    """Compute PCIRM from raw signal components — convenience function.
    
    Combines correlation computation (Eq. 6-7), phase differences,
    and the final PCIRM (Eq. 5) into a single call.
    
    Args:
        noisy_frames: Noisy speech in T-F domain (for correlation).
        clean_frames: Clean speech in T-F domain (for correlation).
        noise_frames: Noise in T-F domain (for correlation).
        noisy_phase: Phase of noisy speech.
        clean_phase: Phase of clean speech.
        noise_phase: Phase of noise.
        clean_mag: |Cs(i,n)| magnitude for mask computation.
        noise_mag: |Zn(i,n)| magnitude for mask computation.
        eps: Numerical stability.
    
    Returns:
        Tuple (pcirm, rho_s, rho_n, phi1, phi2).
    """
    rho_s, rho_n = compute_correlation_coefficients(noisy_frames, clean_frames,
                                                     noise_frames, eps)
    phi1, phi2 = compute_phase_differences(noisy_phase, clean_phase, noise_phase)
    
    pcirm = compute_pcirm(clean_mag, noise_mag, rho_s, rho_n, phi1, phi2, eps)
    
    return pcirm, rho_s, rho_n, phi1, phi2


def apply_pcirm(noisy_tf, pcirm):
    """Apply PCIRM to noisy T-F representation.
    
    Enhanced = PCIRM ⊙ Noisy
    
    Args:
        noisy_tf: Noisy speech in T-F domain.
        pcirm: Phase Correlation Ideal Ratio Mask.
    
    Returns:
        Enhanced T-F representation.
    """
    return noisy_tf * pcirm
