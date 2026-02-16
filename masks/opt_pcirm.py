"""
Optimal PCIRM (OPT-PCIRM) — Hard mask with PSO-optimized quantization.

Implements Eq. 8-13 from the paper:

1. Reframe PCIRM in terms of SNR (Eq. 8)
2. Define SNR boundaries using local criterion LC=-15 dB (Eq. 9-10)
3. Compute M=3 attenuation step values s_m(i,n) = ((m-1)/M)^n
4. Assign each T-F unit one of 3 values based on PCIRM range (Eq. 11)
5. Optimize the middle step x(i,n) using PSO with STOI fitness (Eq. 12-13)

OPT-PCIRM is a hard mask (discrete steps) optimized for intelligibility,
unlike PCIRM which is a soft mask optimized for quality.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from optimizer.pso import ParticleSwarmOptimizer


def compute_snr_boundaries(local_criterion_db=None, num_steps=None):
    """Compute SNR boundaries and step values for OPT-PCIRM.
    
    Eq. 9: n = -log2(10^(LC/10) / (10^(LC/10) + 1))
    Eq. 10: s_m(i,n) = ((m-1)/M)^n
    
    Args:
        local_criterion_db: LC in dB (default -15 dB for IBM).
        num_steps: M — total number of quantization steps.
    
    Returns:
        Tuple (step_values, exponent):
            step_values: Array of M attenuation values [s_1, s_2, ..., s_M].
            exponent: The computed exponent n.
    """
    lc_db = local_criterion_db if local_criterion_db is not None else config.LOCAL_CRITERION_DB
    M = num_steps or config.OPT_NUM_STEPS
    
    # Eq. 9: compute the exponent based on local criterion
    lc_linear = 10 ** (lc_db / 10.0)
    n_exp = -np.log2(lc_linear / (lc_linear + 1.0))
    
    # Eq. 10: compute attenuation values for each step
    step_values = np.zeros(M)
    for m in range(1, M + 1):
        step_values[m - 1] = ((m - 1) / M) ** n_exp
    
    return step_values, n_exp


def quantize_pcirm(pcirm, step_values, middle_value=None):
    """Quantize continuous PCIRM into discrete OPT-PCIRM values.
    
    Eq. 11: Assigns each T-F unit one of M attenuation values based on
    where the PCIRM value falls relative to the step boundaries.
    
    For M=3 steps:
      - s_1 (lowest, near 0) if 0 ≤ PCIRM ≤ s_2
      - x(i,n) (middle, PSO-optimized) if s_2 ≤ PCIRM ≤ s_3
      - s_M (highest, near 1) if s_3 ≤ PCIRM ≤ 1
    
    Args:
        pcirm: Continuous PCIRM values in [0, 1]. Shape: (C, F).
        step_values: Array of M attenuation step values.
        middle_value: Override value for the middle step (PSO-optimized).
                      If None, uses step_values[1].
    
    Returns:
        OPT-PCIRM values (discrete), same shape as input.
    """
    M = len(step_values)
    opt_pcirm = np.zeros_like(pcirm)
    
    if middle_value is not None and M >= 3:
        actual_values = step_values.copy()
        actual_values[1] = middle_value
    else:
        actual_values = step_values
    
    # Define boundaries between steps
    boundaries = np.zeros(M + 1)
    boundaries[0] = 0.0
    boundaries[-1] = 1.0
    for m in range(1, M):
        boundaries[m] = step_values[m]
    
    # Assign each T-F unit to a step
    for m in range(M):
        mask = (pcirm >= boundaries[m]) & (pcirm < boundaries[m + 1])
        opt_pcirm[mask] = actual_values[m]
    
    # Handle the upper boundary (PCIRM == 1)
    opt_pcirm[pcirm >= boundaries[-1]] = actual_values[-1]
    
    return opt_pcirm


def compute_opt_pcirm(pcirm, noisy_signal, clean_signal, fs=None,
                       num_steps=None, use_pso=True, pso_config=None):
    """Compute OPT-PCIRM with PSO-optimized middle step value.
    
    The PSO optimizes x(i,n) to maximize the STOI fitness function
    over the enhanced speech, making the step assignments adaptive
    rather than fixed.
    
    Args:
        pcirm: Continuous PCIRM values. Shape: (C, F).
        noisy_signal: Original noisy speech signal (1D time-domain).
        clean_signal: Clean speech signal (1D time-domain) for STOI fitness.
        fs: Sampling rate.
        num_steps: M — number of quantization steps.
        use_pso: Whether to use PSO optimization (True) or fixed steps (False).
        pso_config: Dict of PSO parameters overriding config defaults.
    
    Returns:
        Tuple (opt_pcirm_mask, step_values, optimized_middle):
            opt_pcirm_mask: Quantized mask. Same shape as pcirm.
            step_values: The M step values used.
            optimized_middle: The PSO-optimized middle step value.
    """
    fs = fs or config.SAMPLE_RATE
    
    # Step 1: Compute SNR boundaries and initial step values
    step_values, exponent = compute_snr_boundaries(num_steps=num_steps)
    
    if not use_pso:
        # Without PSO — use fixed step values
        opt_mask = quantize_pcirm(pcirm, step_values)
        return opt_mask, step_values, step_values[1] if len(step_values) > 1 else None
    
    # Step 2: PSO optimization of the middle step value
    pso_params = {
        'num_particles': config.PSO_NUM_PARTICLES,
        'max_iter': config.PSO_MAX_ITER,
        'w': config.PSO_W,
        'c1': config.PSO_C1,
        'c2': config.PSO_C2,
        'bounds': config.PSO_BOUNDS,
    }
    if pso_config:
        pso_params.update(pso_config)
    
    def fitness_function(x_middle):
        """PSO fitness: STOI of enhanced speech with candidate middle step.
        
        This is the direct optimization loop from the paper —
        STOI as fitness function for PSO.
        """
        from evaluation.stoi import compute_stoi
        
        # Quantize PCIRM with candidate middle value
        candidate_mask = quantize_pcirm(pcirm, step_values, middle_value=x_middle)
        
        # Apply mask to get enhanced speech (simplified: magnitude scaling)
        # For proper reconstruction, we'd use the GFTB synthesis
        # Here we use a simplified frame-based approach
        frame_size = config.FRAME_SIZE
        hop_size = config.HOP_SIZE
        
        num_channels, num_frames = candidate_mask.shape
        signal_len = len(noisy_signal)
        enhanced = np.zeros(signal_len)
        weights = np.zeros(signal_len)
        
        for n in range(num_frames):
            start = n * hop_size
            end = min(start + frame_size, signal_len)
            
            # Average mask across channels for this frame
            avg_mask = np.mean(candidate_mask[:, n])
            
            enhanced[start:end] += noisy_signal[start:end] * avg_mask
            weights[start:end] += 1
        
        weights = np.maximum(weights, 1)
        enhanced = enhanced / weights
        
        # Compute STOI as fitness (maximize)
        stoi_score = compute_stoi(clean_signal, enhanced, fs)
        return stoi_score
    
    # Run PSO
    pso = ParticleSwarmOptimizer(
        fitness_fn=fitness_function,
        num_particles=pso_params['num_particles'],
        max_iter=pso_params['max_iter'],
        w=pso_params['w'],
        c1=pso_params['c1'],
        c2=pso_params['c2'],
        bounds=pso_params['bounds'],
        maximize=True  # Maximize STOI
    )
    
    optimized_middle, best_fitness = pso.optimize()
    
    # Step 3: Create final OPT-PCIRM with optimized middle step
    opt_mask = quantize_pcirm(pcirm, step_values, middle_value=optimized_middle)
    
    return opt_mask, step_values, optimized_middle


def apply_opt_pcirm(noisy_tf, opt_pcirm):
    """Apply OPT-PCIRM to noisy T-F representation.
    
    Enhanced = OPT-PCIRM ⊙ Noisy
    
    Args:
        noisy_tf: Noisy speech in T-F domain.
        opt_pcirm: Optimal Phase Correlation Ideal Ratio Mask.
    
    Returns:
        Enhanced T-F representation.
    """
    return noisy_tf * opt_pcirm
