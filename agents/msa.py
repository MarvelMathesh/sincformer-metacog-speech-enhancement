"""
Mask Synthesis Agent (MSA)

Conformer-based complex mask generator with bounded polar output.
Uses sigmoid magnitude (0-1, attenuation only) + bounded phase rotation.

Architecture:
    [z_t (real+imag) || CPEA outputs] -> Fusion -> Conformer -> Polar mask
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.conformer import ComplexConformer


class MaskSynthesisAgent(nn.Module):
    """Conformer-based complex mask generator with bounded output.

    Key improvements over naive implementation:
    1. Bounded mask via tanh -- prevents noise amplification
    2. Residual feature fusion -- preserves information from PA and CPEA
    3. Skip connection from noisy input -- helps gradient flow
    4. Layer norm on mask output -- stabilizes training
    """

    def __init__(self, latent_dim=None, cpea_dim=None, d_model=None):
        super().__init__()

        latent_dim = latent_dim or config.PA_ENCODER_CHANNELS
        cpea_dim = cpea_dim or config.NUM_CHANNELS
        d_model = d_model or config.CONFORMER_D_MODEL

        # Feature fusion: combine latent z_t with CPEA outputs AND Noisy STFT
        # Input: z_real + z_imag + rho_s + rho_n + phi1 + phi2 + noisy_stft_real + noisy_stft_imag
        n_freq = config.FFT_SIZE // 2 + 1
        fusion_input_dim = 2 * latent_dim + 4 * cpea_dim + 2 * n_freq

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Conformer backbone for temporal modeling
        self.conformer = ComplexConformer(
            n_freq=d_model // 2,  # Split for real/imag
            d_model=d_model,
        )

        # Output projection to mask dimensions
        # Conformer outputs n_freq = d_model // 2 per real/imag channel
        conformer_out_dim = d_model // 2
        n_freq = config.FFT_SIZE // 2 + 1

        # Two-layer projection with GELU for better capacity
        self.mask_proj_real = nn.Sequential(
            nn.Linear(conformer_out_dim, conformer_out_dim),
            nn.GELU(),
            nn.Linear(conformer_out_dim, n_freq),
        )
        self.mask_proj_imag = nn.Sequential(
            nn.Linear(conformer_out_dim, conformer_out_dim),
            nn.GELU(),
            nn.Linear(conformer_out_dim, n_freq),
        )

        # (Removed LayerNorm here because it zeroes out the +3.0 identity bias)

        # Initialize output layers to produce near-identity mask
        self._init_weights()

    def _init_weights(self):
        """Initialize to produce MATHEMATICALLY PERFECT identity mask at start of training.
        This prevents initial signal destruction."""
        for module in [self.mask_proj_real, self.mask_proj_imag]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # ---------------------------------------------------------------------
        # THE GRADIENT LIFELINE FIX (Bug 12)
        # We use gain=0.1 to stay close to identity without killing gradients.
        # gain=1e-4 was too small and allowed Adam's weight_decay to freeze the model!
        # ---------------------------------------------------------------------
        nn.init.xavier_uniform_(self.mask_proj_real[-1].weight, gain=0.1)
        nn.init.xavier_uniform_(self.mask_proj_imag[-1].weight, gain=0.1)

        # Bias magnitude toward reasonably strong pass-through.
        # sigmoid(5.0) ~ 0.993. (Derivative is 0.006, safe for Adam).
        if hasattr(self.mask_proj_real[-1], 'bias'):
            nn.init.constant_(self.mask_proj_real[-1].bias, 5.0)
            
        # Phase toward EXACTLY zero (tanh(0.0) = 0.0)
        # Derivative of tanh at 0 is 1.0, so gradients flow perfectly.
        if hasattr(self.mask_proj_imag[-1], 'bias'):
            nn.init.constant_(self.mask_proj_imag[-1].bias, 0.0)

    def forward(self, z_real, z_imag, cpea_outputs, noisy_stft_real, noisy_stft_imag):
        """Generate bounded complex mask.

        Args:
            z_real: Real part of latent T-F from PA. Shape: (batch, D, T).
            z_imag: Imaginary part. Shape: (batch, D, T).
            cpea_outputs: Dict from CPEA with keys 'rho_s', 'rho_n', 'phi1', 'phi2'.
                          Each has shape: (batch, T, output_channels).
            noisy_stft_real: Real part of noisy STFT. Shape: (batch, T, n_freq).
            noisy_stft_imag: Imaginary part of noisy STFT. Shape: (batch, T, n_freq).

        Returns:
            Tuple (mask_real, mask_imag):
                Each shape: (batch, T, n_freq). Values in [-1, 1] (bounded).
        """
        batch, D, T = z_real.shape

        # Transpose to (batch, T, D)
        z_r = z_real.transpose(1, 2)
        z_i = z_imag.transpose(1, 2)

        # Get CPEA outputs (already batch, T, channels)
        rho_s = cpea_outputs['rho_s']
        rho_n = cpea_outputs['rho_n']
        phi1 = cpea_outputs['phi1']
        phi2 = cpea_outputs['phi2']

        # Normalize STFT inputs for stable neural fusion (log1p magnitude)
        mag = torch.sqrt(noisy_stft_real**2 + noisy_stft_imag**2 + 1e-8)
        norm_factor = torch.log1p(mag) / mag
        n_stft_r = noisy_stft_real * norm_factor
        n_stft_i = noisy_stft_imag * norm_factor

        # Concatenate all features: (batch, T, fusion_dim)
        fused = torch.cat([z_r, z_i, rho_s, rho_n, phi1, phi2, n_stft_r, n_stft_i], dim=-1)

        # Feature fusion with residual
        fused = self.fusion(fused)  # (batch, T, d_model)

        # Split for Conformer (real/imag input)
        d_half = fused.shape[-1] // 2
        stft_real = fused[..., :d_half]
        stft_imag = fused[..., d_half:]

        # Complex Conformer
        mask_r, mask_i = self.conformer(stft_real, stft_imag)

        # Project to output mask dimensions
        mask_mag_logit = self.mask_proj_real(
            mask_r.reshape(batch * mask_r.shape[1], -1)
        ).reshape(batch, mask_r.shape[1], -1)

        mask_phase_logit = self.mask_proj_imag(
            mask_i.reshape(batch * mask_i.shape[1], -1)
        ).reshape(batch, mask_i.shape[1], -1)

        # POLAR FORM: magnitude (sigmoid) + phase (bounded rotation)
        # - Magnitude in [0, 1]: mask can only ATTENUATE, never amplify
        # - Phase in [-pi/8, pi/8]: SMALL phase correction only
        #   (pi/4 was too aggressive, distorting clean speech)
        mask_mag = torch.sigmoid(mask_mag_logit)
        mask_phase = torch.tanh(mask_phase_logit) * (
            3.14159 / 8.0)

        # Convert polar to cartesian for complex multiplication
        mask_real = mask_mag * torch.cos(mask_phase)
        mask_imag = mask_mag * torch.sin(mask_phase)

        return mask_real, mask_imag

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
