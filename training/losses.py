"""
Research-Grade Loss Functions for Speech Enhancement.

Implements four loss functions:

1. MSE Mask Loss (Eq. 14 from paper)
2. Perceptual STOI Loss -- proper 1/3-octave band approximation
3. Multi-Scale Adversarial Loss -- HiFi-GAN style multi-scale disc
4. Multi-Resolution STFT Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MSEMaskLoss(nn.Module):
    """MSE loss between predicted and oracle mask (Eq. 14)."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted_mask, oracle_mask):
        return self.mse(predicted_mask, oracle_mask)


# ============================================================================
# Gap 3 Fix: Research-Grade Perceptual STOI Loss
# ============================================================================

class PerceptualSTOILoss(nn.Module):
    """Differentiable STOI approximation using 1/3-octave band envelopes.

    Follows the STOI algorithm (Taal et al., 2011):
    1. Compute 1/3-octave band envelopes (15 bands, 150-4000 Hz for NB)
    2. Segment into 384ms frames (30 frames at 10ms hop)
    3. Normalize and clip enhanced envelope
    4. Compute per-band, per-frame correlation
    5. Average across bands and frames

    This is a much closer approximation than simple envelope correlation.
    """

    def __init__(self, sample_rate=None):
        super().__init__()
        sr = sample_rate or config.SAMPLE_RATE

        # 1/3-octave band center frequencies (ANSI S1.6-1984)
        # For 8kHz narrowband: 150 Hz to ~3400 Hz
        center_freqs = [
            150, 200, 250, 315, 400, 500, 630, 800,
            1000, 1250, 1600, 2000, 2500, 3150
        ]
        self.num_bands = len(center_freqs)

        # STOI frame parameters
        self.frame_len = 30  # 30 STFT frames per STOI segment (~384ms)
        self.beta = 15.0     # Clipping parameter (dB)

        # Create 1/3-octave band weights matrix
        # Maps from STFT bins to 1/3-octave bands
        n_fft = config.FFT_SIZE
        n_freq = n_fft // 2 + 1
        freqs = np.linspace(0, sr / 2, n_freq)

        band_weights = np.zeros((self.num_bands, n_freq), dtype=np.float32)
        for i, cf in enumerate(center_freqs):
            # 1/3-octave bandwidth
            f_low = cf / (2 ** (1/6))
            f_high = cf * (2 ** (1/6))
            for j, f in enumerate(freqs):
                if f_low <= f <= f_high:
                    band_weights[i, j] = 1.0

        # Normalize each band
        band_sums = band_weights.sum(axis=1, keepdims=True)
        band_sums[band_sums == 0] = 1.0
        band_weights = band_weights / band_sums

        self.register_buffer('band_weights',
                             torch.from_numpy(band_weights))

    def forward(self, enhanced_spec, clean_spec):
        """Compute differentiable STOI loss.

        Args:
            enhanced_spec: Enhanced magnitude spectrogram. (batch, freq, time).
            clean_spec: Clean magnitude spectrogram. (batch, freq, time).

        Returns:
            Scalar loss (negative STOI -- minimize to maximize STOI).
        """
        # Step 1: Compute 1/3-octave band envelopes
        # band_weights: (num_bands, freq), spec: (batch, freq, time)
        # Result: (batch, num_bands, time)
        clean_env = torch.matmul(
            self.band_weights.unsqueeze(0), clean_spec)
        enh_env = torch.matmul(
            self.band_weights.unsqueeze(0), enhanced_spec)

        # Step 2: Segment into 384ms frames
        T = clean_env.shape[-1]
        num_segments = max(1, T // self.frame_len)
        T_use = num_segments * self.frame_len

        clean_seg = clean_env[..., :T_use].reshape(
            clean_env.shape[0], self.num_bands, num_segments, self.frame_len)
        enh_seg = enh_env[..., :T_use].reshape(
            enh_env.shape[0], self.num_bands, num_segments, self.frame_len)

        # Step 3: Normalize and clip
        # Remove mean per segment
        clean_seg = clean_seg - clean_seg.mean(dim=-1, keepdim=True)
        enh_seg = enh_seg - enh_seg.mean(dim=-1, keepdim=True)

        # Clip enhanced envelope: ||enh||/||clean|| <= 10^(beta/20)
        clean_energy = torch.sqrt(
            torch.sum(clean_seg ** 2, dim=-1, keepdim=True) + 1e-8)
        enh_energy = torch.sqrt(
            torch.sum(enh_seg ** 2, dim=-1, keepdim=True) + 1e-8)

        clip_factor = 10 ** (self.beta / 20.0)
        max_ratio = clip_factor * clean_energy / (enh_energy + 1e-8)
        scale = torch.min(torch.ones_like(max_ratio), max_ratio)
        enh_seg_clipped = enh_seg * scale

        # Step 4: Per-band, per-segment correlation
        numer = torch.sum(clean_seg * enh_seg_clipped, dim=-1)
        denom = (torch.sqrt(torch.sum(clean_seg ** 2, dim=-1) + 1e-8) *
                 torch.sqrt(torch.sum(enh_seg_clipped ** 2, dim=-1) + 1e-8))

        correlation = numer / (denom + 1e-8)

        # Step 5: Average across bands and segments
        stoi_approx = correlation.mean()

        return -stoi_approx


# ============================================================================
# Gap 4 Fix: Multi-Scale Discriminator (HiFi-GAN style)
# ============================================================================

class SubDiscriminator(nn.Module):
    """Single-scale discriminator with spectral normalization."""

    def __init__(self, n_freq, channels=None):
        super().__init__()
        channels = channels or [64, 128, 256, 512]

        layers = []
        in_ch = n_freq
        for i, out_ch in enumerate(channels):
            stride = 2 if i < len(channels) - 1 else 1
            layers.extend([
                nn.utils.spectral_norm(
                    nn.Conv1d(in_ch, out_ch, 5, stride=stride, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch

        layers.append(
            nn.utils.spectral_norm(nn.Conv1d(in_ch, 1, 3, padding=1)))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch, freq, time) -> scalar."""
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                features.append(x)
        return x, features


class AdversarialLoss(nn.Module):
    """Multi-scale adversarial loss (HiFi-GAN style).

    Uses 3 discriminators operating at different temporal scales:
    1. Original resolution
    2. 2x downsampled
    3. 4x downsampled

    Features:
    - Spectral normalization for training stability
    - Feature matching loss for richer gradient signal
    - Multi-scale analysis captures both local and global structure
    """

    def __init__(self, input_dim=None):
        super().__init__()

        n_freq = input_dim or (config.FFT_SIZE // 2 + 1)

        # 3 discriminators at different scales
        self.discriminators = nn.ModuleList([
            SubDiscriminator(n_freq, [64, 128, 256, 512]),
            SubDiscriminator(n_freq, [64, 128, 256]),
            SubDiscriminator(n_freq, [32, 64, 128]),
        ])

        # Downsampling for multi-scale
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        # Combined discriminator for backward compat
        self.discriminator = self.discriminators[0]

    def discriminator_loss(self, clean_spec, enhanced_spec):
        """Multi-scale discriminator loss.

        Args:
            clean_spec: (batch, freq, time)
            enhanced_spec: (batch, freq, time)

        Returns:
            Scalar discriminator loss.
        """
        total_loss = torch.tensor(0.0, device=clean_spec.device)
        real_x = clean_spec
        fake_x = enhanced_spec.detach()

        for i, disc in enumerate(self.discriminators):
            real_out, _ = disc(real_x)
            fake_out, _ = disc(fake_x)

            real_loss = F.mse_loss(real_out, torch.ones_like(real_out))
            fake_loss = F.mse_loss(fake_out, torch.zeros_like(fake_out))

            total_loss = total_loss + (real_loss + fake_loss)

            # Downsample for next scale
            if i < len(self.discriminators) - 1:
                real_x = self.downsample(real_x)
                fake_x = self.downsample(fake_x)

        return total_loss / len(self.discriminators)

    def generator_loss(self, enhanced_spec):
        """Multi-scale generator loss with feature matching.

        Args:
            enhanced_spec: (batch, freq, time)

        Returns:
            Scalar generator loss.
        """
        total_loss = torch.tensor(0.0, device=enhanced_spec.device)
        x = enhanced_spec

        for i, disc in enumerate(self.discriminators):
            fake_out, _ = disc(x)

            # Adversarial loss: fool disc into thinking enhanced is real
            total_loss = total_loss + F.mse_loss(
                fake_out, torch.ones_like(fake_out))

            if i < len(self.discriminators) - 1:
                x = self.downsample(x)

        return total_loss / len(self.discriminators)

    def feature_matching_loss(self, clean_spec, enhanced_spec):
        """Feature matching loss: match intermediate disc features.

        Provides richer gradient signal than adversarial loss alone.
        """
        total_loss = torch.tensor(0.0, device=clean_spec.device)
        real_x = clean_spec
        fake_x = enhanced_spec

        for i, disc in enumerate(self.discriminators):
            _, real_feats = disc(real_x)
            _, fake_feats = disc(fake_x)

            for rf, ff in zip(real_feats, fake_feats):
                total_loss = total_loss + F.l1_loss(ff, rf.detach())

            if i < len(self.discriminators) - 1:
                real_x = self.downsample(real_x)
                fake_x = self.downsample(fake_x)

        return total_loss / len(self.discriminators)
