"""
Perception Agent (PA) -- Research-Grade Learned Front-End.

Replaces the fixed Gammatone Filterbank with a learned encoder.
Key upgrade: SincNet-style first layer with auditory-inspired initialization
(bandpass filters initialized from ERB-spaced center frequencies).

Architecture:
    Raw waveform -> SincConv1D (auditory init) -> Conv stack -> z_t + sigma
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SincConv1d(nn.Module):
    """SincNet-style parameterized sinc filter layer.

    Instead of learning arbitrary convolution kernels, this layer
    learns only the low and high cutoff frequencies of bandpass filters.
    Initialized from ERB-spaced center frequencies matching the
    gammatone filterbank, providing auditory inductive bias.

    Reference: Ravanelli & Bengio, "Speaker Recognition from Raw Waveform
    with SincNet", IEEE SLT 2018.
    """

    def __init__(self, out_channels, kernel_size, sample_rate=None,
                 min_low_hz=50, min_band_hz=50):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate or config.SAMPLE_RATE

        # Enforce odd kernel size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1

        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize from ERB-spaced center frequencies
        # matching the gammatone filterbank (auditory prior)
        low_hz = min_low_hz
        high_hz = self.sample_rate / 2 - min_band_hz

        # ERB-spaced center frequencies
        erb_low = 21.4 * math.log10(1 + low_hz / 228.7)
        erb_high = 21.4 * math.log10(1 + high_hz / 228.7)
        erb_points = np.linspace(erb_low, erb_high, out_channels + 1)
        hz_points = 228.7 * (10 ** (erb_points / 21.4) - 1)

        # Learnable parameters: low and band frequencies
        self.low_hz_ = nn.Parameter(
            torch.Tensor(hz_points[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(
            torch.Tensor(np.diff(hz_points)).view(-1, 1))

        # Hamming window (fixed, not learned)
        n = torch.linspace(0, self.kernel_size - 1, self.kernel_size)
        self.register_buffer(
            'window',
            0.54 - 0.46 * torch.cos(2 * math.pi * n / self.kernel_size))

        # Time steps for sinc computation
        n = (self.kernel_size - 1) / 2.0
        self.register_buffer(
            'n_',
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate)

    def forward(self, waveform):
        """Apply learned bandpass filters.

        Args:
            waveform: (batch, 1, samples)

        Returns:
            (batch, out_channels, T)
        """
        # Clamp frequencies to valid range
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            max=self.sample_rate / 2.0)

        # Compute bandpass filter kernels using sinc functions
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate

        # Band-pass = high-pass - low-pass
        band_pass_left = (
            (torch.sin(f_high * self.n_) - torch.sin(f_low * self.n_))
            / (self.n_ / 2.0 + 1e-8))

        # Symmetric filter
        band_pass_center = 2 * (f_high - f_low)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1)

        # Apply Hamming window
        band_pass = band_pass * self.window
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)

        # Reshape for conv1d: (out_channels, 1, kernel_size)
        filters = band_pass.unsqueeze(1)

        return F.conv1d(waveform, filters, stride=1,
                        padding=self.kernel_size // 2)


class _ResidualBlock(nn.Module):
    """Residual block: main(x) + skip(x), then GELU."""
    def __init__(self, main, skip):
        super().__init__()
        self.main = main
        self.skip = skip

    def forward(self, x):
        return F.gelu(self.main(x) + self.skip(x))


class PerceptionAgent(nn.Module):
    """SincNet-based learned front-end with uncertainty estimation.

    Architecture:
        Raw waveform
        -> SincConv1d (auditory ERB-init bandpass filters)
        -> Conv1D stack (progressive downsampling)
        -> Residual blocks with GroupNorm
        -> z_real, z_imag (complex latent)
        -> sigma (uncertainty)
    """

    def __init__(self, encoder_channels=None, sample_rate=None):
        super().__init__()

        self.encoder_channels = encoder_channels or config.PA_ENCODER_CHANNELS
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        D = self.encoder_channels  # shorthand

        # Layer 1: SincNet filterbank (auditory-inspired)
        self.sinc_conv = SincConv1d(
            out_channels=D // 4,  # 64 filters
            kernel_size=251,  # ~31ms at 8kHz -- covers one pitch period
            sample_rate=self.sample_rate,
        )
        self.sinc_norm = nn.GroupNorm(8, D // 4)

        # Layer 2-5: Progressive Conv1D stack with residual connections
        channels = [D // 4, D // 2, D // 2, D]
        self.conv_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.conv_blocks.append(self._make_block(
                channels[i], channels[i + 1], stride=2))

        # Final downsampling to match STFT frame rate
        self.downsample = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, D),
            nn.GELU(),
        )

        # Complex-valued projection (z_real + z_imag)
        self.real_proj = nn.Sequential(
            nn.Conv1d(D, D, 1),
            nn.GroupNorm(16, D),
        )
        self.imag_proj = nn.Sequential(
            nn.Conv1d(D, D, 1),
            nn.GroupNorm(16, D),
        )

        # Uncertainty head (Bayesian)
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(D, D // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(D // 4, 1, 1),
        )

        self._init_weights()

    def _make_block(self, in_ch, out_ch, stride):
        """Residual conv block with GroupNorm, GELU, and TRUE skip connection."""
        main = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3),
            nn.GroupNorm(min(16, out_ch), out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(16, out_ch), out_ch),
        )
        # 1x1 projection for skip connection when dimensions change
        skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride),
            nn.GroupNorm(min(16, out_ch), out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        return _ResidualBlock(main, skip)

    def _init_weights(self):
        """Kaiming init for conv layers, xavier for projections."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) and not isinstance(m, SincConv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, waveform):
        """Process raw waveform through SincNet encoder.

        Args:
            waveform: (batch, samples) or (batch, 1, samples).

        Returns:
            Tuple (z_real, z_imag, sigma):
                z_real: (batch, D, T)
                z_imag: (batch, D, T)
                sigma:  (batch, 1, T)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (batch, 1, samples)

        # SincNet filterbank
        x = self.sinc_conv(waveform)  # (batch, D//4, samples)
        x = self.sinc_norm(x)
        x = F.gelu(x)

        # Conv stack with progressive downsampling + RESIDUAL CONNECTIONS
        for block in self.conv_blocks:
            x = block(x)  # Each block now has a true residual skip

        # Final downsample
        x = self.downsample(x)  # (batch, D, T)

        # Complex-valued output
        z_real = self.real_proj(x)
        z_imag = self.imag_proj(x)

        # Uncertainty
        log_var = self.uncertainty_head(x)
        sigma = torch.exp(0.5 * torch.clamp(log_var, -10, 10))

        return z_real, z_imag, sigma

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
