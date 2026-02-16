"""
Complex Conformer Mask Estimator.

    The Conformer (Gulati et al., 2020) combines local convolutional feature
    extraction with global self-attention, matching the multi-scale temporal 
    structure of speech. Extended to complex domain for joint magnitude-phase
    estimation.

Architecture:
    Input: complex STFT Z ∈ ℂ^{F×T} (F=256 bins, T=frames)
    → Complex linear projection
    → N=6 Conformer blocks (conv + multi-head self-attention)
    → Output: complex mask M̂ ∈ ℂ^{F×T}
    → Enhanced: Ŝ = M̂ ⊙ Z

    ~8M parameters (comparable to the original 5-layer DNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FeedForwardModule(nn.Module):
    """Conformer feed-forward module (half-step residual).
    
    Linear → Swish → Dropout → Linear → Dropout
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + 0.5 * x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention module for the Conformer.
    
    Captures global temporal dependencies across all time frames.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        return residual + x


class ConvolutionModule(nn.Module):
    """Conformer convolution module.
    
    Captures local temporal patterns via depthwise separable convolution.
    Maps to tonotopic organization of the cochlea (local frequency processing).
    
    Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv
    """
    
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise convolution (expand)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                    padding=padding, groups=d_model)
        
        self.batch_norm = nn.BatchNorm1d(d_model)
        
        # Pointwise convolution (project back)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, d_model)
        """
        residual = x
        x = self.layer_norm(x)
        
        # Transpose for Conv1d: (batch, d_model, time)
        x = x.transpose(1, 2)
        
        # Pointwise + GLU
        x = self.pointwise1(x)
        x = F.glu(x, dim=1)
        
        # Depthwise conv
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = x * torch.sigmoid(x)  # Swish
        
        # Pointwise
        x = self.pointwise2(x)
        x = self.dropout(x)
        
        # Transpose back: (batch, time, d_model)
        x = x.transpose(1, 2)
        
        return residual + x


class ConformerBlock(nn.Module):
    """Single Conformer block.
    
    FFN (half-step) → MHSA → Conv → FFN (half-step) → LayerNorm
    """
    
    def __init__(self, d_model, num_heads, d_ff, kernel_size, dropout):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.ff1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.final_norm(x)
        return x


class ComplexConformer(nn.Module):
    """Complex-domain Conformer for mask estimation.
    
    Operates on interleaved real+imaginary STFT to produce a complex mask.
    The complex mask enables joint magnitude-phase estimation, addressing
    the phase recovery problem that magnitude-only masks handle imperfectly.
    
    Architecture:
        Input: complex STFT → stack real+imag → [2*F, T]
        → Linear projection → [d_model, T]
        → N Conformer blocks
        → Linear projection → [2*F, T]
        → Unstack → complex mask [F, T]
    """
    
    def __init__(self, n_freq=None, d_model=None, num_blocks=None,
                 num_heads=None, d_ff=None, kernel_size=None, dropout=None):
        super().__init__()
        
        self.n_freq = n_freq or (config.FFT_SIZE // 2 + 1)
        self.d_model = d_model or config.CONFORMER_D_MODEL
        num_blocks = num_blocks or config.CONFORMER_NUM_BLOCKS
        num_heads = num_heads or config.CONFORMER_NUM_HEADS
        d_ff = d_ff or config.CONFORMER_FF_DIM
        kernel_size = kernel_size or config.CONFORMER_KERNEL_SIZE
        dropout = dropout or config.CONFORMER_DROPOUT
        
        # Input projection: real+imag stacked → d_model
        self.input_proj = nn.Linear(2 * self.n_freq, self.d_model)
        
        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(self.d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output projection: d_model → 2*F (real + imaginary mask)
        self.output_proj = nn.Linear(self.d_model, 2 * self.n_freq)
    
    def forward(self, stft_real, stft_imag):
        """Forward pass: complex STFT → complex mask.
        
        Args:
            stft_real: Real part of STFT. Shape: (batch, time, n_freq).
            stft_imag: Imaginary part of STFT. Shape: (batch, time, n_freq).
        
        Returns:
            Tuple (mask_real, mask_imag):
                mask_real: Real part of complex mask. Shape: (batch, time, n_freq).
                mask_imag: Imaginary part. Shape: (batch, time, n_freq).
        """
        # Stack real and imaginary: (batch, time, 2*n_freq)
        x = torch.cat([stft_real, stft_imag], dim=-1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Global skip for residual learning
        skip = x
        
        # Pass through Conformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Add global residual (network learns correction, not full mask)
        x = x + skip
        
        # Output projection
        x = self.output_proj(x)
        
        # Split into real and imaginary mask
        mask_real = x[..., :self.n_freq]
        mask_imag = x[..., self.n_freq:]
        
        return mask_real, mask_imag
    
    def apply_mask(self, stft_real, stft_imag, mask_real, mask_imag):
        """Apply complex mask: Ŝ = M̂ ⊙ Z (complex multiplication).
        
        Enhanced_real = mask_real * stft_real - mask_imag * stft_imag
        Enhanced_imag = mask_real * stft_imag + mask_imag * stft_real
        
        Args:
            stft_real, stft_imag: Noisy STFT components.
            mask_real, mask_imag: Complex mask components.
        
        Returns:
            Tuple (enhanced_real, enhanced_imag).
        """
        enhanced_real = mask_real * stft_real - mask_imag * stft_imag
        enhanced_imag = mask_real * stft_imag + mask_imag * stft_real
        return enhanced_real, enhanced_imag
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
