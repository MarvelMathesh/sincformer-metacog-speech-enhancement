"""
Correlation-Phase Estimation Agent (CPEA)

    A specialized sub-network (lightweight bidirectional LSTM, ≤2 layers) 
    that estimates the inter-channel correlation tensors ρ̂_s, ρ̂_n and phase 
    difference distributions p(φ₁|z_t), p(φ₂|z_t) *without* access to clean 
    speech — learned entirely from noisy observations via a contrastive 
    self-supervised objective.

This replaces the oracle-dependent correlation computation in Eq. 6-7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CorrelationPhaseEstimationAgent(nn.Module):
    """Estimates correlation coefficients and phase differences from noisy input.
    
    Key insight: learns to predict ρ̂_s, ρ̂_n from noisy observations only,
    eliminating the training-test covariate shift problem where oracle clean
    speech was needed to compute true correlation.
    
    Architecture:
        z_t → BiLSTM (2 layers) → [ρ̂_s head, ρ̂_n head, φ₁ head, φ₂ head]
    """
    
    def __init__(self, input_dim=None, hidden_size=None, num_layers=None,
                 output_channels=None):
        super().__init__()
        
        self.input_dim = input_dim or config.PA_ENCODER_CHANNELS
        self.hidden_size = hidden_size or config.CPEA_HIDDEN_SIZE
        self.num_layers = num_layers or config.CPEA_NUM_LAYERS
        self.output_channels = output_channels or config.NUM_CHANNELS
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if self.num_layers > 1 else 0.0,
        )
        
        lstm_output_dim = self.hidden_size * 2  # Bidirectional
        
        # Estimation heads
        # ρ̂_s: speech correlation coefficient ∈ [0, 1]
        self.rho_s_head = nn.Sequential(
            nn.Linear(lstm_output_dim, self.output_channels),
            nn.Sigmoid(),
        )
        
        # ρ̂_n: noise correlation coefficient ∈ [0, 1]
        self.rho_n_head = nn.Sequential(
            nn.Linear(lstm_output_dim, self.output_channels),
            nn.Sigmoid(),
        )
        
        # φ₁: phase difference between clean and noisy ∈ [-π, π]
        self.phi1_head = nn.Sequential(
            nn.Linear(lstm_output_dim, self.output_channels),
            nn.Tanh(),  # Output ∈ [-1, 1], scaled to [-π, π]
        )
        
        # φ₂: phase difference between noise and noisy ∈ [-π, π]
        self.phi2_head = nn.Sequential(
            nn.Linear(lstm_output_dim, self.output_channels),
            nn.Tanh(),
        )
    
    def forward(self, z_t):
        """Estimate correlation and phase from latent representation.
        
        Args:
            z_t: Latent T-F representation from Perception Agent.
                 Shape: (batch, D, T) or (batch, T, D).
        
        Returns:
            Dict with keys:
                'rho_s': (batch, T, output_channels) — estimated speech correlation
                'rho_n': (batch, T, output_channels) — estimated noise correlation
                'phi1': (batch, T, output_channels) — estimated phase diff (clean-noisy)
                'phi2': (batch, T, output_channels) — estimated phase diff (noise-noisy)
        """
        # Ensure shape is (batch, T, D) where D=input_dim
        if z_t.dim() == 3 and z_t.shape[-1] != self.input_dim:
            # Input is (batch, D, T) -> transpose to (batch, T, D)
            z_t = z_t.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.lstm(z_t)  # (batch, T, hidden*2)
        
        # Estimate each component
        rho_s = self.rho_s_head(lstm_out)
        rho_n = self.rho_n_head(lstm_out)
        phi1 = self.phi1_head(lstm_out) * torch.pi  # Scale to [-π, π]
        phi2 = self.phi2_head(lstm_out) * torch.pi
        
        return {
            'rho_s': rho_s,
            'rho_n': rho_n,
            'phi1': phi1,
            'phi2': phi2,
        }
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
