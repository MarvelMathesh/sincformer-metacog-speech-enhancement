"""
Metacognitive Arbitration Agent (MAA)

    A lightweight meta-controller that determines

    Monitors the PA's uncertainty signal σ_t. When σ_t exceeds a learned 
    threshold τ (itself optimized via meta-learning), MAA triggers one of 
    three responses:
      1. Resampling: Re-run MSA with stochastic dropout for ensemble averaging
      2. Step quantization: Fall back to hard-mask regime (OPT-PCIRM)
      3. Human escalation: Flag the utterance for human review

This agent explicitly addresses the quality-intelligibility tradeoff 
by dynamically switching between soft and hard mask regimes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MetacognitiveArbitrationAgent(nn.Module):
    """Uncertainty-aware decision module for mask strategy selection.
    
    Monitors the Perception Agent's uncertainty signal and decides
    the optimal enhancement strategy for each time window.
    
    Decisions:
        0 (SOFT_MASK):  Use the soft mask directly (high confidence)
        1 (RESAMPLE):   Ensemble average multiple stochastic forward passes
        2 (HARD_MASK):  Fall back to quantized (OPT-PCIRM-style) mask
        3 (ESCALATE):   Flag for human review (hearing-aid calibration)
    """
    
    # Decision constants
    SOFT_MASK = 0
    RESAMPLE = 1
    HARD_MASK = 2
    ESCALATE = 3
    
    def __init__(self, input_dim=1, hidden_dim=64, num_classes=4,
                 initial_threshold=None):
        super().__init__()
        
        self.threshold_init = initial_threshold or config.MAA_THRESHOLD_INIT
        
        # Learned threshold τ (meta-learned)
        self.threshold = nn.Parameter(
            torch.tensor([self.threshold_init])
        )
        
        # Decision network: uncertainty signal → action
        self.decision_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Running statistics for adaptive thresholding
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.register_buffer('num_updates', torch.tensor(0))
    
    def forward(self, sigma):
        """Make enhancement strategy decisions based on uncertainty.
        
        Args:
            sigma: Uncertainty signal from Perception Agent.
                   Shape: (batch, 1, T) or (batch, T).
        
        Returns:
            Dict with keys:
                'decisions': Tensor of decision indices. Shape: (batch, T).
                'logits': Raw decision logits. Shape: (batch, T, 4).
                'threshold': Current learned threshold value.
                'confidence': Confidence scores per time step.
        """
        if sigma.dim() == 3:
            sigma = sigma.squeeze(1)  # (batch, T)
        
        batch, T = sigma.shape
        
        # Normalize uncertainty relative to running statistics
        if self.training:
            self._update_statistics(sigma)
        
        normalized_sigma = (sigma - self.running_mean) / (
            torch.sqrt(self.running_var) + 1e-8
        )
        
        # Decision network
        sigma_input = normalized_sigma.unsqueeze(-1)  # (batch, T, 1)
        logits = self.decision_net(sigma_input)  # (batch, T, 4)
        
        # Hard decisions (argmax during inference, soft probs during training)
        if self.training:
            # Soft probabilities — gradient flows through softmax
            probs = F.softmax(logits, dim=-1)
            decisions = probs.argmax(dim=-1)  # For logging only
        else:
            probs = F.softmax(logits, dim=-1)
            decisions = logits.argmax(dim=-1)
        
        # Confidence: inverse of uncertainty
        confidence = torch.sigmoid(-normalized_sigma)
        
        return {
            'decisions': decisions,
            'probs': probs,
            'logits': logits,
            'threshold': self.threshold,
            'confidence': confidence,
        }
    
    def _update_statistics(self, sigma):
        """Update running mean and variance with exponential moving average."""
        with torch.no_grad():
            batch_mean = sigma.mean()
            batch_var = sigma.var()
            
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
            self.num_updates += 1
    
    def get_strategy_name(self, decision_idx):
        """Get human-readable strategy name."""
        names = {
            0: "SOFT_MASK (high confidence)",
            1: "RESAMPLE (ensemble averaging)",
            2: "HARD_MASK (quantized fallback)",
            3: "ESCALATE (human review)",
        }
        return names.get(decision_idx, "UNKNOWN")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
