"""
5-Layer DNN — Original paper's speech enhancement network.

Architecture from Section 3.1:
  - Input: Concatenated features (AMS + RASTA-PLP + MFCC + GFCC) with context
  - 3 hidden layers × 1024 ReLU units
  - Output: Sigmoid activation (mask ∈ [0,1])
  - Dropout: 0.2
  - Momentum: 0.5 for first 5 epochs → 0.9 afterwards
  - Pre-training: RBM layer-wise (see rbm.py)
  - Loss: MSE between predicted and oracle mask (Eq. 14)
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SpeechEnhancementDNN(nn.Module):
    """5-layer DNN for T-F mask estimation.
    
    Maps noisy speech features → mask values ∈ [0,1].
    
    The network learns the mapping from noisy features to the oracle mask
    (PCIRM or OPT-PCIRM) computed from clean speech during training.
    At inference, only the noisy features are needed.
    
    Architecture:
        Input (feature_dim) → [Linear → ReLU → Dropout] ×3 → Linear → Sigmoid
    """
    
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None,
                 num_hidden_layers=None, dropout=None):
        """
        Args:
            input_dim: Input feature dimension (with context).
            hidden_dim: Hidden layer dimension (default 1024).
            output_dim: Output dimension (number of T-F units to predict).
            num_hidden_layers: Number of hidden layers (default 3).
            dropout: Dropout rate (default 0.2).
        """
        super().__init__()
        
        self.input_dim = input_dim or 594  # Will be set from features
        self.hidden_dim = hidden_dim or config.DNN_HIDDEN_UNITS
        self.output_dim = output_dim or config.NUM_CHANNELS
        self.num_hidden_layers = num_hidden_layers or config.DNN_HIDDEN_LAYERS
        self.dropout_rate = dropout or config.DNN_DROPOUT
        
        # Build layers
        layers = []
        
        # Input layer → first hidden
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden layers
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer with sigmoid (mask values ∈ [0,1])
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass: features → mask prediction.
        
        Args:
            x: Input feature tensor of shape (batch_size, input_dim).
        
        Returns:
            Predicted mask values of shape (batch_size, output_dim).
        """
        return self.network(x)
    
    def get_layer_params(self):
        """Get parameters for each layer (for RBM pre-training).
        
        Returns:
            List of (weight, bias) tuples for each linear layer.
        """
        params = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                params.append((module.weight, module.bias))
        return params
    
    def load_rbm_weights(self, rbm_weights):
        """Load weights from RBM pre-training.
        
        Args:
            rbm_weights: List of (weight_matrix, visible_bias, hidden_bias)
                         tuples from RBM training, one per hidden layer.
        """
        linear_layers = [m for m in self.network if isinstance(m, nn.Linear)]
        
        for i, (w, vb, hb) in enumerate(rbm_weights):
            if i < len(linear_layers) - 1:  # Don't set output layer
                with torch.no_grad():
                    linear_layers[i].weight.copy_(torch.tensor(w.T))
                    linear_layers[i].bias.copy_(torch.tensor(hb))
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dnn(feature_dim, mask_dim=None):
    """Factory function to create a DNN with the paper's configuration.
    
    Args:
        feature_dim: Input feature dimension (from FeatureExtractor.feature_dim).
        mask_dim: Output mask dimension (default: NUM_CHANNELS=64).
    
    Returns:
        SpeechEnhancementDNN instance.
    """
    return SpeechEnhancementDNN(
        input_dim=feature_dim,
        hidden_dim=config.DNN_HIDDEN_UNITS,
        output_dim=mask_dim or config.NUM_CHANNELS,
        num_hidden_layers=config.DNN_HIDDEN_LAYERS,
        dropout=config.DNN_DROPOUT,
    )
