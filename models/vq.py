"""
Vector Quantization with Straight-Through Estimator (STE).

    Replace PSO with a differentiable quantization layer using VQ with
    commitment loss. After the soft mask M̂ is produced, a learned quantizer Q
    maps continuous values to one of M=3 learnable centroid values {c₁, c₂, c₃}.
    
    Forward: hard assignment to nearest centroid.
    Backward: straight-through gradient (STE).
    Commitment loss: L_commit = ||sg(M̂) - c_k||²

This reduces training time from O(PSO_iters × TF_units × utterances) to a 
single forward-backward pass, enabling proper end-to-end training.

Neuroscience connection: VQ implements categorical perception — the brain's
tendency to perceive graded acoustic signals as discrete phonemic categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class VectorQuantizer(nn.Module):
    """Differentiable Vector Quantization layer.
    
    Maps continuous mask values to discrete centroids via nearest-neighbor
    lookup with straight-through gradient estimation.
    
    Usage:
        vq = VectorQuantizer(num_centroids=3)
        quantized, indices, commitment_loss = vq(soft_mask)
    """
    
    def __init__(self, num_centroids=None, commitment_weight=None):
        """
        Args:
            num_centroids: M — number of learnable centroid values.
            commitment_weight: β — weight for commitment loss.
        """
        super().__init__()
        
        self.num_centroids = num_centroids or config.VQ_NUM_CENTROIDS
        self.beta = commitment_weight or config.VQ_COMMITMENT_WEIGHT
        
        # Learnable centroids initialized evenly in [0, 1]
        initial_centroids = torch.linspace(0, 1, self.num_centroids)
        self.centroids = nn.Parameter(initial_centroids)
    
    def forward(self, x):
        """Quantize input to nearest centroid.
        
        Forward pass: hard assignment (non-differentiable).
        Backward pass: straight-through estimator (gradient passes through).
        
        Args:
            x: Continuous input values. Shape: (*, ) — any shape.
        
        Returns:
            Tuple (quantized, indices, commitment_loss):
                quantized: Quantized values, same shape as input.
                indices: Centroid indices, same shape as input.
                commitment_loss: Scalar loss for codebook utilization.
        """
        # Compute distances to each centroid
        # x shape: (*,), centroids shape: (M,)
        x_flat = x.reshape(-1, 1)  # (N, 1)
        centroids = self.centroids.reshape(1, -1)  # (1, M)
        
        distances = (x_flat - centroids) ** 2  # (N, M)
        
        # Find nearest centroid
        indices = torch.argmin(distances, dim=-1)  # (N,)
        
        # Quantize
        quantized_flat = self.centroids[indices]
        quantized = quantized_flat.reshape(x.shape)
        indices = indices.reshape(x.shape)
        
        # Codebook loss: ||sg(x) - c_k||² 
        # Moves codebook centroids toward the encoder output
        codebook_loss = F.mse_loss(x.detach(), quantized)
        
        # Commitment loss: ||sg(c_k) - x||²
        # Encourages encoder output to commit to the codebook
        commitment_loss = self.beta * F.mse_loss(x, quantized.detach())
        
        # Straight-Through Estimator: quantized has gradient of x
        quantized = x + (quantized - x).detach()
        
        total_vq_loss = commitment_loss + codebook_loss
        
        return quantized, indices, total_vq_loss
    
    def get_centroids(self):
        """Return current centroid values.
        
        Returns:
            1D tensor of centroid values, sorted ascending.
        """
        sorted_centroids, _ = torch.sort(self.centroids)
        return sorted_centroids
    
    @torch.no_grad()
    def get_utilization(self, indices):
        """Compute centroid utilization (what fraction of inputs use each centroid).
        
        Args:
            indices: Assignment indices from forward pass.
        
        Returns:
            1D tensor of utilization fractions per centroid.
        """
        total = indices.numel()
        utilization = torch.zeros(self.num_centroids)
        for i in range(self.num_centroids):
            utilization[i] = (indices == i).sum().float() / total
        return utilization


class VQMaskQuantizer(nn.Module):
    """End-to-end mask quantization module.
    
    Wraps a mask estimator (DNN or Conformer) with a VQ layer to produce
    quantized mask outputs suitable for intelligibility optimization.
    
    mask_estimator → soft mask → VQ → quantized mask
    """
    
    def __init__(self, mask_estimator, num_centroids=None):
        """
        Args:
            mask_estimator: nn.Module that outputs soft mask values in [0, 1].
            num_centroids: Number of VQ centroids.
        """
        super().__init__()
        self.mask_estimator = mask_estimator
        self.vq = VectorQuantizer(num_centroids=num_centroids)
    
    def forward(self, x, return_soft=False):
        """Forward pass: features → soft mask → quantized mask.
        
        Args:
            x: Input features for the mask estimator.
            return_soft: If True, also return the soft mask.
        
        Returns:
            If return_soft:
                Tuple (quantized_mask, soft_mask, vq_loss).
            Else:
                Tuple (quantized_mask, vq_loss).
        """
        soft_mask = self.mask_estimator(x)
        quantized_mask, indices, vq_loss = self.vq(soft_mask)
        
        if return_soft:
            return quantized_mask, soft_mask, vq_loss
        return quantized_mask, vq_loss
