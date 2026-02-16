"""
Restricted Boltzmann Machine (RBM) — Layer-wise pre-training.

Implements the RBM pre-training (Hinton 2006) as described in Section 3.1:

    Pre-training with RBM is used to obtain robust initial weights and
    biases for each DNN layer. Without pre-training, randomly initialized
    weights may lead to singular values unsuitable for DNN training.

Uses Contrastive Divergence (CD-k) for efficient learning.
"""

import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RBM:
    """Restricted Boltzmann Machine for unsupervised pre-training.
    
    Each RBM is trained on one layer of the DNN to learn good initial
    weights via Contrastive Divergence.
    
    Usage:
        rbm = RBM(n_visible=594, n_hidden=1024)
        rbm.train(data, epochs=10)
        weights, v_bias, h_bias = rbm.get_weights()
    """
    
    def __init__(self, n_visible, n_hidden, learning_rate=None,
                 k_steps=None):
        """
        Args:
            n_visible: Number of visible units (input dimension).
            n_hidden: Number of hidden units.
            learning_rate: Learning rate for weight updates.
            k_steps: Number of Gibbs sampling steps in CD-k.
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate or config.RBM_LEARNING_RATE
        self.k = k_steps or config.RBM_K_STEPS
        
        # Initialize weights from small random Gaussian
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.v_bias = np.zeros(n_visible)
        self.h_bias = np.zeros(n_hidden)
    
    @staticmethod
    def sigmoid(x):
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def sample_hidden(self, v):
        """Sample hidden units given visible units.
        
        P(h=1|v) = σ(v·W + h_bias)
        
        Args:
            v: Visible unit activations. Shape: (batch, n_visible).
        
        Returns:
            Tuple (prob_h, sample_h) — probabilities and binary samples.
        """
        activation = v @ self.W + self.h_bias
        prob_h = self.sigmoid(activation)
        sample_h = (prob_h > np.random.random(prob_h.shape)).astype(float)
        return prob_h, sample_h
    
    def sample_visible(self, h):
        """Sample visible units given hidden units.
        
        P(v=1|h) = σ(h·W^T + v_bias)
        
        Args:
            h: Hidden unit activations. Shape: (batch, n_hidden).
        
        Returns:
            Tuple (prob_v, sample_v) — probabilities and binary samples.
        """
        activation = h @ self.W.T + self.v_bias
        prob_v = self.sigmoid(activation)
        sample_v = (prob_v > np.random.random(prob_v.shape)).astype(float)
        return prob_v, sample_v
    
    def contrastive_divergence(self, v_data):
        """Perform one step of CD-k learning.
        
        Positive phase: compute <v·h^T> from the data.
        Negative phase: run k Gibbs sampling steps, compute <v·h^T>
                        from the reconstruction.
        Update: ΔW = lr * (<v·h^T>_data - <v·h^T>_recon)
        
        Args:
            v_data: Batch of visible unit data. Shape: (batch, n_visible).
        
        Returns:
            Reconstruction error (MSE between data and reconstruction).
        """
        batch_size = v_data.shape[0]
        
        # Positive phase
        pos_h_prob, pos_h_sample = self.sample_hidden(v_data)
        pos_associations = v_data.T @ pos_h_prob
        
        # Negative phase (k steps of Gibbs sampling)
        h_sample = pos_h_sample
        for _ in range(self.k):
            neg_v_prob, neg_v_sample = self.sample_visible(h_sample)
            neg_h_prob, h_sample = self.sample_hidden(neg_v_prob)
        
        neg_associations = neg_v_prob.T @ neg_h_prob
        
        # Update weights and biases
        self.W += self.lr * (pos_associations - neg_associations) / batch_size
        self.v_bias += self.lr * np.mean(v_data - neg_v_prob, axis=0)
        self.h_bias += self.lr * np.mean(pos_h_prob - neg_h_prob, axis=0)
        
        # Reconstruction error
        error = np.mean((v_data - neg_v_prob) ** 2)
        return error
    
    def train(self, data, epochs=None, batch_size=None, verbose=True):
        """Train the RBM using Contrastive Divergence.
        
        Args:
            data: Training data. Shape: (num_samples, n_visible).
            epochs: Number of training epochs.
            batch_size: Batch size for mini-batch training.
            verbose: Print training progress.
        
        Returns:
            List of reconstruction errors per epoch.
        """
        epochs = epochs or config.RBM_EPOCHS
        batch_size = batch_size or config.RBM_BATCH_SIZE
        
        num_samples = data.shape[0]
        errors = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            epoch_error = 0
            num_batches = 0
            
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch = data[indices[start:end]]
                
                error = self.contrastive_divergence(batch)
                epoch_error += error
                num_batches += 1
            
            avg_error = epoch_error / num_batches
            errors.append(avg_error)
            
            if verbose:
                print(f"  RBM Epoch {epoch+1}/{epochs}: "
                      f"Reconstruction Error = {avg_error:.6f}")
        
        return errors
    
    def transform(self, data):
        """Transform data through the RBM (get hidden activations).
        
        Used to feed data to the next RBM layer in stacked pre-training.
        
        Args:
            data: Input data. Shape: (num_samples, n_visible).
        
        Returns:
            Hidden unit probabilities. Shape: (num_samples, n_hidden).
        """
        prob_h, _ = self.sample_hidden(data)
        return prob_h
    
    def get_weights(self):
        """Get the trained weights and biases.
        
        Returns:
            Tuple (W, v_bias, h_bias).
        """
        return self.W.copy(), self.v_bias.copy(), self.h_bias.copy()


def pretrain_dnn_with_rbm(data, layer_sizes, verbose=True):
    """Perform layer-wise RBM pre-training for the DNN.
    
    Trains a stack of RBMs, one per hidden layer, where each RBM's
    hidden activations become the input to the next RBM.
    
    Args:
        data: Training features. Shape: (num_samples, input_dim).
        layer_sizes: List of layer dimensions [input, hidden1, hidden2, ...].
        verbose: Print progress.
    
    Returns:
        List of (W, v_bias, h_bias) tuples, one per layer.
    """
    rbm_weights = []
    current_data = data.copy()
    
    for i in range(len(layer_sizes) - 1):
        n_visible = layer_sizes[i]
        n_hidden = layer_sizes[i + 1]
        
        if verbose:
            print(f"\n--- RBM Layer {i+1}: {n_visible} → {n_hidden} ---")
        
        rbm = RBM(n_visible, n_hidden)
        rbm.train(current_data, verbose=verbose)
        
        # Store weights
        rbm_weights.append(rbm.get_weights())
        
        # Transform data for next layer
        current_data = rbm.transform(current_data)
    
    return rbm_weights
