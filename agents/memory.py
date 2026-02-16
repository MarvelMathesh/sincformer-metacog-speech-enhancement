"""
Memory Module -- Episodic Key-Value Store for Domain Adaptation.

    An episodic memory bank (key-value store, keys = noise environment
    embedding, values = recent mask statistics) enables rapid domain
    adaptation. When the PA detects a new noise environment via embedding
    distance, MASES retrieves the closest memory and uses it to warm-start
    CPEA parameters, implementing continual learning without catastrophic
    forgetting.

Architecture:
    PA embeddings -> Memory lookup -> Bias for CPEA/MSA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EpisodicMemory(nn.Module):
    """Episodic key-value memory for noise environment adaptation.

    Stores learned noise environment embeddings as keys and corresponding
    mask statistics as values. During inference, retrieves the closest
    memory entry to the current noise environment and uses it to bias
    the enhancement pipeline.

    This implements a form of continual learning: the memory bank
    accumulates experience across noise environments without requiring
    retraining or risking catastrophic forgetting.

    Architecture:
        Key: noise environment embedding (from PA encoder pooled output)
        Value: mask bias vector (shifts MSA output toward known-good masks)
    """

    def __init__(self, key_dim=None, value_dim=None, num_slots=64,
                 temperature=1.0):
        """
        Args:
            key_dim: Dimension of environment embedding keys.
            value_dim: Dimension of mask bias values.
            num_slots: Maximum number of memory entries.
            temperature: Softmax temperature for retrieval.
        """
        super().__init__()

        self.key_dim = key_dim or config.PA_ENCODER_CHANNELS
        self.value_dim = value_dim or (config.FFT_SIZE // 2 + 1)
        self.num_slots = num_slots
        self.temperature = temperature

        # Learnable memory bank
        self.keys = nn.Parameter(
            torch.randn(num_slots, self.key_dim) * 0.01)
        self.values = nn.Parameter(
            torch.randn(num_slots, self.value_dim) * 0.01)

        # Key projection: PA encoder output -> memory key space
        self.key_proj = nn.Sequential(
            nn.Linear(self.key_dim, self.key_dim),
            nn.LayerNorm(self.key_dim),
            nn.GELU(),
            nn.Linear(self.key_dim, self.key_dim),
        )

        # Value projection: retrieved memory -> mask bias
        self.value_proj = nn.Sequential(
            nn.Linear(self.value_dim, self.value_dim),
            nn.Tanh(),  # Bounded bias
        )
        # ---------------------------------------------------------------------
        # THE GRADIENT LIFELINE FIX (Bug 12)
        # We CANNOT use exact zeros for weights or we kill the backward pass
        # for the entire memory agent. Use very small initial weights.
        # ---------------------------------------------------------------------
        nn.init.xavier_uniform_(self.value_proj[0].weight, gain=0.01)
        nn.init.zeros_(self.value_proj[0].bias)

        # Confidence gate: how much to trust the memory
        self.gate = nn.Sequential(
            nn.Linear(self.key_dim + self.value_dim, 1),
            nn.Sigmoid(),
        )

        # Track memory usage for analysis
        self.register_buffer('usage_count',
                             torch.zeros(num_slots))
        self.register_buffer('num_queries', torch.tensor(0))

    def forward(self, environment_embedding):
        """Retrieve mask bias from memory based on noise environment.

        Args:
            environment_embedding: Pooled PA encoder output.
                Shape: (batch, key_dim).

        Returns:
            Dict with:
                'bias': Mask bias from memory. (batch, value_dim).
                'gate': Confidence gate [0, 1]. (batch, 1).
                'top_indices': Indices of closest memory slots. (batch,).
                'similarity': Cosine similarity to closest slot. (batch,).
        """
        batch = environment_embedding.shape[0]

        # Project query into key space
        query = self.key_proj(environment_embedding)  # (batch, key_dim)

        # Normalize for cosine similarity
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(self.keys, dim=-1)

        # Compute similarity: (batch, num_slots)
        similarity = torch.matmul(
            query_norm, keys_norm.t()) / self.temperature

        # Soft attention over memory slots
        attention = F.softmax(similarity, dim=-1)  # (batch, num_slots)

        # Retrieve values: (batch, value_dim)
        retrieved = torch.matmul(attention, self.values)

        # Apply value projection (bounded bias)
        bias = self.value_proj(retrieved)

        # Confidence gate
        gate_input = torch.cat([query, retrieved], dim=-1)
        gate_value = self.gate(gate_input)

        # Track usage statistics
        if self.training:
            top_idx = similarity.argmax(dim=-1)
            with torch.no_grad():
                for idx in top_idx:
                    self.usage_count[idx] += 1
                self.num_queries += batch

        return {
            'bias': bias * gate_value,  # Gated bias
            'gate': gate_value,
            'top_indices': similarity.argmax(dim=-1),
            'similarity': similarity.max(dim=-1)[0],
        }

    def get_usage_stats(self):
        """Return memory slot utilization statistics."""
        total = self.num_queries.item()
        if total == 0:
            return torch.zeros(self.num_slots)
        return self.usage_count / total

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
