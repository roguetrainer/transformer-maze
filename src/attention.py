"""
Attention Mechanisms - The Core of Transformers

This module implements attention from scratch to demonstrate the fundamental
mechanism that enables transformers to "see" the entire sequence at once.

Key concepts:
- Scaled dot-product attention
- Multi-head attention
- Positional encoding
- Attention masking
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    The fundamental attention mechanism.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    This is the "teleportation" mechanism - it allows each position to
    directly access information from any other position in the sequence.
    """
    
    def __init__(self, temperature: float = None, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor [batch_size, n_heads, seq_len, d_k]
            k: Key tensor [batch_size, n_heads, seq_len, d_k]
            v: Value tensor [batch_size, n_heads, seq_len, d_v]
            mask: Optional mask tensor [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: Attention output [batch_size, n_heads, seq_len, d_v]
            attention: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = q.size(-1)
        
        # Use provided temperature or default to sqrt(d_k)
        temperature = self.temperature if self.temperature else math.sqrt(d_k)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # This computes similarity between all query-key pairs
        scores = torch.matmul(q, k.transpose(-2, -1)) / temperature
        
        # Apply mask (if provided)
        # Mask is used for:
        # 1. Padding: ignore padded positions
        # 2. Causality: prevent looking at future tokens (for autoregressive models)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        # Each row sums to 1 - these are the "teleportation probabilities"
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention weights to values
        # This is the weighted sum that "pulls" information from relevant positions
        output = torch.matmul(attention, v)
        
        return output, attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: Multiple attention mechanisms in parallel.
    
    Each "head" learns to attend to different aspects:
    - Head 1 might focus on syntax
    - Head 2 might focus on semantics
    - Head 3 might focus on long-range dependencies
    
    This is like having multiple "search strategies" operating simultaneously.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        # These learn what to "query", what to use as "keys", and what "values" to return
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        # Output projection
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query [batch_size, seq_len, d_model]
            k: Key [batch_size, seq_len, d_model]
            v: Value [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = q.size()
        residual = q
        
        # Linear projections and split into multiple heads
        # Shape: [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        q = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for multiple heads
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
        
        # Apply attention
        output, attention = self.attention(q, k, v, mask)
        
        # Concatenate heads and apply final linear projection
        # Shape: [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.dropout(self.fc(output))
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attention


class PositionalEncoding(nn.Module):
    """
    Positional Encoding: Add position information to embeddings.
    
    Without this, attention is permutation-invariant - it can't tell the
    difference between "dog bites man" and "man bites dog".
    
    Uses sinusoidal functions of different frequencies:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the positional encodings using sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Modern alternative to sinusoidal.
    
    Used in models like GPT-Neo, PaLM, LLaMA.
    Applies rotation matrices to embed relative position information.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for rotations
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached rotation matrices if sequence length changes"""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            q: Queries [batch_size, n_heads, seq_len, d_k]
            k: Keys [batch_size, n_heads, seq_len, d_k]
            
        Returns:
            Rotated queries and keys
        """
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device)
        
        # Apply rotation
        q_rot = (q * self._cos_cached) + (self._rotate_half(q) * self._sin_cached)
        k_rot = (k * self._cos_cached) + (self._rotate_half(k) * self._sin_cached)
        
        return q_rot, k_rot
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input"""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal mask for autoregressive generation.
    
    This prevents positions from attending to future positions.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        mask: [seq_len, seq_len] lower triangular mask
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create padding mask to ignore padded positions.
    
    Args:
        seq: Sequence tensor [batch_size, seq_len]
        pad_idx: Padding token index
        
    Returns:
        mask: [batch_size, seq_len] mask (1 for real tokens, 0 for padding)
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


class SimpleAttentionDemo:
    """
    Simplified attention for educational demonstrations.
    Uses NumPy for clarity (no autograd complexity).
    """
    
    @staticmethod
    def scaled_dot_product(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure NumPy implementation of attention.
        
        Args:
            Q: Queries [seq_len, d_k]
            K: Keys [seq_len, d_k]
            V: Values [seq_len, d_v]
            mask: Optional mask [seq_len, seq_len]
            
        Returns:
            output: Attention output [seq_len, d_v]
            attention: Attention weights [seq_len, seq_len]
        """
        d_k = Q.shape[-1]
        
        # Compute attention scores
        scores = np.matmul(Q, K.T) / np.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores + (1 - mask) * -1e9
        
        # Softmax
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = attention / attention.sum(axis=-1, keepdims=True)
        
        # Apply to values
        output = np.matmul(attention, V)
        
        return output, attention
    
    @staticmethod
    def visualize_attention_pattern(seq_len: int = 10):
        """
        Create a simple attention pattern for visualization.
        
        Returns:
            Attention matrix showing different patterns
        """
        attention = np.zeros((seq_len, seq_len))
        
        # Each position attends to different patterns
        for i in range(seq_len):
            # Attend mostly to self
            attention[i, i] = 0.5
            
            # Attend to previous tokens (causal pattern)
            if i > 0:
                attention[i, :i] = 0.3 / i
            
            # Attend to first token (global context)
            attention[i, 0] += 0.2
        
        # Normalize rows
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        return attention


if __name__ == "__main__":
    print("Testing Attention Mechanisms...\n")
    
    # Test 1: Simple attention with NumPy
    print("=== NumPy Attention Demo ===")
    seq_len, d_k = 5, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    output, attention = SimpleAttentionDemo.scaled_dot_product(Q, K, V)
    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    print(f"Attention weights sum (per row): {attention.sum(axis=1)}")
    print()
    
    # Test 2: Multi-head attention with PyTorch
    print("=== PyTorch Multi-Head Attention ===")
    batch_size, seq_len, d_model = 2, 10, 64
    n_heads = 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, attention = mha(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mha.parameters()):,}")
    print()
    
    # Test 3: Positional encoding
    print("=== Positional Encoding ===")
    pe = PositionalEncoding(d_model=64, max_len=100)
    x = torch.randn(1, 20, 64)
    x_with_pos = pe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_with_pos.shape}")
    print()
    
    print("Attention mechanisms ready!")
