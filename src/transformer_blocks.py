"""
Transformer Blocks - Complete Architecture Components

This module implements full transformer encoder and decoder blocks,
combining attention with feed-forward networks and residual connections.

The "conveyor belt" architecture where information flows through layers
with incremental refinements at each step.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from attention import MultiHeadAttention, PositionalEncoding, create_causal_mask


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Applied independently to each position after attention.
    Typically: FFN(x) = max(0, xW1 + b1)W2 + b2
    
    This adds non-linear transformation capacity beyond what attention provides.
    """
    
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern choice; original used ReLU
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.
    
    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> Add -> 
             LayerNorm -> FeedForward -> Add -> output
    
    The residual connections (Add) are the "conveyor belt" - 
    most information flows straight through, with layers making small refinements.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention: [batch_size, n_heads, seq_len, seq_len]
        """
        # Self-attention with residual connection
        # Pre-LN variant (used in modern transformers)
        residual = x
        x = self.norm1(x)
        attn_output, attention = self.attention(x, x, x, mask)
        x = residual + self.dropout1(attn_output)
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout2(ff_output)
        
        return x, attention


class DecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block.
    
    Architecture:
        x -> LayerNorm -> Masked Self-Attention -> Add ->
             LayerNorm -> Cross-Attention (with encoder) -> Add ->
             LayerNorm -> FeedForward -> Add -> output
    
    The masked self-attention prevents looking at future tokens (autoregressive).
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Decoder input [batch_size, tgt_len, d_model]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            src_mask: Source mask for cross-attention
            tgt_mask: Target mask for self-attention (causal)
            
        Returns:
            output: [batch_size, tgt_len, d_model]
            self_attention: [batch_size, n_heads, tgt_len, tgt_len]
            cross_attention: [batch_size, n_heads, tgt_len, src_len]
        """
        # Masked self-attention
        residual = x
        x = self.norm1(x)
        self_attn_output, self_attention = self.self_attention(x, x, x, tgt_mask)
        x = residual + self.dropout1(self_attn_output)
        
        # Cross-attention with encoder output
        residual = x
        x = self.norm2(x)
        cross_attn_output, cross_attention = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = residual + self.dropout2(cross_attn_output)
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout3(ff_output)
        
        return x, self_attention, cross_attention


class TransformerEncoder(nn.Module):
    """
    Stack of Encoder Blocks.
    
    This is the "map builder" - processes the entire input sequence in parallel,
    building rich representations at each layer.
    """
    
    def __init__(self, num_layers: int, d_model: int, n_heads: int,
                 d_ff: int = 2048, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: List of attention matrices from each layer
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        attention_weights = []
        
        # Pass through encoder layers
        for layer in self.layers:
            x, attention = layer(x, mask)
            attention_weights.append(attention)
        
        # Final layer norm
        x = self.norm(x)
        
        return x, attention_weights


class TransformerDecoder(nn.Module):
    """
    Stack of Decoder Blocks.
    
    Generates output sequence autoregressively, attending to both
    previously generated tokens and the encoder output.
    """
    
    def __init__(self, num_layers: int, d_model: int, n_heads: int,
                 d_ff: int = 2048, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list, list]:
        """
        Args:
            x: Decoder input [batch_size, tgt_len, d_model]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            src_mask: Source mask
            tgt_mask: Target mask (causal)
            
        Returns:
            output: [batch_size, tgt_len, d_model]
            self_attention_weights: List of self-attention matrices
            cross_attention_weights: List of cross-attention matrices
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        self_attention_weights = []
        cross_attention_weights = []
        
        # Pass through decoder layers
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        # Final layer norm
        x = self.norm(x)
        
        return x, self_attention_weights, cross_attention_weights


class Transformer(nn.Module):
    """
    Complete Transformer model (Encoder-Decoder architecture).
    
    The full "Attention is All You Need" architecture.
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, n_heads: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, n_heads, d_ff, dropout, max_len
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, n_heads, d_ff, dropout, max_len
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_mask: Source mask
            tgt_mask: Target mask
            
        Returns:
            logits: [batch_size, tgt_len, tgt_vocab_size]
        """
        # Embed source and target
        src_embedded = self.src_embedding(src)
        tgt_embedded = self.tgt_embedding(tgt)
        
        # Encode
        encoder_output, _ = self.encoder(src_embedded, src_mask)
        
        # Decode
        decoder_output, _, _ = self.decoder(tgt_embedded, encoder_output, 
                                           src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(self, src: torch.Tensor, max_len: int = 100, 
                start_token: int = 1, end_token: int = 2) -> torch.Tensor:
        """
        Autoregressive generation (greedy decoding).
        
        Args:
            src: Source sequence [batch_size, src_len]
            max_len: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            
        Returns:
            generated: [batch_size, generated_len]
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        src_embedded = self.src_embedding(src)
        encoder_output, _ = self.encoder(src_embedded)
        
        # Initialize decoder input with start token
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        # Generate token by token
        for _ in range(max_len):
            # Create causal mask
            tgt_len = tgt.size(1)
            tgt_mask = create_causal_mask(tgt_len, device).unsqueeze(0).unsqueeze(0)
            
            # Decode
            tgt_embedded = self.tgt_embedding(tgt)
            decoder_output, _, _ = self.decoder(tgt_embedded, encoder_output, 
                                               tgt_mask=tgt_mask)
            
            # Get next token prediction
            logits = self.output_projection(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if all sequences have generated end token
            if (next_token == end_token).all():
                break
        
        return tgt


def create_transformer_for_maze(maze_size: int = 15, d_model: int = 256,
                               n_heads: int = 8, num_layers: int = 4) -> Transformer:
    """
    Convenience function to create transformer for maze solving.
    
    Args:
        maze_size: Size of maze
        d_model: Model dimension
        n_heads: Number of attention heads
        num_layers: Number of encoder/decoder layers
        
    Returns:
        Transformer model
    """
    # Vocabulary: wall, path, start, goal + actions + special tokens
    vocab_size = 10  # Simplified
    
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_model * 4,
        max_len=maze_size * maze_size
    )
    
    return model


if __name__ == "__main__":
    print("Testing Transformer Blocks...\n")
    
    # Test encoder block
    print("=== Encoder Block ===")
    encoder_block = EncoderBlock(d_model=256, n_heads=8, d_ff=1024)
    x = torch.randn(2, 10, 256)
    output, attention = encoder_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    print()
    
    # Test full encoder
    print("=== Full Encoder ===")
    encoder = TransformerEncoder(num_layers=4, d_model=256, n_heads=8)
    output, attentions = encoder(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of attention matrices: {len(attentions)}")
    print()
    
    # Test full transformer
    print("=== Complete Transformer ===")
    model = create_transformer_for_maze(maze_size=15, d_model=256, 
                                       n_heads=8, num_layers=4)
    
    src = torch.randint(0, 10, (2, 20))
    tgt = torch.randint(0, 10, (2, 15))
    
    output = model(src, tgt)
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    print("Transformer blocks ready!")
