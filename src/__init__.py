"""
Transformer Maze: Understanding Transformers Through Code

A progressive implementation of transformer architectures, demonstrating
the shift from sequential (RNN) to parallel (Transformer) processing
through the metaphor of maze solving.
"""

__version__ = "0.1.0"
__author__ = "Ian Forde"

# Core maze utilities
from maze_envs import (
    Maze,
    MazeConfig,
    MazeDataset,
    generate_simple_maze,
)

# Visualization tools
from visualizations import (
    MazeVisualizer,
    AttentionVisualizer,
    TrainingVisualizer,
    ComparisonVisualizer,
    set_style,
)

# RNN components
from rnn_solver import (
    RNNMazeSolver,
    RNNConfig,
    RNNTrainer,
    create_simple_rnn,
    create_lstm,
)

# Attention mechanisms
from attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    RotaryPositionalEmbedding,
    create_causal_mask,
    create_padding_mask,
    SimpleAttentionDemo,
)

# Transformer blocks
from transformer_blocks import (
    FeedForward,
    EncoderBlock,
    DecoderBlock,
    TransformerEncoder,
    TransformerDecoder,
    Transformer,
    create_transformer_for_maze,
)

__all__ = [
    # Maze utilities
    "Maze",
    "MazeConfig",
    "MazeDataset",
    "generate_simple_maze",
    
    # Visualizations
    "MazeVisualizer",
    "AttentionVisualizer",
    "TrainingVisualizer",
    "ComparisonVisualizer",
    "set_style",
    
    # RNN
    "RNNMazeSolver",
    "RNNConfig",
    "RNNTrainer",
    "create_simple_rnn",
    "create_lstm",
    
    # Attention
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionalEncoding",
    "RotaryPositionalEmbedding",
    "create_causal_mask",
    "create_padding_mask",
    "SimpleAttentionDemo",
    
    # Transformer
    "FeedForward",
    "EncoderBlock",
    "DecoderBlock",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
    "create_transformer_for_maze",
]
