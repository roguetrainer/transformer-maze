# Quick Reference Guide

## Module Import Patterns

### Basic Maze Operations

```python
from maze_envs import generate_simple_maze, MazeDataset, MazeConfig

# Quick maze
maze = generate_simple_maze(size=15, seed=42)
solution = maze.solve()
print(maze.to_text(show_solution=True))

# Custom maze
config = MazeConfig(
    height=20, 
    width=20, 
    wall_probability=0.3,
    ensure_solvable=True,
    seed=123
)
maze = Maze(config)

# Dataset for training
dataset = MazeDataset(num_mazes=100, config=config)
training_pairs = dataset.get_training_pairs()
```

### Visualization

```python
from visualizations import (
    MazeVisualizer, 
    AttentionVisualizer, 
    TrainingVisualizer,
    set_style
)

set_style()  # Apply consistent styling

# Plot maze
fig, ax = plt.subplots(figsize=(8, 8))
MazeVisualizer.plot_maze(maze, ax=ax, show_solution=True)

# Plot attention
AttentionVisualizer.plot_attention_heatmap(
    attention_weights,
    title="Self-Attention Pattern"
)

# Training curves
TrainingVisualizer.plot_training_curves({
    'train_loss': losses,
    'val_loss': val_losses
})
```

### RNN Models

```python
from rnn_solver import create_lstm, RNNTrainer

# Create model
model = create_lstm(maze_size=15, hidden_dim=128)

# Training
trainer = RNNTrainer(model, learning_rate=1e-3)

# Training step
loss = trainer.train_step(
    maze_grids,      # [B, H, W]
    position_seqs,   # [B, T, 2]
    action_seqs      # [B, T]
)

# Generation
path = model.generate_path(maze_grid, start_pos, max_steps=100)
```

### Attention Mechanisms

```python
from attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    SimpleAttentionDemo
)

# NumPy attention (for teaching)
Q = np.random.randn(10, 64)
K = np.random.randn(10, 64)
V = np.random.randn(10, 64)

output, attn_weights = SimpleAttentionDemo.scaled_dot_product(Q, K, V)

# PyTorch multi-head
mha = MultiHeadAttention(d_model=256, n_heads=8)
x = torch.randn(batch_size, seq_len, 256)
output, attention = mha(x, x, x)

# Positional encoding
pos_enc = PositionalEncoding(d_model=256)
x_with_pos = pos_enc(x)
```

### Transformer Blocks

```python
from transformer_blocks import (
    EncoderBlock,
    TransformerEncoder,
    Transformer,
    create_transformer_for_maze
)

# Single encoder block
encoder_block = EncoderBlock(d_model=256, n_heads=8, d_ff=1024)
output, attention = encoder_block(x)

# Full encoder
encoder = TransformerEncoder(
    num_layers=6,
    d_model=256,
    n_heads=8
)
encoded, all_attentions = encoder(x)

# Complete transformer
model = create_transformer_for_maze(
    maze_size=15,
    d_model=256,
    n_heads=8,
    num_layers=4
)

# Forward pass
logits = model(src_tokens, tgt_tokens)

# Generation
generated = model.generate(src_tokens, max_len=100)
```

## Common Workflows

### Workflow 1: Train RNN on Mazes

```python
# 1. Generate data
dataset = MazeDataset(num_mazes=1000, config=maze_config)

# 2. Create model
model = create_lstm(maze_size=15, hidden_dim=128)
trainer = RNNTrainer(model)

# 3. Training loop
for epoch in range(num_epochs):
    for maze, solution in dataset:
        # Convert to tensors
        maze_grid = torch.tensor(maze.grid).unsqueeze(0)
        positions = torch.tensor(solution).unsqueeze(0)
        actions = torch.tensor(maze.path_to_actions(solution)).unsqueeze(0)
        
        # Train
        loss = trainer.train_step(maze_grid, positions, actions)
    
    print(f"Epoch {epoch}: Loss = {loss:.4f}")

# 4. Evaluate
test_maze = generate_simple_maze(seed=999)
pred_path = model.generate_path(test_maze.grid, test_maze.start)

# 5. Visualize
MazeVisualizer.plot_maze_comparison(
    test_maze, 
    predicted_path=pred_path,
    true_path=test_maze.solution
)
```

### Workflow 2: Attention Analysis

```python
# 1. Create attention layer
attention = MultiHeadAttention(d_model=256, n_heads=8)

# 2. Forward pass
output, attn_weights = attention(x, x, x)

# 3. Visualize attention patterns
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    # attn_weights: [B, n_heads, seq_len, seq_len]
    head_attention = attn_weights[0, i].cpu().numpy()
    AttentionVisualizer.plot_attention_heatmap(
        head_attention,
        ax=ax,
        title=f"Head {i+1}"
    )
```

### Workflow 3: Compare RNN vs Transformer

```python
# 1. Create both models
rnn_model = create_lstm(maze_size=15, hidden_dim=128)
transformer = create_transformer_for_maze(maze_size=15)

# 2. Test on increasing maze sizes
sizes = [10, 15, 20, 25, 30]
rnn_accuracies = []
transformer_accuracies = []

for size in sizes:
    test_mazes = [generate_simple_maze(size=size, seed=i) 
                  for i in range(50)]
    
    rnn_correct = sum(test_rnn(rnn_model, m) for m in test_mazes)
    trans_correct = sum(test_transformer(transformer, m) for m in test_mazes)
    
    rnn_accuracies.append(rnn_correct / 50)
    transformer_accuracies.append(trans_correct / 50)

# 3. Visualize comparison
TrainingVisualizer.plot_performance_vs_length(
    sizes,
    [rnn_accuracies, transformer_accuracies],
    ['RNN (LSTM)', 'Transformer']
)
```

## Common Gotchas

### Issue 1: Tensor Shapes
```python
# Wrong: Forgot batch dimension
maze_grid = torch.tensor(maze.grid)  # [H, W]

# Right: Include batch
maze_grid = torch.tensor(maze.grid).unsqueeze(0)  # [1, H, W]
```

### Issue 2: Position Encoding
```python
# Wrong: Forget to add positional encoding
output = encoder(embeddings)

# Right: Add before encoding
embeddings = pos_encoding(embeddings)
output = encoder(embeddings)
```

### Issue 3: Attention Mask
```python
# Wrong: No mask for causal attention
output, attn = decoder(x, encoder_out)

# Right: Create causal mask
tgt_mask = create_causal_mask(seq_len, device)
output, attn = decoder(x, encoder_out, tgt_mask=tgt_mask)
```

### Issue 4: Device Mismatch
```python
# Wrong: Model on GPU, data on CPU
model = model.cuda()
loss = model(data)  # Error!

# Right: Move data to same device
model = model.cuda()
data = data.cuda()
loss = model(data)
```

## Performance Tips

### For Training
- Start with small mazes (10×10)
- Use batch size 32-64
- Learning rate 1e-3 to 1e-4
- Gradient clipping (max_norm=1.0)
- Early stopping on validation loss

### For Visualization
- Use `set_style()` at notebook start
- Save figures: `fig.savefig('output.png', dpi=150, bbox_inches='tight')`
- For papers/presentations, use vector format: `fig.savefig('output.pdf')`

### For Debugging
- Print tensor shapes liberally
- Visualize attention weights early
- Check gradient flow: `print(param.grad.norm())`
- Use smaller models first
- Test on single example before batching

## Useful Snippets

### Check Model Size
```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,} | Trainable: {trainable_params:,}")
```

### Save/Load Model
```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Profile Training Time
```python
import time

start = time.time()
for epoch in range(num_epochs):
    # training code
    pass
elapsed = time.time() - start
print(f"Training took {elapsed:.2f}s ({elapsed/num_epochs:.2f}s per epoch)")
```

### Reproducibility
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

**Remember**: All modules are in `src/`, so notebooks should start with:
```python
import sys
sys.path.insert(0, '../src')
```

Or after installation:
```python
# Just import directly
from maze_envs import generate_simple_maze
```
