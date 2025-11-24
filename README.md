# The Maze and The Map: Understanding Transformers Through Code

<div align="center">

**From Sequential Processing to Parallel Intelligence**

*An educational journey through transformer architectures using maze-solving as a metaphor*

🐉 [Baby Dragon Hatchling](https://github.com/roguetrainer/bdh-welsh-dragons) • 🔗 [Related Projects](#related-work)

</div>

---
![](./img/Mazes-RNNs-vs-Transformer.png)

---
> *“Ah no,” he said, “I see the source of the misunderstanding now. No, look, you see what happened was that we used to do experiments on them. They were often used in behavioral research, Pavlov and all that sort of stuff. So what happened was that the mice would be set all sorts of tests, learning to ring bells, run round mazes and things so that the whole nature of the learning process could be examined. From our observations of their behavior we were able to learn all sorts of things about our own …” Arthur’s voice trailed off.   
“Such subtlety …” said Slartibartfast, “one has to admire it.”    
“What?” said Arthur.    
“How better to disguise their real natures, and how better to guide your thinking. Suddenly running down a maze the wrong way, eating the wrong bit of cheese, unexpectedly dropping dead of myxomatosis. If it’s finely calculated the cumulative effect is enormous.” He paused for effect. “You see, Earthman, they really are particularly clever hyper-intelligent pandimensional beings. Your planet and people have formed the matrix of an organic computer running a ten-million-year research program…. Let me tell you the whole story.”*.  
― Douglas Adams, The Hitchhiker's Guide to the Galaxy
---

## Overview

This repository teaches transformer architecture from first principles through progressive implementation and the metaphor of maze-solving:

- **The Mouse (RNN)**: Sequential processing, step-by-step exploration
- **The Map (Transformer)**: Parallel processing, bird's-eye view
- **The Flow (Neural ODEs)**: Continuous dynamics vs discrete iterations

Rather than explaining transformers abstractly, we build them piece by piece, watching how each component enables the shift from sequential to parallel reasoning.

## The Core Metaphor

When you're dropped into a maze, you have two options:

1. **Walk it step-by-step** (RNN approach)
   - Can only see immediate surroundings
   - Must remember where you've been
   - Memory fades over distance

2. **View from above** (Transformer approach)
   - See entrance, exit, and all paths simultaneously
   - Draw direct connections between any two points
   - No sequential constraint

**The key insight**: Transformers turn a **temporal problem** (processing sequences one step at a time) into a **spatial problem** (processing all positions at once).

## What You'll Learn

### Conceptual Understanding
- Why attention enables "teleportation" between distant positions
- How residual connections create a "conveyor belt" architecture
- The tradeoff between sequential and parallel processing
- Where transformers excel vs where RNNs still matter

### Implementation Skills
- Building attention mechanisms from scratch (NumPy and PyTorch)
- Constructing complete transformer blocks
- Training sequence-to-sequence models
- Analyzing what attention heads learn
- Implementing modern variants (RoPE, sparse attention, etc.)

### Theoretical Grounding
- Connection to Neural ODEs (continuous vs discrete dynamics)
- Computational complexity analysis
- Gradient flow through deep networks
- Positional encoding mathematics

## Repository Structure

```
transformer-maze/
├── src/                          # Core implementation modules
│   ├── maze_envs.py             # Maze generation and utilities
│   ├── visualizations.py        # Plotting and analysis tools
│   ├── rnn_solver.py            # Sequential processing models
│   ├── attention.py             # Attention mechanisms
│   └── transformer_blocks.py    # Complete transformer components
│
├── notebooks/                    # Progressive learning path
│   ├── 01_the_mouse_rnn_maze.ipynb
│   ├── 02_the_map_attention_basics.ipynb
│   ├── 03_building_transformer_blocks.ipynb
│   ├── 04_full_transformer.ipynb
│   ├── 05_modern_variants.ipynb
│   └── 06_continuous_vs_discrete.ipynb
│
├── docs/                         # Documentation
│   ├── mazes_in_ai_history.md   # Historical context
│   └── architecture_guide.md    # Detailed technical reference
│
├── tests/                        # Unit tests
├── requirements.txt             # Python dependencies
├── setup.sh                     # Environment setup script
└── setup.py                     # Package installation
```

## Quick Start

### Prerequisites

- Python 3.8+
- Basic understanding of neural networks
- Familiarity with NumPy and PyTorch

### Installation

```bash
# Clone the repository
git clone https://github.com/roguetrainer/transformer-maze.git
cd transformer-maze

# Run setup script (creates venv, installs dependencies)
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Launch Jupyter

```bash
source venv/bin/activate
jupyter notebook
```

Open `notebooks/01_the_mouse_rnn_maze.ipynb` to begin.

## Learning Path

### Notebook 1: The Mouse in the Maze (RNN)

**Goal**: Understand sequential processing limitations

- Implement RNN/LSTM from scratch
- Train on maze path generation
- Visualize hidden state evolution
- Observe how memory degrades with sequence length

**Key Takeaway**: Sequential processing is inherently limited by the "memory backpack" size.

### Notebook 2: The Map - Attention Fundamentals

**Goal**: See how attention creates instant connections

- Implement scaled dot-product attention (NumPy)
- Visualize attention weights as "teleportation links"
- Apply to maze problem
- Compare RNN vs Attention on same task

**Key Takeaway**: Attention transforms temporal problems into spatial problems.

### Notebook 3: Building Transformer Blocks

**Goal**: Assemble the full machinery

- Multi-head attention (different search strategies)
- Positional encoding (why position matters)
- Residual connections (the "conveyor belt")
- Layer normalization (training stability)

**Key Takeaway**: Transformers are modular - each component serves a specific purpose.

### Notebook 4: Full Transformer on Real Tasks

**Goal**: See complete architecture in action

- Encoder-only (BERT-style): maze classification
- Decoder-only (GPT-style): path generation
- Encoder-decoder (T5-style): maze translation

**Key Takeaway**: Different architectures for different tasks.

### Notebook 5: Modern Variants

**Goal**: Explore the cutting edge

- Sparse attention (efficiency)
- RoPE (better position encoding)
- State space models (Mamba)
- Flash attention (speed optimization)

**Key Takeaway**: The field continues evolving beyond vanilla transformers.

### Notebook 6: Continuous vs Discrete (Neural ODEs)

**Goal**: Connect to broader ML theory

- Implement Neural ODE approach (🇨🇦 UofT contribution)
- Compare discrete layers vs continuous depth
- Show adaptive computation
- Connect to Toronto's ML legacy

**Key Takeaway**: Transformers are one point on a spectrum of approaches.

## Key Visualizations

The repository includes extensive visualizations:

- **Maze solving animations**: Watch models explore step-by-step
- **Attention heatmaps**: See what each head focuses on
- **Hidden state evolution**: Track information flow through layers
- **Performance vs length**: Quantify the sequential bottleneck
- **Multi-head comparison**: Understand specialized attention patterns

## Mathematical Foundations

### The Core Difference

**Sequential (RNN)**:
```
h_t = f(h_{t-1}, x_t)
```
*Must process step-by-step; can't skip ahead*

**Parallel (Transformer)**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
*All positions processed simultaneously*

**Continuous (Neural ODE)**:
```
dh/dt = f(h(t), t, θ)
```
*Adaptive depth based on problem complexity*

See [docs/architecture_guide.md](docs/architecture_guide.md) for detailed derivations.

## Related Work

### Companion Projects

This repository is part of [a broader exploration of deep learning concepts](https://github.com/roguetrainer/around-the-world-in-81-repos):

**Interpretability & Training Dynamics**
- 🐉 [Baby Dragon Hatchling](https://github.com/roguetrainer/bdh-welsh-dragons) - Tracking concept formation in transformers trained on Welsh folk tales
- [Piecewise Linear Surfaces](https://github.com/roguetrainer/piecewise-linear-surfaces-in-deep-learning) - Understanding neural network decision boundaries
- [Pseudo-Spectral Landscapes](https://github.com/roguetrainer/pseudo-spectral-landscapes) - Loss landscape visualization

**Educational Foundations**
- [Deep Learning Not Mysterious](https://github.com/roguetrainer/dl-not-mysterious) - Demystifying neural networks from first principles
- [AI Literacy](https://github.com/roguetrainer/ai-literacy) - Frameworks for understanding AI capabilities and limitations

**Applied AI Systems**
- [Multi-Agent Team](https://github.com/roguetrainer/multi-agent-team) - Coordinating multiple AI agents
- [Agentic RAG](https://github.com/roguetrainer/agentic-rag) - Retrieval-augmented generation systems
- [Local RAG Pipeline](https://github.com/roguetrainer/local-rag-pipeline) - Self-hosted RAG implementation
- [Functional Programming in LLM Interactions](https://github.com/roguetrainer/functional-programming-in-llm-interactions) - Composable AI workflows

**Infrastructure**
- [Kimi K2 Cloud Setup](https://github.com/roguetrainer/kimi-k2-cloud-setup) - Cloud infrastructure for AI development

### Theoretical Foundations

**Neural ODEs** (Chen, Rubanova, Bettencourt, Duvenaud, NeurIPS 2018 Best Paper) 🇨🇦  
University of Toronto  
[Paper](https://arxiv.org/abs/1806.07366)

This influential work from Toronto's ML community treats neural networks as continuous transformations, enabling adaptive computation depth. Notebook 6 explores the connection between discrete transformer layers and continuous depth.

**"Attention Is All You Need"** (Vaswani et al., 2017)  
The original transformer paper that started it all.

## Canadian AI Contributions 🇨🇦

This work builds on Toronto's rich deep learning heritage:

- **Backpropagation**: Hinton, Rumelhart, Williams (1986)
- **Attention Mechanisms**: Bahdanau, Cho, Bengio (2014)
- **Neural ODEs**: Chen et al. (2018) - covered in Notebook 6
- **Transformer Interpretability**: Ongoing work at Anthropic and elsewhere

## Use Cases

### For Students
- Learn transformers by building them
- Understand *why* components exist, not just *what* they are
- Visualize abstract concepts through concrete examples

### For Educators
- Ready-made curriculum for transformer architecture
- Progressive complexity with clear learning objectives
- Extensive visualizations for classroom use

### For Researchers
- Clean implementations to modify and extend
- Benchmarking infrastructure
- Starting point for novel architecture exploration

### For Practitioners
- Deepen intuition beyond using pre-trained models
- Understand when to use which architecture
- Debug attention patterns in production systems

## Contributing

Contributions welcome! Areas of interest:

- Additional maze types (3D, dynamic obstacles)
- More modern architectures (Perceiver, Reformer, etc.)
- Interactive web visualizations
- Translation to other frameworks (JAX, MLX)
- Additional language translations for docs

See `CONTRIBUTING.md` for guidelines.

## Citation

If you use this repository in your research or teaching:

```bibtex
@software{forde2025transformer_maze,
  author = {Forde, Ian},
  title = {The Maze and The Map: Understanding Transformers Through Code},
  year = {2025},
  url = {https://github.com/roguetrainer/transformer-maze}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **David Duvenaud** and the UofT ML group for Neural ODE inspiration
- **Anthropic** for interpretability insights reflected in the BDH project
- **The open-source ML community** for tools and frameworks
- **Toronto's AI ecosystem** for continued innovation

## Contact

**Ian Forde**  
Partnerships Lead, Agnostiq (DataRobot)  
Toronto, ON, Canada

- GitHub: [@roguetrainer](https://github.com/roguetrainer)
- LinkedIn: [Ian Forde](https://www.linkedin.com/in/ianforde)

---

<div align="center">

**"The transformer didn't just speed up maze-solving - it changed the game from navigation to cartography."**

Made with 🧠 in Toronto 🇨🇦

</div>
