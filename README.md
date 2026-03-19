# Dynamic Graph Diffusion Network (DGDN)

A neural network for learning on **graphs that evolve over time**, built on two complementary ideas:

1. **Heat diffusion message passing** — within each graph snapshot, information propagates via the heat equation on the graph Laplacian.
2. **Temporal attention** — a learned attention mechanism integrates node representations across all time steps, capturing how structure and roles change.

---

## Motivation

Many real-world graphs are not static:

- Social networks where friendships form and dissolve
- Molecular interaction networks across biological conditions  
- Transaction graphs where edges carry timestamps
- Traffic or sensor networks with time-varying topology

Standard GNNs process a single fixed graph.  DGDN models the full temporal sequence.

---

## Architecture

```
Snapshot t=0        Snapshot t=1        ...   Snapshot t=T
TemporalGraph       TemporalGraph               TemporalGraph
     │                   │                           │
GraphDiffusionLayer  GraphDiffusionLayer   GraphDiffusionLayer
     │                   │                           │
[N, D] embeddings   [N, D] embeddings         [N, D] embeddings
     └───────────────────┴───────────┬────────────────┘
                                     │
                          TemporalAttention
                       (Time2Vec + MultiHeadAttn)
                                     │
                               [N, D] summary
                                     │
                            mean-pool (graph task)
                                     │
                               Linear head
                                     │
                              class logits
```

### Core components

| Class | File | Role |
|---|---|---|
| `TemporalGraph` | `temporal_graph.py` | Snapshot at time *t*: nodes, edges, timestamps, features |
| `SyntheticTemporalGraph` | `temporal_graph.py` | Generator for synthetic datasets |
| `GraphDiffusionLayer` | `diffusion.py` | Heat diffusion message passing via `exp(-t·L)` |
| `TemporalAttention` | `attention.py` | Multi-head self-attention over snapshot sequence |
| `DynamicGraphDiffusionNet` | `model.py` | Full model combining all components |

### GraphDiffusionLayer

The heat equation on a graph,

```
dX/dt = -L X
```

has the closed-form solution `X(t) = exp(-t·L) X(0)`.  We approximate this with *K* Euler steps:

```
H_k = H_{k-1} + dt · (Â · H_{k-1} - H_{k-1})
    = H_{k-1} · (I + dt·(Â - I))
```

where `Â = D^{-1/2} A D^{-1/2}` is the symmetric normalised adjacency.  Edges are weighted by their age: older edges get exponentially downweighted, so recent topology matters more.

### TemporalAttention

Uses **Time2Vec** positional encoding (`v_k = sin(w_k·t + b_k)`) to embed irregular timestamps, then applies standard multi-head self-attention over the time axis.  Three pooling strategies: `"last"`, `"mean"`, or learned `"attn"`.

---

## Installation

```bash
pip install torch numpy
git clone https://github.com/danieleschmidt/dynamic-graph-diffusion-net
cd dynamic-graph-diffusion-net
# No further installation needed — import directly from src/
```

Dependencies: **PyTorch ≥ 2.0**, **NumPy ≥ 1.24**.  No `torch_geometric` required.

---

## Quick Start

```python
import sys; sys.path.insert(0, "src")
from dgdn import DynamicGraphDiffusionNet, SyntheticTemporalGraph

# Generate a sequence of graph snapshots
gen = SyntheticTemporalGraph(
    num_nodes=20, num_steps=8, feature_dim=16, pattern="community"
)
snapshots = gen.generate()   # list of TemporalGraph objects

# Build the model
model = DynamicGraphDiffusionNet(
    in_features=16,
    hidden_dim=64,
    num_diff_layers=2,
    num_classes=3,
    task="graph",   # or "node"
)

# Forward pass — returns class logits
logits = model(snapshots)     # [3]
embeddings = model.embed(snapshots)  # [20, 64]  node embeddings
```

---

## Demo

```bash
~/anaconda3/bin/python3 examples/demo.py
```

Trains a 3-class classifier to distinguish:
- **random** — edges form/dissolve uniformly at random
- **community** — strong cluster structure, intra-cluster edges are stable
- **growing** — preferential-attachment growth (new nodes connect to hubs)

Sample output:

```
Epoch  60  loss=0.0977  train_acc=98.3%
Test accuracy: 66.7%
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

27 tests covering all components, edge cases (isolated nodes, single snapshots, irregular timestamps), backward pass, and all evolution patterns.

---

## Structure

```
src/dgdn/
  __init__.py          exports
  temporal_graph.py    TemporalGraph, SyntheticTemporalGraph
  diffusion.py         GraphDiffusionLayer
  attention.py         TemporalAttention, TimeEncoding
  model.py             DynamicGraphDiffusionNet

tests/
  test_dgdn.py         27 unit tests

examples/
  demo.py              end-to-end classification demo
```

---

## License

MIT
