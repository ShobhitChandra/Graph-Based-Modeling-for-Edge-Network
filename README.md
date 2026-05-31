# GNN-Based Surrogate Discovery in Edge Networks
### RouteNet-Inspired Graph Neural Network · PyTorch Geometric

---

## What Was Built

This project models a distributed edge network — made up of **edge devices**, **brokers**, and **surrogates** — as a graph, and trains a GNN to replace DHT-based surrogate discovery with a learned, topology-aware ranking.

It is inspired by **RouteNet** (Rusek et al., 2020), which introduced the idea of using GRU-based message passing over network links and paths to model end-to-end behaviour in computer networks.

---

## Architecture

```
Raw node features (6-dim)  ──►  Node Encoder (MLP)        ──►  node_emb  [N×64]
Raw edge features (4-dim)  ──►  Edge Encoder (MLP)        ──►  edge_emb  [E×64]
                                                               │
                           RouteNetEdgeAggregation (GRU)  ◄───┘
                           ↓ link_ctx  [N×64]
                           fuse(node_emb || link_ctx) → h [N×64]
                                │
              ┌─────────────────┼────────────────────┐
              │  HeteroNodeLayer × 3 (GAT + GINEConv) │
              └─────────────────┬────────────────────┘
                                │  final embedding h [N×64]
              ┌─────────────────┴──────────────────┐
        Surrogate Scorer              Load Predictor
        (MLP → Sigmoid)               (MLP → Sigmoid)
        score ∈ [0,1]                 load ∈ [0,1]
```

### Key design choices mirroring RouteNet:
| RouteNet                      | This Project                                 |
|-------------------------------|----------------------------------------------|
| LSTM over path segments       | GRU over incident edge embeddings per node   |
| Link-state & path embeddings  | EdgeEncoder + RouteNetEdgeAggregation        |
| Per-flow delay prediction     | Per-surrogate quality score prediction       |
| Graph readout                 | Node-level dual-head readout                 |

---

## Project Layout

```
gnn_surrogate_discovery/
├── main.py                        ← Entry point (all run modes)
├── requirements.txt
├── src/
│   ├── graph_builder.py           ← Synthetic edge-network generator
│   │                                 NetworkX → PyG Data conversion
│   │                                 Surrogate quality label generator
│   ├── gnn_model.py               ← RouteNet-inspired GNN
│   │                                 EdgeEncoder, RouteNetEdgeAggregation
│   │                                 HeteroNodeLayer (GAT + GINEConv)
│   │                                 SurrogateDiscoveryGNN
│   │                                 SurrogateDiscoveryLoss
│   ├── dataset.py                 ← EdgeNetworkDataset + train/val/test splits
│   ├── trainer.py                 ← Training loop, LR scheduler, early stopping
│   │                                 TensorBoard logging, checkpointing
│   ├── surrogate_discovery.py     ← Inference: discover(), embedding_similarity_search()
│   │                                 DHT baseline, GNN-vs-DHT comparison utility
│   └── visualize.py               ← Network graph, training curves,
│                                     score distributions, t-SNE embeddings
└── outputs/
    ├── checkpoints/best_model.pt
    ├── history.json
    ├── runs/  (TensorBoard logs)
    └── plots/
        ├── network_graph.png
        ├── training_curves.png
        ├── score_dist.png
        └── tsne_embeddings.png
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train from scratch
python main.py --mode train

# 3. Evaluate on test set (loads best checkpoint)
python main.py --mode evaluate

# 4. Demo: run surrogate discovery on a fresh graph
python main.py --mode discover

# 5. Compare GNN precision@k vs DHT baseline
python main.py --mode compare

# 6. Regenerate all plots
python main.py --mode visualize

# 7. Run the entire pipeline
python main.py --mode all
```

---

## Step-by-Step: How to Understand What Was Built

### Step 1 — Understand the Graph (`src/graph_builder.py`)
Open a Python shell and run:
```python
from src.graph_builder import generate_edge_network, networkx_to_pyg
G    = generate_edge_network(seed=42)
data = networkx_to_pyg(G)
print(data)         # PyG summary
print(data.x[:5])   # First 5 node feature vectors
print(data.edge_attr[:5])  # First 5 edge feature vectors
```
Nodes have a 6-dim feature: [type_one_hot(3), capacity, load, latency_ms].
Edges have a 4-dim feature: [bandwidth_mbps, delay_ms, reliability, utilisation].

### Step 2 — Inspect the Model (`src/gnn_model.py`)
```python
from src.gnn_model import SurrogateDiscoveryGNN
model = SurrogateDiscoveryGNN()
print(model)                         # Full architecture
sum(p.numel() for p in model.parameters())  # Parameter count
```
Walk through the `forward()` method step by step to see how node/edge
features flow through the encoder → RouteNet aggregation → message passing → heads.

### Step 3 — Run Training and Watch TensorBoard
```bash
python main.py --mode train
tensorboard --logdir outputs/runs
```
Open `http://localhost:6006` in your browser. You will see:
- `train/loss_total`, `val/loss_total`
- `val/loss_surrogate` (the main task)
- `val/loss_load`     (auxiliary task)
- `lr`               (learning rate schedule)

### Step 4 — Examine the Plots
- `outputs/plots/network_graph.png` — see node types and topology
- `outputs/plots/training_curves.png` — convergence across epochs
- `outputs/plots/score_dist.png` — how close predictions are to ground truth
- `outputs/plots/tsne_embeddings.png` — are the three node types separable?

### Step 5 — Run Discovery and Compare with DHT
```bash
python main.py --mode compare
```
This prints Precision@k for both GNN and DHT. With more training epochs
and a larger dataset the GNN should beat the hop-count baseline because
it incorporates load and capacity, not just graph distance.

---

## What to Do Next

### Immediate Improvements

1. **Train longer / more data**
   Increase `num_graphs` to 2000 and `epochs` to 200 in `CFG` inside `main.py`.

2. **Use a real topology**
   Replace `generate_edge_network()` with a parser for BRITE / CAIDA / Rocketfuel
   traces. Pass real bandwidth and latency measurements as edge features.

3. **Add heterogeneous node types (HeteroData)**
   PyG's `HeteroData` lets you define separate message-passing channels for
   device→broker, broker→surrogate, etc. This matches the real-world hierarchy better.

4. **Temporal graphs**
   Wrap the model in a Temporal-GNN (e.g. `torch_geometric_temporal`) to handle
   dynamic load changes. Each time step becomes a graph snapshot.

5. **Contrastive / self-supervised pre-training**
   Use GraphCL or BGRL to pre-train node embeddings on unlabelled network traces
   before fine-tuning the surrogate scorer on labelled data.

### Research Extensions (toward a paper contribution)

6. **Federated GNN training**
   Each broker trains a local GNN on its neighbourhood; periodically exchange
   model parameters. This models real distributed discovery without a central oracle.

7. **Replace DHT completely**
   Use the GNN as the routing function: given a service request, produce an
   attention-weighted path to the best surrogate end-to-end.

8. **Evaluate with real service metrics**
   Replace the synthetic quality label with a real SLO metric (response time,
   cache hit rate, energy cost). Collect traces from a MEC / fog testbed.

9. **Compare with graph-based DHT variants**
   Chord, Kademlia, CAN — implement them in NetworkX and add them to
   `surrogate_discovery.py`'s comparison harness.

10. **Explainability**
    Use `torch_geometric.explain` (GNNExplainer / CaptumExplainer) to highlight
    which edges and nodes drive each surrogate recommendation. Critical for debugging
    and for convincing system operators to trust the GNN.

---

## Paper Reference

> Rusek, K., Suárez-Varela, J., Mestres, A., Barlet-Ros, P., & Cabellos-Aparicio, A. (2020).
> **RouteNet: Leveraging Graph Neural Networks for Network Modeling and Optimization
> in SDN**. *IEEE Journal on Selected Areas in Communications*, 38(10), 2260–2270.
> https://doi.org/10.1109/JSAC.2020.3000405

Key ideas borrowed:
- Treating links as first-class graph citizens with their own embedding state
- GRU-based sequential aggregation of incident link states per node
- Interleaved node-link message passing for end-to-end network modelling
