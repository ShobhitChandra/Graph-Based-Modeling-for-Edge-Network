"""
gnn_model.py
============
RouteNet-Inspired Graph Neural Network for Surrogate Discovery.

Architecture
────────────
RouteNet (Rusek et al., 2020) models networks with two interleaved
message-passing RNNs – one over *links* and one over *paths*.

We adapt this idea for edge-network surrogate discovery:

  1. EdgeMP  – A GRU-based message-passing layer that propagates
               *link-level* embeddings (bandwidth, delay, utilisation …)
               across the graph.

  2. NodeMP  – A standard GCN/GAT layer that aggregates neighbour
               *node* embeddings (capacity, load, type …).

  3. A two-tower readout:
       • Surrogate scorer   → quality score ∈ [0,1]  (regression)
       • Load predictor     → node load ∈ [0,1]      (auxiliary task)

Both RouteNet-style recurrent edge aggregation AND node-level attention
are implemented so you can ablate either component.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GINEConv, global_mean_pool
from torch_geometric.data import Data


# ──────────────────────────────────────────────────────────────────────
#  1.  Edge Embedding Network  (RouteNet-style link encoder)
# ──────────────────────────────────────────────────────────────────────

class EdgeEncoder(nn.Module):
    """Projects raw edge features to a hidden-dim embedding."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_attr: Tensor) -> Tensor:
        return self.net(edge_attr)


# ──────────────────────────────────────────────────────────────────────
#  2.  RouteNet-style Recurrent Edge Aggregation
# ──────────────────────────────────────────────────────────────────────

class RouteNetEdgeAggregation(nn.Module):
    """
    For each node, collect incident edge embeddings and aggregate them
    with a GRU — mimicking RouteNet's path-level LSTM.

    Step-by-step:
      • Sort incident edges by delay (proxy for path ordering)
      • Feed the sequence into a GRU
      • Use the final hidden state as the node's 'link context'
    """
    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(edge_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        edge_embeddings: Tensor,   # [E, edge_dim]
        edge_index: Tensor,        # [2, E]
        num_nodes: int,
    ) -> Tensor:                   # [N, hidden_dim]
        device = edge_embeddings.device
        hidden_dim = self.gru.hidden_size
        agg = torch.zeros(num_nodes, hidden_dim, device=device)

        for node_idx in range(num_nodes):
            # Incident edges where this node is the *destination*
            mask = (edge_index[1] == node_idx)
            inc  = edge_embeddings[mask]          # [k, edge_dim]
            if inc.shape[0] == 0:
                continue
            # GRU over the sequence of incident link embeddings
            _, h = self.gru(inc.unsqueeze(0))     # h: [1, 1, hidden_dim]
            agg[node_idx] = h.squeeze()

        return agg


# ──────────────────────────────────────────────────────────────────────
#  3.  Node Message-Passing Layers  (GAT + GINE)
# ──────────────────────────────────────────────────────────────────────

class HeteroNodeLayer(nn.Module):
    """
    Combines:
      • GATConv  – multi-head attention over *node* neighbourhood
      • GINEConv – edge-feature-aware graph isomorphism network
    Output is the concatenation of both, then projected.
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.gat = GATConv(
            in_channels=node_dim, out_channels=hidden_dim,
            heads=heads, edge_dim=edge_dim, concat=True, dropout=0.1
        )
        # GINEConv: nn operates on (x_j + edge_attr) → needs in_channels == node_dim
        # edge_dim param triggers a linear projection so dims need not match
        gin_nn = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gine = GINEConv(gin_nn, edge_dim=edge_dim)
        self.proj = nn.Linear(hidden_dim * heads + hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        gat_out  = self.gat(x, edge_index, edge_attr)    # [N, hidden*heads]
        gine_out = self.gine(x, edge_index, edge_attr)   # [N, hidden]
        out = self.proj(torch.cat([gat_out, gine_out], dim=-1))
        return self.norm(F.relu(out))


# ──────────────────────────────────────────────────────────────────────
#  4.  Full GNN Model
# ──────────────────────────────────────────────────────────────────────

class SurrogateDiscoveryGNN(nn.Module):
    """
    End-to-end GNN that outputs per-node embeddings and two task heads:

    ┌──────────────────────────────────────────────────────┐
    │  Raw node features  ──►  Node Encoder (MLP)          │
    │  Raw edge features  ──►  Edge Encoder (MLP)          │
    │                                                      │
    │  RouteNetEdgeAggregation  ──► link context           │
    │                                                      │
    │  [node_enc || link_ctx] ──► L × HeteroNodeLayer      │
    │                                                      │
    │  Final node embedding ──► Surrogate Scorer (MLP)     │
    │                       └─► Load Predictor  (MLP)      │
    └──────────────────────────────────────────────────────┘

    Parameters
    ──────────
    node_in_dim   : raw node feature size  (default 6)
    edge_in_dim   : raw edge feature size  (default 4)
    hidden_dim    : internal representation width
    num_layers    : number of message-passing rounds
    heads         : GAT attention heads
    dropout       : dropout rate in readout heads
    """

    def __init__(
        self,
        node_in_dim: int  = 6,
        edge_in_dim: int  = 4,
        hidden_dim: int   = 64,
        num_layers: int   = 3,
        heads: int        = 4,
        dropout: float    = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Encoders ──────────────────────────────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = EdgeEncoder(edge_in_dim, hidden_dim)

        # ── RouteNet-style edge aggregation ───────────────────────────
        self.routenet_agg = RouteNetEdgeAggregation(hidden_dim, hidden_dim)

        # Fuse node_enc + link_ctx
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Stacked message-passing layers ────────────────────────────
        self.mp_layers = nn.ModuleList([
            HeteroNodeLayer(hidden_dim, hidden_dim, hidden_dim, heads)
            for _ in range(num_layers)
        ])

        # ── Task heads ────────────────────────────────────────────────
        self.surrogate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),          # score ∈ [0, 1]
        )
        self.load_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),          # predicted load ∈ [0, 1]
        )

    # ── Forward pass ──────────────────────────────────────────────────
    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. Encode raw features
        node_emb = self.node_encoder(x)                        # [N, H]
        edge_emb = self.edge_encoder(edge_attr)                # [E, H]

        # 2. RouteNet-style link context per node
        link_ctx = self.routenet_agg(edge_emb, edge_index, data.num_nodes)

        # 3. Fuse node + link context
        h = F.relu(self.fuse(torch.cat([node_emb, link_ctx], dim=-1)))

        # 4. Stacked message-passing rounds
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_emb)

        # 5. Task-specific heads
        surrogate_scores = self.surrogate_head(h).squeeze(-1)  # [N]
        load_predictions = self.load_head(h).squeeze(-1)       # [N]

        return surrogate_scores, load_predictions, h           # also return embeddings


# ──────────────────────────────────────────────────────────────────────
#  5.  Loss Function
# ──────────────────────────────────────────────────────────────────────

class SurrogateDiscoveryLoss(nn.Module):
    """
    Multi-task loss:
      L = λ₁ · MSE(surrogate_score, label)   [only on surrogate nodes]
        + λ₂ · MSE(load_pred, true_load)      [all nodes]
    """
    def __init__(self, lambda1: float = 1.0, lambda2: float = 0.3):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mse = nn.MSELoss()

    def forward(
        self,
        surrogate_scores: Tensor,
        load_predictions: Tensor,
        surrogate_labels: Tensor,   # -1 for non-surrogate nodes
        true_loads: Tensor,
        node_type: Tensor,
    ) -> tuple[Tensor, dict]:

        # Only compute surrogate loss on actual surrogate nodes
        sur_mask = (node_type == 2)                             # type 2 = surrogate
        if sur_mask.sum() > 0:
            l_surrogate = self.mse(
                surrogate_scores[sur_mask],
                surrogate_labels[sur_mask],
            )
        else:
            l_surrogate = torch.tensor(0.0)

        l_load = self.mse(load_predictions, true_loads)

        total = self.lambda1 * l_surrogate + self.lambda2 * l_load
        return total, {
            "loss_surrogate": l_surrogate.item(),
            "loss_load":      l_load.item(),
            "loss_total":     total.item(),
        }
