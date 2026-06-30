"""
hetero_model.py
===============
Improved GNN for surrogate discovery.

What changed vs. the original SurrogateDiscoveryGNN
───────────────────────────────────────────────────
  1. HETEROGENEOUS message passing.
     Uses `HeteroConv` with a separate `TransformerConv` per relation
     (uplink / backbone / serves / direct + reverses), so a broker↔broker
     link is learned differently from a device→broker link.

  2. EDGE-CENTRIC propagation.
     The naive per-node GRU loop is gone.  Every layer is a real
     message-passing op where edge features modulate every message
     (TransformerConv with `edge_dim`).  This is fully vectorised.

  3. RESIDUAL + NORM + GATING.
     Each layer adds a residual connection and a LayerNorm, plus a
     learned gate that blends new and old node states (GRU-style update,
     but applied to node embeddings, not as an inner loop).

  4. ATTENTION ACROSS PATHS.
     TransformerConv heads give multi-head attention; stacking L layers
     lets information flow along multi-hop paths, the RouteNet intuition.

Temporal variant
────────────────
  `TemporalSurrogateGNN` wraps the spatial encoder with a GRU over the
  sequence of snapshot embeddings, so the model sees how load evolves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HeteroConv, TransformerConv, Linear
from torch_geometric.data import HeteroData


NODE_TYPE_NAMES = ["device", "broker", "surrogate"]
EDGE_DIM = 4


# ──────────────────────────────────────────────────────────────────────
#  One heterogeneous, residual, gated message-passing block
# ──────────────────────────────────────────────────────────────────────

class HeteroMPBlock(nn.Module):
    """
    A single round of typed message passing with:
      • per-relation TransformerConv (edge-feature aware, multi-head)
      • residual connection
      • LayerNorm per node type
      • GRUCell gating to blend old/new state (stabilises deep stacks)
    """
    def __init__(self, hidden_dim: int, metadata, heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        conv_dict = {}
        for edge_type in metadata[1]:                      # metadata[1] = edge types
            conv_dict[edge_type] = TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                edge_dim=EDGE_DIM,
                dropout=0.1,
                beta=True,                                 # learned skip weighting
            )
        self.conv = HeteroConv(conv_dict, aggr="sum")

        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim) for nt in metadata[0]
        })
        self.gates = nn.ModuleDict({
            nt: nn.GRUCell(hidden_dim, hidden_dim) for nt in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        out = self.conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        new_x = {}
        for nt, h_old in x_dict.items():
            h_new = out.get(nt, None)
            if h_new is None or h_new.shape[0] == 0:
                new_x[nt] = h_old
                continue
            h_new = F.relu(h_new)
            # GRU gate: blend previous state with new message
            h_gated = self.gates[nt](h_new, h_old)
            # residual + norm
            new_x[nt] = self.norms[nt](h_gated + h_old)
        return new_x


# ──────────────────────────────────────────────────────────────────────
#  Spatial encoder  (operates on one snapshot)
# ──────────────────────────────────────────────────────────────────────

class HeteroEncoder(nn.Module):
    """
    Encodes a single HeteroData snapshot into per-node embeddings.
    """
    def __init__(self, metadata, in_dim: int = 3, hidden_dim: int = 64,
                 num_layers: int = 3, heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-type input projection (each node type → hidden_dim)
        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for nt in metadata[0]
        })

        self.blocks = nn.ModuleList([
            HeteroMPBlock(hidden_dim, metadata, heads)
            for _ in range(num_layers)
        ])

    def forward(self, data: HeteroData):
        x_dict = {nt: self.input_proj[nt](data[nt].x)
                  for nt in data.node_types}
        edge_index_dict = data.edge_index_dict
        edge_attr_dict  = {et: data[et].edge_attr for et in data.edge_types}

        for block in self.blocks:
            x_dict = block(x_dict, edge_index_dict, edge_attr_dict)
        return x_dict


# ──────────────────────────────────────────────────────────────────────
#  Task heads
# ──────────────────────────────────────────────────────────────────────

class ReadoutHeads(nn.Module):
    """Surrogate scorer (on surrogate nodes) + load predictor (all nodes)."""
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.surrogate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )
        self.load_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, h: Tensor):
        # Raw surrogate score (logit) — kept unbounded so ranking losses work;
        # apply sigmoid externally when a [0,1] score is needed.
        return self.surrogate_head(h).squeeze(-1), self.load_head(h).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
#  Full spatial model
# ──────────────────────────────────────────────────────────────────────

class HeteroSurrogateGNN(nn.Module):
    """
    Spatial-only model: encode one snapshot → score surrogates + predict load.

    forward(data) returns:
      sur_logits   : raw surrogate scores  (on 'surrogate' nodes)         [S]
      load_pred    : predicted load ∈ [0,1] (on all nodes, concatenated)  dict
      h_dict       : per-type node embeddings
    """
    def __init__(self, metadata, in_dim: int = 3, hidden_dim: int = 64,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = HeteroEncoder(metadata, in_dim, hidden_dim, num_layers, heads)
        self.heads   = ReadoutHeads(hidden_dim, dropout)

    def forward(self, data: HeteroData):
        h_dict = self.encoder(data)

        # Surrogate scoring only on surrogate nodes
        sur_logits, _ = self.heads(h_dict["surrogate"])

        # Load prediction on every node type
        load_pred = {}
        for nt, h in h_dict.items():
            _, lp = self.heads(h)
            load_pred[nt] = lp

        return sur_logits, load_pred, h_dict


# ──────────────────────────────────────────────────────────────────────
#  Temporal model  (GRU over snapshot sequence)
# ──────────────────────────────────────────────────────────────────────

class TemporalSurrogateGNN(nn.Module):
    """
    Processes a *sequence* of snapshots.  The shared HeteroEncoder embeds
    each snapshot; a GRU then integrates the surrogate embeddings over time
    before the final scoring head — capturing load trends, not just a
    single instant.
    """
    def __init__(self, metadata, in_dim: int = 3, hidden_dim: int = 64,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = HeteroEncoder(metadata, in_dim, hidden_dim, num_layers, heads)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.heads   = ReadoutHeads(hidden_dim, dropout)

    def forward(self, snapshots: list[HeteroData]):
        """
        snapshots: chronologically ordered list of HeteroData with the SAME
        node set (same topology, evolving features).
        Predicts surrogate scores at the LAST time step using history.
        """
        sur_seq = []        # [T, S, H]
        h_last = None
        for data in snapshots:
            h_dict = self.encoder(data)
            sur_seq.append(h_dict["surrogate"])
            h_last = h_dict

        sur_seq = torch.stack(sur_seq, dim=1)          # [S, T, H]
        out, _  = self.temporal(sur_seq)               # [S, T, H]
        h_final = out[:, -1, :]                        # last step [S, H]

        sur_logits, _ = self.heads(h_final)

        load_pred = {}
        for nt, h in h_last.items():
            _, lp = self.heads(h)
            load_pred[nt] = lp

        return sur_logits, load_pred, h_last
