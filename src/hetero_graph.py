"""
hetero_graph.py
===============
Converts a role-labelled nx.Graph (from real_data.py) into a PyG
`HeteroData` object with separate node and edge *types*, enabling
type-specific message passing.

Node types
──────────
  'device'     – edge devices (requesters)
  'broker'     – rendezvous / routing nodes
  'surrogate'  – service hosts (ranking targets)

Edge (relation) types — only the physically meaningful ones are kept:
  ('device',    'uplink',   'broker')      device → broker access link
  ('broker',    'backbone', 'broker')      broker ↔ broker core link
  ('broker',    'serves',   'surrogate')   broker → surrogate service link
  ('device',    'direct',   'surrogate')   opportunistic direct link
Plus every relation's reverse (added with ToUndirected) so messages
flow both ways.

Each node type keeps its own `x` (we drop the 3 one-hot type dims since
the type is now encoded structurally → 3 raw features remain:
[capacity, load, latency_ms]).  Each edge type keeps its 4-dim edge_attr.
"""

import torch
import networkx as nx
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from src.real_data import NODE_TYPES


TYPE_TO_NAME = {0: "device", 1: "broker", 2: "surrogate"}

# Which (src_type, dst_type) pairs map to which relation name
RELATION_OF = {
    ("device", "broker"):     "uplink",
    ("broker", "broker"):     "backbone",
    ("broker", "surrogate"):  "serves",
    ("device", "surrogate"):  "direct",
}


def _relation(src_name: str, dst_name: str):
    """Return canonical (relation_name, flipped?) for an undirected pair."""
    if (src_name, dst_name) in RELATION_OF:
        return RELATION_OF[(src_name, dst_name)], False
    if (dst_name, src_name) in RELATION_OF:
        return RELATION_OF[(dst_name, src_name)], True
    return None, None


def nx_to_hetero(G: nx.Graph) -> HeteroData:
    """
    Build a HeteroData object from a single role-labelled snapshot graph.
    """
    data = HeteroData()

    # ── Group nodes by type and build per-type local index maps ───────
    local_idx = {"device": {}, "broker": {}, "surrogate": {}}
    feats     = {"device": [], "broker": [], "surrogate": []}
    labels    = {"device": [], "broker": [], "surrogate": []}
    global_id = {"device": [], "broker": [], "surrogate": []}

    for n, d in G.nodes(data=True):
        name = TYPE_TO_NAME[d["node_type"]]
        local_idx[name][n] = len(feats[name])
        feats[name].append(d["x"][3:])          # drop one-hot → [cap, load, lat]
        labels[name].append(d.get("y", -1.0))
        global_id[name].append(n)

    for name in ["device", "broker", "surrogate"]:
        if feats[name]:
            data[name].x = torch.tensor(feats[name], dtype=torch.float)
            data[name].y = torch.tensor(labels[name], dtype=torch.float)
            data[name].global_id = torch.tensor(global_id[name], dtype=torch.long)
        else:
            data[name].x = torch.zeros((0, 3), dtype=torch.float)
            data[name].y = torch.zeros((0,), dtype=torch.float)
            data[name].global_id = torch.zeros((0,), dtype=torch.long)

    # ── Build typed edges ─────────────────────────────────────────────
    rel_edges = {}      # (src,rel,dst) -> [[src_local...],[dst_local...]]
    rel_attrs = {}      # (src,rel,dst) -> [edge_attr...]

    for u, v, d in G.edges(data=True):
        nu = TYPE_TO_NAME[G.nodes[u]["node_type"]]
        nv = TYPE_TO_NAME[G.nodes[v]["node_type"]]
        rel, flip = _relation(nu, nv)
        if rel is None:
            continue
        if flip:                                 # canonical src→dst order
            u, v, nu, nv = v, u, nv, nu
        key = (nu, rel, nv)
        rel_edges.setdefault(key, [[], []])
        rel_attrs.setdefault(key, [])
        rel_edges[key][0].append(local_idx[nu][u])
        rel_edges[key][1].append(local_idx[nv][v])
        rel_attrs[key].append(d["edge_attr"])

    for key, (src, dst) in rel_edges.items():
        data[key].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data[key].edge_attr  = torch.tensor(rel_attrs[key], dtype=torch.float)

    # ── Make undirected (adds reverse relations) ──────────────────────
    data = T.ToUndirected(merge=False)(data)

    return data


def collate_temporal(snapshots: list[nx.Graph]) -> list[HeteroData]:
    """Convert a list of snapshot graphs to a list of HeteroData objects."""
    return [nx_to_hetero(g) for g in snapshots]
