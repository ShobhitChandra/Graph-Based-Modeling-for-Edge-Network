"""
graph_builder.py
================
Constructs the Edge Network Graph where:
  - Nodes  = Brokers | Surrogates | Edge-Devices
  - Edges  = Network links (weighted by bandwidth, latency, reliability)

Inspired by RouteNet: every node / link carries feature vectors that the
GNN will learn to aggregate.
"""

import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


# ─────────────────────────────────────────────
#  Node / Edge feature dimensionalities
# ─────────────────────────────────────────────
NODE_FEATURE_DIM = 6   # [type_onehot(3), capacity, load, latency_ms]
EDGE_FEATURE_DIM = 4   # [bandwidth_mbps, delay_ms, reliability, utilisation]

NODE_TYPES = {"edge_device": 0, "broker": 1, "surrogate": 2}


# ─────────────────────────────────────────────
#  Synthetic network generator
# ─────────────────────────────────────────────

def generate_edge_network(
    n_edge_devices: int = 10,
    n_brokers: int = 4,
    n_surrogates: int = 6,
    seed: int = 42,
) -> nx.Graph:
    """
    Create a synthetic edge-network graph.

    Topology rules
    ──────────────
    • Edge devices connect to ≥1 broker  (access links – high latency)
    • Brokers are fully meshed           (backbone links – low latency)
    • Surrogates attach to brokers       (service links – medium latency)
    • A few edge-device↔surrogate direct links (opportunistic paths)
    """
    random.seed(seed)
    np.random.seed(seed)

    G = nx.Graph()
    node_id = 0

    edge_device_ids, broker_ids, surrogate_ids = [], [], []

    # ── Add nodes ──────────────────────────────────────────────────────
    for _ in range(n_edge_devices):
        feats = _edge_device_features()
        G.add_node(node_id, **feats)
        edge_device_ids.append(node_id)
        node_id += 1

    for _ in range(n_brokers):
        feats = _broker_features()
        G.add_node(node_id, **feats)
        broker_ids.append(node_id)
        node_id += 1

    for _ in range(n_surrogates):
        feats = _surrogate_features()
        G.add_node(node_id, **feats)
        surrogate_ids.append(node_id)
        node_id += 1

    # ── Add edges ──────────────────────────────────────────────────────
    # 1. Edge-devices → Brokers (each device connects to 1–2 random brokers)
    for ed in edge_device_ids:
        k = random.randint(1, min(2, n_brokers))
        for b in random.sample(broker_ids, k):
            G.add_edge(ed, b, **_access_link())

    # 2. Broker full mesh
    for i, b1 in enumerate(broker_ids):
        for b2 in broker_ids[i + 1:]:
            G.add_edge(b1, b2, **_backbone_link())

    # 3. Surrogates → Brokers (each surrogate connects to 1–2 brokers)
    for s in surrogate_ids:
        k = random.randint(1, min(2, n_brokers))
        for b in random.sample(broker_ids, k):
            G.add_edge(s, b, **_service_link())

    # 4. Direct device→surrogate shortcuts (30 % chance)
    for ed in edge_device_ids:
        for s in surrogate_ids:
            if random.random() < 0.3:
                G.add_edge(ed, s, **_opportunistic_link())

    return G


# ─────────────────────────────────────────────
#  Node feature factories
# ─────────────────────────────────────────────

def _one_hot(idx: int, size: int = 3) -> list:
    v = [0.0] * size
    v[idx] = 1.0
    return v


def _edge_device_features() -> dict:
    oh = _one_hot(NODE_TYPES["edge_device"])
    return dict(
        node_type=NODE_TYPES["edge_device"],
        x=oh + [
            round(random.uniform(0.1, 1.0), 3),   # capacity (normalised)
            round(random.uniform(0.0, 0.9), 3),   # current load
            round(random.uniform(5, 50), 2),       # local latency ms
        ],
    )


def _broker_features() -> dict:
    oh = _one_hot(NODE_TYPES["broker"])
    return dict(
        node_type=NODE_TYPES["broker"],
        x=oh + [
            round(random.uniform(0.6, 1.0), 3),
            round(random.uniform(0.1, 0.7), 3),
            round(random.uniform(1, 10), 2),
        ],
    )


def _surrogate_features() -> dict:
    oh = _one_hot(NODE_TYPES["surrogate"])
    return dict(
        node_type=NODE_TYPES["surrogate"],
        x=oh + [
            round(random.uniform(0.4, 1.0), 3),
            round(random.uniform(0.0, 0.8), 3),
            round(random.uniform(2, 20), 2),
        ],
    )


# ─────────────────────────────────────────────
#  Edge feature factories
# ─────────────────────────────────────────────

def _access_link() -> dict:
    return dict(edge_attr=[
        round(random.uniform(10, 100), 2),    # bandwidth Mbps
        round(random.uniform(10, 80), 2),     # delay ms
        round(random.uniform(0.8, 1.0), 3),  # reliability
        round(random.uniform(0.1, 0.6), 3),  # utilisation
    ])


def _backbone_link() -> dict:
    return dict(edge_attr=[
        round(random.uniform(100, 1000), 2),
        round(random.uniform(1, 10), 2),
        round(random.uniform(0.95, 1.0), 3),
        round(random.uniform(0.2, 0.8), 3),
    ])


def _service_link() -> dict:
    return dict(edge_attr=[
        round(random.uniform(50, 500), 2),
        round(random.uniform(2, 30), 2),
        round(random.uniform(0.9, 1.0), 3),
        round(random.uniform(0.1, 0.5), 3),
    ])


def _opportunistic_link() -> dict:
    return dict(edge_attr=[
        round(random.uniform(5, 50), 2),
        round(random.uniform(20, 100), 2),
        round(random.uniform(0.7, 0.95), 3),
        round(random.uniform(0.0, 0.4), 3),
    ])


# ─────────────────────────────────────────────
#  NetworkX → PyG Data conversion
# ─────────────────────────────────────────────

def networkx_to_pyg(G: nx.Graph) -> Data:
    """Convert the NetworkX graph to a PyTorch-Geometric Data object."""
    node_features, edge_index_list, edge_features = [], [], []

    node_list = list(G.nodes())
    nid_map = {n: i for i, n in enumerate(node_list)}

    for n in node_list:
        node_features.append(G.nodes[n]["x"])

    for u, v, attrs in G.edges(data=True):
        i, j = nid_map[u], nid_map[v]
        edge_index_list += [[i, j], [j, i]]            # undirected → bidirectional
        edge_features   += [attrs["edge_attr"]] * 2

    x          = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_features,  dtype=torch.float)

    # Node-type mask tensors (useful for loss masking)
    node_types = torch.tensor(
        [G.nodes[n]["node_type"] for n in node_list], dtype=torch.long
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                node_type=node_types)
    data.num_nodes = x.shape[0]
    return data


# ─────────────────────────────────────────────
#  Label generation  (supervised target)
# ─────────────────────────────────────────────

def generate_surrogate_labels(data: Data) -> torch.Tensor:
    """
    Synthetic 'quality score' for each surrogate node [0, 1].

    Score = f(capacity, load, latency)  – higher is better.
    Only surrogate nodes get meaningful labels; others receive −1.
    """
    labels = torch.full((data.num_nodes,), -1.0)
    for i in range(data.num_nodes):
        if data.node_type[i].item() == NODE_TYPES["surrogate"]:
            cap  = data.x[i, 3].item()   # capacity
            load = data.x[i, 4].item()   # load
            lat  = data.x[i, 5].item()   # latency (ms)
            score = cap * (1 - load) / (1 + lat / 50.0)
            labels[i] = round(min(max(score, 0.0), 1.0), 4)
    return labels
