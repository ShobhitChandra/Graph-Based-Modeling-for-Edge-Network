"""
real_data.py
============
Loads REAL network topologies from the Internet Topology Zoo
(https://topology-zoo.org — 250+ operator-provided PoP-level networks)
and turns them into role-labelled, temporally-dynamic edge-network graphs.

This replaces the fully synthetic generator. What is *real* here:
  • Topology / connectivity        → directly from operator GraphML
  • Node geolocation (lat / lon)   → from GraphML
  • Link propagation delay         → haversine distance / speed-of-light-in-fibre
What is *simulated* on top (because the Zoo has no load traces):
  • Node roles (device/broker/surrogate) → assigned by graph centrality
  • Per-node capacity / load             → diurnal traffic model
  • Temporal snapshots                   → load & utilisation evolve over T steps

Public API
──────────
  load_topology(path)                  → nx.Graph with roles + geo
  assign_roles(G)                      → tag each node device|broker|surrogate
  build_temporal_snapshots(G, T)       → list[nx.Graph] (one per time step)
  list_topologies(dir)                 → available .graphml paths
"""

import os
import glob
import math
import random
import numpy as np
import networkx as nx


NODE_TYPES = {"edge_device": 0, "broker": 1, "surrogate": 2}
SPEED_OF_LIGHT_FIBRE_KM_S = 200_000.0   # ~2/3 c in optical fibre


# ──────────────────────────────────────────────────────────────────────
#  Topology discovery / loading
# ──────────────────────────────────────────────────────────────────────

def list_topologies(directory: str) -> list[str]:
    """Return sorted list of .graphml topology files."""
    return sorted(glob.glob(os.path.join(directory, "*.graphml")))


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def load_topology(path: str) -> nx.Graph:
    """
    Read a Topology-Zoo GraphML file and return a clean, relabelled
    undirected nx.Graph with integer node ids and geo attributes.
    """
    G_raw = nx.read_graphml(path)
    G_raw = nx.Graph(G_raw)            # collapse multi-edges, force undirected

    # Relabel nodes to contiguous integers
    mapping = {n: i for i, n in enumerate(G_raw.nodes())}
    G = nx.relabel_nodes(G_raw, mapping)

    # Attach geo coords (fall back to random jitter if missing on a node)
    for n, d in G.nodes(data=True):
        try:
            d["lat"] = float(d.get("Latitude"))
            d["lon"] = float(d.get("Longitude"))
        except (TypeError, ValueError):
            d["lat"] = random.uniform(-60, 60)
            d["lon"] = random.uniform(-120, 120)

    # Compute real propagation delay per link from geo distance
    for u, v, d in G.edges(data=True):
        km = _haversine_km(G.nodes[u]["lat"], G.nodes[u]["lon"],
                           G.nodes[v]["lat"], G.nodes[v]["lon"])
        km = max(km, 1.0)                                   # avoid 0
        d["distance_km"]  = round(km, 2)
        d["prop_delay_ms"] = round(km / SPEED_OF_LIGHT_FIBRE_KM_S * 1000, 4)

    return G


# ──────────────────────────────────────────────────────────────────────
#  Role assignment  (structure-driven, deterministic)
# ──────────────────────────────────────────────────────────────────────

def assign_roles(G: nx.Graph, broker_frac: float = 0.25,
                 surrogate_frac: float = 0.35, seed: int = 0) -> nx.Graph:
    """
    Assign device / broker / surrogate roles using betweenness centrality:

      • Top `broker_frac` most-central nodes      → BROKERS
        (they sit on the most shortest paths — natural rendezvous points)
      • Next `surrogate_frac`                      → SURROGATES
        (well-connected service hosts)
      • Remaining leaf-ish nodes                   → EDGE DEVICES

    Roles are written to node attribute `node_type` (int) and `role` (str).
    """
    random.seed(seed)
    bc = nx.betweenness_centrality(G)
    ranked = sorted(G.nodes(), key=lambda n: bc[n], reverse=True)

    n = len(ranked)
    n_broker    = max(1, int(round(n * broker_frac)))
    n_surrogate = max(1, int(round(n * surrogate_frac)))

    broker_set    = set(ranked[:n_broker])
    surrogate_set = set(ranked[n_broker:n_broker + n_surrogate])

    for node in G.nodes():
        if node in broker_set:
            role, t = "broker", NODE_TYPES["broker"]
        elif node in surrogate_set:
            role, t = "surrogate", NODE_TYPES["surrogate"]
        else:
            role, t = "edge_device", NODE_TYPES["edge_device"]
        G.nodes[node]["role"] = role
        G.nodes[node]["node_type"] = t

    return G


# ──────────────────────────────────────────────────────────────────────
#  Temporal snapshot generation  (diurnal load model)
# ──────────────────────────────────────────────────────────────────────

def _diurnal_factor(t: int, T: int, phase: float) -> float:
    """
    Smooth day/night load multiplier ∈ [0.2, 1.0] using a sine wave.
    Different nodes get different phase so the network isn't uniform.
    """
    x = 2 * math.pi * (t / max(T, 1)) + phase
    return 0.6 + 0.4 * math.sin(x)


def build_temporal_snapshots(
    G: nx.Graph,
    T: int = 6,
    seed: int = 0,
) -> list[nx.Graph]:
    """
    Produce T time-step snapshots of the same topology.  Connectivity
    is fixed (real topology), but per-node load / capacity-utilisation
    and per-link utilisation evolve with a diurnal pattern + noise.

    Each returned graph carries, per node:
      x = [type_onehot(3), capacity, load, latency_ms]      (6-dim)
    and per edge:
      edge_attr = [bandwidth_norm, delay_ms, reliability, utilisation]  (4-dim)
    plus a target surrogate-quality label.
    """
    rng = random.Random(seed)

    # Static per-node base properties (capacity, base latency, phase)
    base = {}
    for node, d in G.nodes(data=True):
        t = d["node_type"]
        if t == NODE_TYPES["broker"]:
            cap = rng.uniform(0.7, 1.0)
        elif t == NODE_TYPES["surrogate"]:
            cap = rng.uniform(0.4, 1.0)
        else:
            cap = rng.uniform(0.1, 0.7)
        base[node] = {
            "capacity":     round(cap, 3),
            "base_latency": rng.uniform(2, 25),
            "phase":        rng.uniform(0, 2 * math.pi),
        }

    # Static per-edge bandwidth (heavier between brokers)
    edge_bw = {}
    for u, v in G.edges():
        tu, tv = G.nodes[u]["node_type"], G.nodes[v]["node_type"]
        if tu == NODE_TYPES["broker"] and tv == NODE_TYPES["broker"]:
            bw = rng.uniform(0.6, 1.0)       # backbone
        elif NODE_TYPES["edge_device"] in (tu, tv):
            bw = rng.uniform(0.05, 0.4)      # access
        else:
            bw = rng.uniform(0.3, 0.7)       # service
        edge_bw[(u, v)] = round(bw, 3)

    snapshots = []
    for t in range(T):
        H = G.copy()
        for node, d in H.nodes(data=True):
            b = base[node]
            load = _diurnal_factor(t, T, b["phase"]) * rng.uniform(0.7, 1.0)
            load = min(max(load * (1 - b["capacity"] * 0.3), 0.0), 0.98)
            lat  = b["base_latency"] * (1 + 0.5 * load)
            oh = [0.0, 0.0, 0.0]
            oh[d["node_type"]] = 1.0
            d["x"] = oh + [b["capacity"], round(load, 3), round(lat, 2)]

        for u, v, d in H.edges(data=True):
            util = _diurnal_factor(t, T, 0.0) * rng.uniform(0.1, 0.9)
            reliab = rng.uniform(0.85, 1.0)
            d["edge_attr"] = [
                edge_bw[(u, v)],
                round(d.get("prop_delay_ms", 1.0) + rng.uniform(0, 5), 3),
                round(reliab, 3),
                round(min(util, 0.98), 3),
            ]

        _attach_labels(H)
        snapshots.append(H)

    return snapshots


def _attach_labels(G: nx.Graph):
    """
    Surrogate quality label ∈ [0,1] = f(capacity, load, latency, centrality).
    Non-surrogate nodes get -1.  Stored on attribute `y`.
    """
    deg_cent = nx.degree_centrality(G)
    for node, d in G.nodes(data=True):
        if d["node_type"] == NODE_TYPES["surrogate"]:
            cap  = d["x"][3]
            load = d["x"][4]
            lat  = d["x"][5]
            cent = deg_cent[node]
            score = (cap * (1 - load) / (1 + lat / 50.0)) * (0.7 + 0.6 * cent)
            d["y"] = round(min(max(score, 0.0), 1.0), 4)
        else:
            d["y"] = -1.0
