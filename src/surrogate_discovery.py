"""
surrogate_discovery.py
======================
Inference-time surrogate discovery:  given a trained GNN and a live
network graph, find the *best* surrogates for each requesting edge device.

This replaces DHT-based routing with a learned, graph-aware ranking.

Key methods
───────────
  discover(data, top_k)         → ranked list of surrogates per device
  explain_ranking(data, dev_id) → human-readable feature breakdown
  simulate_dht_baseline(G)      → simple DHT hop-count baseline for comparison
"""

import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data

from src.graph_builder import (
    generate_edge_network,
    networkx_to_pyg,
    generate_surrogate_labels,
    NODE_TYPES,
)
from src.gnn_model import SurrogateDiscoveryGNN


# ──────────────────────────────────────────────────────────────────────

class SurrogateDiscovery:
    """
    High-level discovery interface.

    Parameters
    ──────────
    model   : trained SurrogateDiscoveryGNN
    device  : torch device
    """

    def __init__(self, model: SurrogateDiscoveryGNN, device: torch.device = None):
        self.model  = model
        self.device = device or torch.device("cpu")
        self.model.eval()

    # ── Core discovery ────────────────────────────────────────────────

    @torch.no_grad()
    def discover(
        self,
        data: Data,
        top_k: int = 3,
    ) -> dict[int, list[dict]]:
        """
        For each edge-device node, return the top-k surrogate candidates
        ranked by GNN-predicted quality score.

        Returns
        ───────
        {device_node_id: [{"surrogate": int, "score": float, "features": dict}, ...]}
        """
        data = data.to(self.device)
        sur_scores, load_preds, embeddings = self.model(data)

        device_ids   = (data.node_type == NODE_TYPES["edge_device"]).nonzero(as_tuple=True)[0]
        surrogate_ids = (data.node_type == NODE_TYPES["surrogate"]).nonzero(as_tuple=True)[0]

        results = {}
        for dev_idx in device_ids.tolist():
            ranked = []
            for sur_idx in surrogate_ids.tolist():
                score = sur_scores[sur_idx].item()
                feat  = {
                    "capacity":    round(data.x[sur_idx, 3].item(), 3),
                    "load":        round(data.x[sur_idx, 4].item(), 3),
                    "latency_ms":  round(data.x[sur_idx, 5].item(), 2),
                    "pred_load":   round(load_preds[sur_idx].item(), 3),
                }
                ranked.append({"surrogate": sur_idx, "score": score, "features": feat})

            ranked.sort(key=lambda d: d["score"], reverse=True)
            results[dev_idx] = ranked[:top_k]

        return results

    # ── Embedding similarity search ───────────────────────────────────

    @torch.no_grad()
    def embedding_similarity_search(
        self,
        data: Data,
        query_node: int,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Find the most similar surrogates to `query_node` in embedding space
        (cosine similarity).  Useful when you don't know the query node type.
        """
        data = data.to(self.device)
        _, _, embeddings = self.model(data)

        q_emb = embeddings[query_node]           # [H]
        sur_ids = (data.node_type == NODE_TYPES["surrogate"]).nonzero(as_tuple=True)[0]

        sims = []
        for s in sur_ids.tolist():
            cos_sim = torch.nn.functional.cosine_similarity(
                q_emb.unsqueeze(0), embeddings[s].unsqueeze(0)
            ).item()
            sims.append({"surrogate": s, "similarity": round(cos_sim, 4)})

        sims.sort(key=lambda d: d["similarity"], reverse=True)
        return sims[:top_k]


# ──────────────────────────────────────────────────────────────────────
#  DHT Baseline
# ──────────────────────────────────────────────────────────────────────

def simulate_dht_baseline(G: nx.Graph, top_k: int = 3) -> dict[int, list[dict]]:
    """
    Naïve DHT-style routing: rank surrogates by shortest-hop-count from each
    edge device. This is the baseline we are trying to beat with the GNN.
    """
    device_ids   = [n for n, d in G.nodes(data=True) if d["node_type"] == NODE_TYPES["edge_device"]]
    surrogate_ids = [n for n, d in G.nodes(data=True) if d["node_type"] == NODE_TYPES["surrogate"]]

    results = {}
    for dev in device_ids:
        lengths = nx.single_source_shortest_path_length(G, dev)
        ranked  = []
        for s in surrogate_ids:
            hops = lengths.get(s, 999)
            ranked.append({"surrogate": s, "hops": hops})
        ranked.sort(key=lambda d: d["hops"])
        results[dev] = ranked[:top_k]

    return results


# ──────────────────────────────────────────────────────────────────────
#  Comparison Utility
# ──────────────────────────────────────────────────────────────────────

def compare_gnn_vs_dht(
    gnn_results: dict,
    dht_results: dict,
    ground_truth_labels: torch.Tensor,
) -> dict:
    """
    Compute precision@k: fraction of top-k predictions that match
    the top-k ground-truth surrogates (by label score).

    Returns precision@k for both GNN and DHT.
    """
    top_k = len(next(iter(gnn_results.values())))
    surrogate_ids = [i for i, l in enumerate(ground_truth_labels.tolist()) if l >= 0]
    gt_top_k = sorted(surrogate_ids,
                      key=lambda i: ground_truth_labels[i].item(),
                      reverse=True)[:top_k]
    gt_set = set(gt_top_k)

    gnn_prec, dht_prec = [], []
    for dev in gnn_results:
        gnn_top = set(d["surrogate"] for d in gnn_results[dev])
        gnn_prec.append(len(gnn_top & gt_set) / top_k)

    for dev in dht_results:
        dht_top = set(d["surrogate"] for d in dht_results[dev])
        dht_prec.append(len(dht_top & gt_set) / top_k)

    return {
        "gnn_precision_at_k":  round(np.mean(gnn_prec), 4),
        "dht_precision_at_k":  round(np.mean(dht_prec), 4),
        "top_k": top_k,
    }
