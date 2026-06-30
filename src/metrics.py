"""
metrics.py
==========
Evaluation metrics for surrogate ranking + load prediction.

Ranking metrics (computed over the surrogate set of each graph):
  • precision@k          – fraction of top-k predicted that are truly top-k
  • recall@k             – fraction of truly top-k that are retrieved in top-k
  • NDCG@k               – normalised discounted cumulative gain (graded)
  • Spearman / Kendall   – rank-correlation robustness of the full ordering

Regression metric:
  • load MAE / RMSE      – accuracy of the auxiliary load predictor

All functions take plain numpy / torch 1-D arrays for one graph and return
floats; `aggregate_metrics` averages dicts across many graphs.
"""

import math
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ──────────────────────────────────────────────────────────────────────
#  Ranking metrics
# ──────────────────────────────────────────────────────────────────────

def precision_at_k(pred_scores, true_labels, k: int = 3) -> float:
    pred_scores, true_labels = _to_np(pred_scores), _to_np(true_labels)
    n = len(pred_scores)
    if n == 0:
        return 0.0
    k = min(k, n)
    pred_top = set(np.argsort(-pred_scores)[:k])
    true_top = set(np.argsort(-true_labels)[:k])
    return len(pred_top & true_top) / k


def recall_at_k(pred_scores, true_labels, k: int = 3) -> float:
    pred_scores, true_labels = _to_np(pred_scores), _to_np(true_labels)
    n = len(pred_scores)
    if n == 0:
        return 0.0
    k = min(k, n)
    pred_top = set(np.argsort(-pred_scores)[:k])
    true_top = set(np.argsort(-true_labels)[:k])
    return len(pred_top & true_top) / len(true_top) if true_top else 0.0


def ndcg_at_k(pred_scores, true_labels, k: int = 3) -> float:
    """Graded NDCG using the true quality scores as relevance grades."""
    pred_scores, true_labels = _to_np(pred_scores), _to_np(true_labels)
    n = len(pred_scores)
    if n == 0:
        return 0.0
    k = min(k, n)

    def dcg(order):
        return sum(true_labels[idx] / math.log2(rank + 2)
                   for rank, idx in enumerate(order))

    pred_order  = np.argsort(-pred_scores)[:k]
    ideal_order = np.argsort(-true_labels)[:k]
    idcg = dcg(ideal_order)
    return dcg(pred_order) / idcg if idcg > 0 else 0.0


def rank_correlation(pred_scores, true_labels) -> tuple[float, float]:
    """Return (spearman_rho, kendall_tau) over the full surrogate ordering."""
    pred_scores, true_labels = _to_np(pred_scores), _to_np(true_labels)
    if len(pred_scores) < 3:
        return 0.0, 0.0
    rho, _ = spearmanr(pred_scores, true_labels)
    tau, _ = kendalltau(pred_scores, true_labels)
    return (0.0 if math.isnan(rho) else rho,
            0.0 if math.isnan(tau) else tau)


# ──────────────────────────────────────────────────────────────────────
#  Regression metric
# ──────────────────────────────────────────────────────────────────────

def load_errors(pred_load, true_load) -> tuple[float, float]:
    pred_load, true_load = _to_np(pred_load), _to_np(true_load)
    if len(pred_load) == 0:
        return 0.0, 0.0
    err = pred_load - true_load
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


# ──────────────────────────────────────────────────────────────────────
#  Per-graph + aggregation
# ──────────────────────────────────────────────────────────────────────

def evaluate_graph(sur_scores, sur_labels, load_pred, true_load) -> dict:
    """Compute all metrics for a single graph."""
    rho, tau = rank_correlation(sur_scores, sur_labels)
    mae, rmse = load_errors(load_pred, true_load)
    return {
        "precision@1": precision_at_k(sur_scores, sur_labels, 1),
        "precision@3": precision_at_k(sur_scores, sur_labels, 3),
        "recall@3":    recall_at_k(sur_scores, sur_labels, 3),
        "ndcg@3":      ndcg_at_k(sur_scores, sur_labels, 3),
        "spearman":    rho,
        "kendall":     tau,
        "load_mae":    mae,
        "load_rmse":   rmse,
    }


def aggregate_metrics(per_graph: list[dict]) -> dict:
    """Average a list of per-graph metric dicts."""
    if not per_graph:
        return {}
    keys = per_graph[0].keys()
    return {k: float(np.mean([d[k] for d in per_graph])) for k in keys}


def robustness_under_perturbation(model_fn, snapshots, noise_std: float = 0.05,
                                  trials: int = 5) -> dict:
    """
    Top-k ranking robustness: re-rank surrogates after adding Gaussian noise
    to node features and measure how stable precision@3 and the ordering are.

    model_fn(data) → (sur_scores, sur_labels) for the LAST snapshot.
    Returns mean ± std of precision@3 across perturbation trials.
    """
    base_scores, base_labels = model_fn(snapshots)
    base_p3 = precision_at_k(base_scores, base_labels, 3)

    p3s, taus = [], []
    for _ in range(trials):
        noisy = []
        for g in snapshots:
            g2 = g.clone()
            for nt in g2.node_types:
                if g2[nt].x.numel() > 0:
                    g2[nt].x = g2[nt].x + noise_std * torch.randn_like(g2[nt].x)
            noisy.append(g2)
        ns, nl = model_fn(noisy)
        p3s.append(precision_at_k(ns, nl, 3))
        _, tau = rank_correlation(ns, base_scores)
        taus.append(tau)

    return {
        "base_precision@3":   base_p3,
        "noisy_precision@3":  float(np.mean(p3s)),
        "noisy_precision@3_std": float(np.std(p3s)),
        "ranking_stability_tau": float(np.mean(taus)),
    }
