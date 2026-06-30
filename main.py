"""
main_v2.py
==========
Entry point for the IMPROVED surrogate-discovery pipeline:
real topologies + HeteroData + ranking losses + temporal model + full metrics.

Run modes
─────────
  python main_v2.py --mode train     --model spatial   → train spatial hetero GNN
  python main_v2.py --mode train     --model temporal  → train temporal GNN
  python main_v2.py --mode evaluate                     → holdout metrics
  python main_v2.py --mode robustness                   → perturbation robustness
  python main_v2.py --mode compare                      → GNN vs DHT (NDCG, P@k)
  python main_v2.py --mode all       --model temporal
"""

import argparse
import os
import json
import pprint
import numpy as np
import torch
import networkx as nx

from src.hetero_dataset import build_datasets, build_sequence
from src.hetero_model import HeteroSurrogateGNN, TemporalSurrogateGNN
from src.hetero_trainer import HeteroTrainer
from src.metrics import (evaluate_graph, aggregate_metrics,
                         robustness_under_perturbation, precision_at_k, ndcg_at_k)
from src.real_data import (load_topology, assign_roles, build_temporal_snapshots,
                           list_topologies, NODE_TYPES)


CFG = {
    "topo_dir":      "data/topologies",
    "T":             6,
    "max_topologies": 40,
    "hidden_dim":    64,
    "num_layers":    3,
    "heads":         4,
    "dropout":       0.2,
    "epochs":        80,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "w_rank":        1.0,
    "w_reg":         0.3,
    "w_load":        0.3,
    "w_balance":     0.1,
    "early_stop_patience": 20,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "ckpt_dir":      "outputs/checkpoints",
}
os.makedirs("outputs", exist_ok=True)


def _build_model(metadata, temporal: bool):
    Cls = TemporalSurrogateGNN if temporal else HeteroSurrogateGNN
    return Cls(metadata, in_dim=3, hidden_dim=CFG["hidden_dim"],
               num_layers=CFG["num_layers"], heads=CFG["heads"],
               dropout=CFG["dropout"])


def _get_metadata():
    paths = list_topologies(CFG["topo_dir"])
    G = assign_roles(load_topology(paths[0]), seed=0)
    snaps = build_temporal_snapshots(G, T=CFG["T"], seed=0)
    from src.hetero_graph import nx_to_hetero
    return nx_to_hetero(snaps[-1]).metadata()


# ──────────────────────────────────────────────────────────────────────

def mode_train(temporal: bool):
    print(f"\n[TRAIN] model={'temporal' if temporal else 'spatial'}")
    train, val, holdout = build_datasets(
        CFG["topo_dir"], T=CFG["T"], max_topologies=CFG["max_topologies"])
    metadata = train[0][0][-1].metadata()
    model = _build_model(metadata, temporal)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    cfg = dict(CFG); cfg["temporal"] = temporal
    trainer = HeteroTrainer(model, train, val, holdout, cfg)
    history = trainer.train()
    with open("outputs/history_v2.json", "w") as f:
        json.dump(history, f, indent=2)

    trainer.load_best()
    hold_metrics = trainer.evaluate(holdout)
    print("\n  ── HOLDOUT (unseen topologies) ──")
    pprint.pprint({k: round(v, 4) for k, v in hold_metrics.items()})
    with open("outputs/holdout_metrics.json", "w") as f:
        json.dump(hold_metrics, f, indent=2)
    return trainer, holdout


def mode_evaluate(temporal: bool):
    print("\n[EVALUATE]")
    train, val, holdout = build_datasets(
        CFG["topo_dir"], T=CFG["T"], max_topologies=CFG["max_topologies"])
    metadata = train[0][0][-1].metadata()
    model = _build_model(metadata, temporal)
    cfg = dict(CFG); cfg["temporal"] = temporal
    trainer = HeteroTrainer(model, train, val, holdout, cfg)
    try:
        trainer.load_best()
    except FileNotFoundError:
        print("  No checkpoint — train first.")
        return
    metrics = trainer.evaluate(holdout)
    print("\n  Holdout metrics:")
    pprint.pprint({k: round(v, 4) for k, v in metrics.items()})


def mode_robustness(temporal: bool):
    print("\n[ROBUSTNESS] top-k stability under feature noise")
    train, val, holdout = build_datasets(
        CFG["topo_dir"], T=CFG["T"], max_topologies=CFG["max_topologies"])
    metadata = train[0][0][-1].metadata()
    model = _build_model(metadata, temporal)
    cfg = dict(CFG); cfg["temporal"] = temporal
    trainer = HeteroTrainer(model, train, val, holdout, cfg)
    try:
        trainer.load_best()
    except FileNotFoundError:
        print("  No checkpoint — train first.")
        return
    model.eval()
    device = torch.device(CFG["device"])

    def model_fn(seq):
        seq = [g.to(device) for g in seq]
        with torch.no_grad():
            if temporal:
                sl, _, _ = model(seq)
            else:
                sl, _, _ = model(seq[-1])
        return torch.sigmoid(sl), seq[-1]["surrogate"].y

    results = []
    for seq, name in holdout[:5]:
        r = robustness_under_perturbation(model_fn, seq, noise_std=0.05, trials=5)
        r["topology"] = name
        results.append(r)
        print(f"  {name:20s} base P@3={r['base_precision@3']:.3f} "
              f"noisy P@3={r['noisy_precision@3']:.3f}±{r['noisy_precision@3_std']:.3f} "
              f"stability τ={r['ranking_stability_tau']:.3f}")
    return results


# ── DHT baseline on hetero graph (hop-count) ─────────────────────────
def _dht_scores(G: nx.Graph):
    """Return per-surrogate negative mean hop-distance from all devices."""
    devices = [n for n, d in G.nodes(data=True) if d["node_type"] == NODE_TYPES["edge_device"]]
    surrogates = [n for n, d in G.nodes(data=True) if d["node_type"] == NODE_TYPES["surrogate"]]
    scores, labels = [], []
    for s in surrogates:
        hops = []
        for dev in devices:
            try:
                hops.append(nx.shortest_path_length(G, dev, s))
            except nx.NetworkXNoPath:
                hops.append(99)
        scores.append(-np.mean(hops))          # closer = higher score
        labels.append(G.nodes[s]["y"])
    return np.array(scores), np.array(labels)


def mode_compare(temporal: bool):
    print("\n[COMPARE] GNN vs DHT hop-count baseline (holdout topologies)")
    train, val, holdout = build_datasets(
        CFG["topo_dir"], T=CFG["T"], max_topologies=CFG["max_topologies"])
    metadata = train[0][0][-1].metadata()
    model = _build_model(metadata, temporal)
    cfg = dict(CFG); cfg["temporal"] = temporal
    trainer = HeteroTrainer(model, train, val, holdout, cfg)
    try:
        trainer.load_best()
    except FileNotFoundError:
        print("  No checkpoint — train first.")
        return
    model.eval()
    device = torch.device(CFG["device"])

    gnn_m, dht_m = [], []
    paths = list_topologies(CFG["topo_dir"])
    # rebuild the same holdout graphs as nx for DHT
    for seq, name in holdout:
        target = seq[-1].to(device)
        with torch.no_grad():
            if temporal:
                sl, _, _ = model([g.to(device) for g in seq])
            else:
                sl, _, _ = model(target)
        gnn_scores = torch.sigmoid(sl).cpu().numpy()
        gnn_labels = target["surrogate"].y.cpu().numpy()
        gnn_m.append({
            "precision@1": precision_at_k(gnn_scores, gnn_labels, 1),
            "precision@3": precision_at_k(gnn_scores, gnn_labels, 3),
            "ndcg@3":      ndcg_at_k(gnn_scores, gnn_labels, 3),
        })
        # DHT on the matching topology
        match = [p for p in paths if p.endswith(name + ".graphml")]
        if match:
            G = assign_roles(load_topology(match[0]), seed=0)
            snaps = build_temporal_snapshots(G, T=CFG["T"], seed=0)
            ds, dl = _dht_scores(snaps[-1])
            dht_m.append({
                "precision@1": precision_at_k(ds, dl, 1),
                "precision@3": precision_at_k(ds, dl, 3),
                "ndcg@3":      ndcg_at_k(ds, dl, 3),
            })

    gnn_avg = aggregate_metrics(gnn_m)
    dht_avg = aggregate_metrics(dht_m)
    print("\n  Metric        GNN      DHT      Δ")
    for k in ["precision@1", "precision@3", "ndcg@3"]:
        g, d = gnn_avg.get(k, 0), dht_avg.get(k, 0)
        print(f"  {k:12s} {g:.4f}   {d:.4f}   {g-d:+.4f}")
    return gnn_avg, dht_avg


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "evaluate", "robustness",
                                       "compare", "all"], default="all")
    ap.add_argument("--model", choices=["spatial", "temporal"], default="temporal")
    args = ap.parse_args()
    temporal = (args.model == "temporal")

    print("═" * 64)
    print("  Improved Surrogate Discovery — Real Topologies + HeteroGNN")
    print("═" * 64)

    if args.mode in ("train", "all"):
        mode_train(temporal)
    if args.mode in ("evaluate",):
        mode_evaluate(temporal)
    if args.mode in ("compare", "all"):
        mode_compare(temporal)
    if args.mode in ("robustness", "all"):
        mode_robustness(temporal)
    print("\n  Done.")
