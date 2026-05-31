"""
main.py
=======
Entry point for the GNN-based Surrogate Discovery pipeline.

Run modes
─────────
  python main.py --mode train      → train the GNN from scratch
  python main.py --mode evaluate   → load best checkpoint, run test set
  python main.py --mode discover   → demo surrogate discovery on 1 graph
  python main.py --mode compare    → GNN vs DHT baseline comparison
  python main.py --mode visualize  → generate all plots
  python main.py --mode all        → run everything in sequence
"""

import argparse
import os
import json
import pprint
import torch
import numpy as np
import networkx as nx

from src.graph_builder import (
    generate_edge_network,
    networkx_to_pyg,
    generate_surrogate_labels,
)
from src.dataset         import build_splits
from src.gnn_model       import SurrogateDiscoveryGNN
from src.trainer         import Trainer
from src.surrogate_discovery import (
    SurrogateDiscovery,
    simulate_dht_baseline,
    compare_gnn_vs_dht,
)
from src.visualize import (
    draw_network,
    plot_training,
    plot_score_distribution,
    plot_embeddings,
)

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

CFG = {
    # Data
    "num_graphs":        500,
    "n_edge_devices":    10,
    "n_brokers":         4,
    "n_surrogates":      6,
    # Model
    "node_in_dim":       6,
    "edge_in_dim":       4,
    "hidden_dim":        64,
    "num_layers":        3,
    "heads":             4,
    "dropout":           0.2,
    # Training
    "epochs":            100,
    "lr":                1e-3,
    "weight_decay":      1e-4,
    "batch_size":        16,
    "lambda1":           1.0,
    "lambda2":           0.3,
    "early_stop_patience": 15,
    # I/O
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
    "log_dir":           "outputs/runs",
    "ckpt_dir":          "outputs/checkpoints",
    "plot_dir":          "outputs/plots",
}

os.makedirs(CFG["plot_dir"], exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

def build_model() -> SurrogateDiscoveryGNN:
    return SurrogateDiscoveryGNN(
        node_in_dim = CFG["node_in_dim"],
        edge_in_dim = CFG["edge_in_dim"],
        hidden_dim  = CFG["hidden_dim"],
        num_layers  = CFG["num_layers"],
        heads       = CFG["heads"],
        dropout     = CFG["dropout"],
    )


def demo_graph():
    G    = generate_edge_network(seed=999)
    data = networkx_to_pyg(G)
    data.y         = generate_surrogate_labels(data)
    data.true_load = data.x[:, 4].clone()
    return G, data


# ──────────────────────────────────────────────────────────────────────
#  Modes
# ──────────────────────────────────────────────────────────────────────

def mode_train():
    print("\n[MODE] TRAIN")
    train_ds, val_ds, test_ds = build_splits(num_graphs=CFG["num_graphs"])

    model   = build_model()
    trainer = Trainer(model, train_ds, val_ds, test_ds, CFG)
    history = trainer.train()

    # Save history for later plotting
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Quick test evaluation
    trainer.load_best()
    test_metrics = trainer.evaluate(trainer.test_loader)
    print("\n  Test metrics:")
    pprint.pprint(test_metrics)
    return history


def mode_evaluate():
    print("\n[MODE] EVALUATE")
    _, _, test_ds = build_splits(num_graphs=CFG["num_graphs"])

    model   = build_model()
    trainer = Trainer(model, test_ds, test_ds, test_ds, CFG)
    trainer.load_best()
    metrics = trainer.evaluate(trainer.test_loader)
    print("\n  Test metrics:")
    pprint.pprint(metrics)
    return metrics


def mode_discover():
    print("\n[MODE] DISCOVER  (single-graph demo)")
    G, data = demo_graph()

    model = build_model()
    ckpt  = os.path.join(CFG["ckpt_dir"], "best_model.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=CFG["device"])
        model.load_state_dict(state["model_state"])
        print(f"  Loaded checkpoint (ep {state['epoch']})")
    else:
        print("  No checkpoint found – using random weights (demo mode)")

    disc    = SurrogateDiscovery(model, device=torch.device(CFG["device"]))
    results = disc.discover(data, top_k=3)

    print("\n  ── GNN Surrogate Rankings ──────────────────────────────")
    for dev, surrogates in list(results.items())[:3]:
        print(f"\n  Edge Device {dev}:")
        for rank, s in enumerate(surrogates, 1):
            print(f"    #{rank}  Surrogate {s['surrogate']:>3d}  "
                  f"score={s['score']:.4f}  "
                  f"cap={s['features']['capacity']}  "
                  f"load={s['features']['load']}  "
                  f"lat={s['features']['latency_ms']}ms")
    return results


def mode_compare():
    print("\n[MODE] COMPARE  GNN vs DHT baseline")
    G, data = demo_graph()

    model = build_model()
    ckpt  = os.path.join(CFG["ckpt_dir"], "best_model.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=CFG["device"])
        model.load_state_dict(state["model_state"])

    disc     = SurrogateDiscovery(model, device=torch.device(CFG["device"]))
    gnn_res  = disc.discover(data, top_k=3)
    dht_res  = simulate_dht_baseline(G, top_k=3)
    comp     = compare_gnn_vs_dht(gnn_res, dht_res, data.y)

    print(f"\n  Precision@{comp['top_k']}:")
    print(f"    GNN  : {comp['gnn_precision_at_k']:.4f}")
    print(f"    DHT  : {comp['dht_precision_at_k']:.4f}")
    print(f"    Δ    : {comp['gnn_precision_at_k'] - comp['dht_precision_at_k']:+.4f}")
    return comp


def mode_visualize():
    print("\n[MODE] VISUALIZE")
    G, data = demo_graph()

    # 1. Network graph
    draw_network(G, save_path=os.path.join(CFG["plot_dir"], "network_graph.png"))
    print("  Saved: network_graph.png")

    # 2. Training curves (need saved history)
    hist_path = "outputs/history.json"
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        plot_training(history, save_path=os.path.join(CFG["plot_dir"], "training_curves.png"))
        print("  Saved: training_curves.png")
    else:
        print("  Skipping training curves (run --mode train first)")

    # 3. Score distribution
    model = build_model()
    ckpt  = os.path.join(CFG["ckpt_dir"], "best_model.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=CFG["device"])
        model.load_state_dict(state["model_state"])

    model.eval()
    device = torch.device(CFG["device"])
    data   = data.to(device)
    model  = model.to(device)

    with torch.no_grad():
        scores, _, emb = model(data)

    sur_mask = (data.node_type == 2)
    pred = scores[sur_mask].cpu().numpy()
    true = data.y[sur_mask].cpu().numpy()

    plot_score_distribution(pred, true,
        save_path=os.path.join(CFG["plot_dir"], "score_dist.png"))
    print("  Saved: score_dist.png")

    # 4. t-SNE embeddings
    plot_embeddings(emb.cpu(), data.node_type.cpu(),
        save_path=os.path.join(CFG["plot_dir"], "tsne_embeddings.png"))
    print("  Saved: tsne_embeddings.png")


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN-based Surrogate Discovery in Edge Networks"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "discover", "compare", "visualize", "all"],
        default="all",
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("   Graph-Based Surrogate Discovery in Edge Networks")
    print("   RouteNet-Inspired GNN  •  PyTorch Geometric")
    print("═" * 60)
    print(f"   Device : {CFG['device']}")

    if args.mode in ("train", "all"):
        history = mode_train()

    if args.mode in ("evaluate", "all"):
        mode_evaluate()

    if args.mode in ("discover", "all"):
        mode_discover()

    if args.mode in ("compare", "all"):
        mode_compare()

    if args.mode in ("visualize", "all"):
        mode_visualize()

    print("\n  Done. Outputs saved to outputs/")
