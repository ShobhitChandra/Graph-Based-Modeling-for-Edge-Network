"""
hetero_dataset.py
=================
Builds the training/validation/holdout datasets from REAL topologies.

Crucially, the split is by TOPOLOGY, not by graph snapshot:
  • train / val topologies and the holdout-test topologies are DISJOINT
    networks (e.g. train on Aarnet/Abilene…, test on completely unseen
    operators).  This is the proper "validate on a holdout set with
    different network parameters" generalisation test.

Each dataset item is a *temporal sequence* (list[HeteroData]) for one
topology — the temporal model consumes the whole sequence; the spatial
model just uses the last snapshot.
"""

import random
from src.real_data import (
    list_topologies, load_topology, assign_roles, build_temporal_snapshots,
)
from src.hetero_graph import nx_to_hetero


def build_sequence(path: str, T: int = 6, seed: int = 0):
    """Return (list[HeteroData], topology_name) for one topology file."""
    G = load_topology(path)
    G = assign_roles(G, seed=seed)
    snaps = build_temporal_snapshots(G, T=T, seed=seed)
    hetero_seq = [nx_to_hetero(s) for s in snaps]
    name = path.split("/")[-1].replace(".graphml", "")
    return hetero_seq, name


def build_datasets(
    topo_dir: str = "data/topologies",
    T: int = 6,
    holdout_frac: float = 0.25,
    val_frac: float = 0.15,
    seed: int = 42,
    max_topologies: int = None,
):
    """
    Returns (train, val, holdout) where each is a list of
    (sequence, name) tuples.  Topologies are partitioned disjointly.
    """
    paths = list_topologies(topo_dir)
    if max_topologies:
        paths = paths[:max_topologies]

    rng = random.Random(seed)
    rng.shuffle(paths)

    n = len(paths)
    n_hold = max(1, int(n * holdout_frac))
    n_val  = max(1, int(n * val_frac))

    holdout_paths = paths[:n_hold]
    val_paths     = paths[n_hold:n_hold + n_val]
    train_paths   = paths[n_hold + n_val:]

    def make(plist):
        out = []
        for i, p in enumerate(plist):
            try:
                seq, name = build_sequence(p, T=T, seed=seed + i)
                # skip degenerate graphs with no surrogates or no edges
                if seq[-1]["surrogate"].x.shape[0] >= 2:
                    out.append((seq, name))
            except Exception as e:
                print(f"  [skip] {p}: {e}")
        return out

    train   = make(train_paths)
    val     = make(val_paths)
    holdout = make(holdout_paths)

    print(f"  Topologies → train={len(train)}  val={len(val)}  holdout={len(holdout)}")
    return train, val, holdout
