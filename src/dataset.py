"""
dataset.py
==========
Generates a synthetic dataset of edge-network graphs for training
and evaluation.  Each graph is an independent network snapshot.
"""

import torch
from torch_geometric.data import Dataset, Data
from torch.utils.data import random_split
from src.graph_builder import (
    generate_edge_network,
    networkx_to_pyg,
    generate_surrogate_labels,
)


class EdgeNetworkDataset(Dataset):
    """
    In-memory dataset of randomly generated edge-network graphs.

    Each item is a PyG Data object enriched with:
      data.y          – surrogate quality labels  (-1 for non-surrogates)
      data.true_load  – ground-truth node load (from node features)
    """

    def __init__(
        self,
        num_graphs: int = 500,
        n_edge_devices: int = 10,
        n_brokers: int = 4,
        n_surrogates: int = 6,
        base_seed: int = 0,
    ):
        super().__init__()
        self.graphs: list[Data] = []

        for i in range(num_graphs):
            G = generate_edge_network(
                n_edge_devices=n_edge_devices,
                n_brokers=n_brokers,
                n_surrogates=n_surrogates,
                seed=base_seed + i,
            )
            data = networkx_to_pyg(G)
            data.y          = generate_surrogate_labels(data)
            data.true_load  = data.x[:, 4].clone()   # load column
            self.graphs.append(data)

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]


def build_splits(
    num_graphs: int = 500,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """Return train / val / test Dataset splits."""
    dataset = EdgeNetworkDataset(num_graphs=num_graphs)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)
