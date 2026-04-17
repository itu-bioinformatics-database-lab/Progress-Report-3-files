import copy
import numpy as np
from typing import Any, Dict, List, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


def build_label_encoder(store: Dict[str, Dict[str, Any]]) -> Dict[Any, int]:
    raw_labels = sorted({sample_obj["label"] for sample_obj in store.values()})
    return {lab: i for i, lab in enumerate(raw_labels)}


def infer_num_classes(dataset: List[HeteroData]) -> int:
    labels = []
    for g in dataset:
        labels.extend(g.y.view(-1).detach().cpu().tolist())
    return int(max(labels)) + 1


def split_dataset(
    dataset: List[HeteroData],
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[List[HeteroData], List[HeteroData]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dataset))
    rng.shuffle(idx)

    split = int(len(dataset) * train_frac)
    train_idx = idx[:split]
    test_idx = idx[split:]

    train_dataset = [copy.deepcopy(dataset[i]) for i in train_idx]
    test_dataset = [copy.deepcopy(dataset[i]) for i in test_idx]
    return train_dataset, test_dataset


def make_loaders(
    train_dataset: List[HeteroData],
    test_dataset: List[HeteroData],
    batch_size: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader