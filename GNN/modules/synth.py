# GNN/synth.py

from __future__ import annotations
from typing import Dict
import random
import numpy as np
from torch_geometric.data import HeteroData


def generate_fake_sample_x(hd: HeteroData, n_per_type: int = 100, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Generate synthetic omics values per node type using existing node_ids in hd.
    """
    random.seed(seed)
    np.random.seed(seed)

    sample_x: Dict[str, Dict[str, float]] = {}

    for ntype in hd.node_types:
        node_ids = hd[ntype].node_ids
        if not node_ids:
            continue

        k = min(n_per_type, len(node_ids))
        chosen_nodes = random.sample(node_ids, k)

        # simple per-type distributions
        if ntype in ["gene", "transcript"]:
            values = np.random.normal(loc=0, scale=2, size=k)
        elif ntype in ["protein", "protein_complex"]:
            values = np.random.normal(loc=1, scale=1.5, size=k)
        elif ntype == "miRNA":
            values = np.random.normal(loc=0, scale=1.5, size=k)
        elif ntype in ["Enhancer", "Promoter", "Promoter/Enhancer"]:
            values = np.random.normal(loc=0, scale=1, size=k)
        else:
            values = np.random.normal(loc=0, scale=1, size=k)

        sample_x[ntype] = {node: float(val) for node, val in zip(chosen_nodes, values)}

    return sample_x