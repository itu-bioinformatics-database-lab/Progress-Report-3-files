# GNN/model.py

from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HeteroImputer(nn.Module):
    """
    HGT-based scalar imputer: input x is [N, 1] per node type, output is [N, 1].
    """
    def __init__(self, metadata, hidden: int = 64, num_layers: int = 2, heads: int = 2):
        super().__init__()
        node_types, edge_types = metadata

        self.in_proj = nn.ModuleDict({nt: Linear(1, hidden) for nt in node_types})
        self.convs = nn.ModuleList(
            [HGTConv(hidden, hidden, metadata, heads=heads) for _ in range(num_layers)]
        )
        self.decoders = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            for nt in node_types
        })

    def forward(self, hd):
        x_dict = {nt: self.in_proj[nt](hd[nt].x) for nt in hd.node_types}

        for conv in self.convs:
            x_dict = conv(x_dict, hd.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        out = {nt: self.decoders[nt](x_dict[nt]) for nt in hd.node_types}
        return out