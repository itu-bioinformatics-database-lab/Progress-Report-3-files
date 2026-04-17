# GNN/multitask_model.py

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear, global_mean_pool


class HeteroMultiTaskGNN(nn.Module):
    """
    Shared heterogeneous GNN for:
      1) node-value imputation
      2) graph/sample classification

    Assumptions
    -----------
    - Each node type has scalar input x of shape [N_type, 1]
      (or you can change input_dim if needed).
    - Graph structure is shared, sample-specific values are attached to nodes.
    - Batch object is a PyG HeteroData or batched HeteroData.
    - If batch vectors exist per node type, graph classification supports mini-batching.
      Otherwise it falls back to single-graph pooling.

    Outputs
    -------
    node_out:
        dict[node_type] -> tensor [N_type, out_dim]
    logits:
        tensor [batch_size, num_classes]
    """

    def __init__(
        self,
        metadata,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        decoder_hidden_dim: Optional[int] = None,
        out_dim: int = 1,
        pooling: str = "mean",
    ):
        super().__init__()

        node_types, _ = metadata
        self.node_types = list(node_types)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.pooling = pooling

        if decoder_hidden_dim is None:
            decoder_hidden_dim = hidden_dim

        # Input projection per node type
        self.in_proj = nn.ModuleDict({
            nt: Linear(input_dim, hidden_dim)
            for nt in self.node_types
        })

        # Shared hetero encoder
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads,
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim)
            for nt in self.node_types
        })

        # Node decoders for imputation / reconstruction
        self.decoders = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(hidden_dim, decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(decoder_hidden_dim, out_dim),
            )
            for nt in self.node_types
        })

        # Graph/sample classifier head
        # We pool each node type separately, then combine them.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(self.node_types), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, hd) -> Dict[str, torch.Tensor]:
        """
        Encode node features for all node types.
        """
        x_dict = {}
        for nt in hd.node_types:
            if "x" not in hd[nt]:
                raise ValueError(f"Node type '{nt}' is missing .x")
            x = hd[nt].x
            x_dict[nt] = self.in_proj[nt](x)

        for conv in self.convs:
            x_dict = conv(x_dict, hd.edge_index_dict)
            x_dict = {
                nt: self.norms[nt](F.relu(x))
                for nt, x in x_dict.items()
            }
            x_dict = {
                nt: F.dropout(x, p=self.dropout, training=self.training)
                for nt, x in x_dict.items()
            }

        return x_dict

    def decode_nodes(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Node-level outputs for imputation.
        """
        return {
            nt: self.decoders[nt](x_dict[nt])
            for nt in x_dict
        }

    def pool_graph(self, hd, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Pool node embeddings into graph/sample embeddings.

        Supports:
        - batched HeteroData with hd[nt].batch
        - single sample graph without batch vectors
        """
        pooled_per_type = []

        for nt in self.node_types:
            x = x_dict[nt]

            if hasattr(hd[nt], "batch") and hd[nt].batch is not None:
                # Batched graphs
                pooled = global_mean_pool(x, hd[nt].batch)
            else:
                # Single graph fallback -> shape [1, hidden_dim]
                if self.pooling == "mean":
                    pooled = x.mean(dim=0, keepdim=True)
                elif self.pooling == "sum":
                    pooled = x.sum(dim=0, keepdim=True)
                elif self.pooling == "max":
                    pooled = x.max(dim=0, keepdim=True).values
                else:
                    raise ValueError(f"Unsupported pooling='{self.pooling}'")

            pooled_per_type.append(pooled)

        # Concatenate pooled node-type embeddings
        graph_emb = torch.cat(pooled_per_type, dim=-1)
        return graph_emb

    def classify(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Graph/sample-level logits.
        """
        return self.classifier(graph_emb)

    def forward(self, hd):
        """
        Returns
        -------
        node_out : dict[node_type] -> [N_type, out_dim]
        logits   : [batch_size, num_classes]
        """
        x_dict = self.encode(hd)
        node_out = self.decode_nodes(x_dict)
        graph_emb = self.pool_graph(hd, x_dict)
        logits = self.classify(graph_emb)
        return node_out, logits