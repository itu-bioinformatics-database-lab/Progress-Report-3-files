import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, global_mean_pool, global_add_pool
from modules.models.heads import SimpleMLPHead

EdgeType = Tuple[str, str, str]


class MissingAwareHeteroClassifier(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[EdgeType]],
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggr: str = "sum",
        head: Optional[nn.Module] = None,
    ):
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.input_proj = nn.ModuleDict({
            nt: Linear(2, hidden_dim)
            for nt in self.node_types
        })

        self.missing_type_embedding = nn.ParameterDict({
            nt: nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
            for nt in self.node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                et: SAGEConv((-1, -1), hidden_dim)
                for et in self.edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr=aggr))

        graph_emb_dim = hidden_dim * len(self.node_types)

        self.head = head if head is not None else SimpleMLPHead(
            input_dim=graph_emb_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

    def pool_one_type(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            return x.sum(dim=0, keepdim=True)
        return global_add_pool(x, batch)

    def encode(self, data: HeteroData) -> torch.Tensor:
        x_dict: Dict[str, torch.Tensor] = {}

        for nt in self.node_types:
            x = data[nt].x
            h = self.input_proj[nt](x)

            missing_mask = x[:, 1:2]
            h = h + missing_mask * self.missing_type_embedding[nt]

            h = F.relu(h)
            x_dict[nt] = h

        for conv in self.convs:
            updated = conv(x_dict, data.edge_index_dict)

            new_x_dict: Dict[str, torch.Tensor] = {}
            for nt in self.node_types:
                h = updated.get(nt, x_dict[nt])
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                new_x_dict[nt] = h
            x_dict = new_x_dict

        pooled = []
        for nt in self.node_types:
            batch = getattr(data[nt], "batch", None)
            pooled.append(self.pool_one_type(x_dict[nt], batch))

        graph_emb = torch.cat(pooled, dim=-1)
        return graph_emb

    def forward(self, data: HeteroData) -> torch.Tensor:
        graph_emb = self.encode(data)
        logits = self.head(graph_emb)
        return logits