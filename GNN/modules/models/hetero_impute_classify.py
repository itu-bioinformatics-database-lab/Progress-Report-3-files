import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, global_add_pool

from modules.models.heads import SimpleMLPHead

EdgeType = Tuple[str, str, str]


class HeteroImputeClassifyModel(nn.Module):
    """
    Heterogeneous GNN that:
    1. Encodes node features using graph structure
    2. Predicts/imputes node values for each node
    3. Replaces missing values with predicted values
    4. Classifies the graph/sample using the filled graph representation

    Expected node feature format:
        x[:, 0] = scalar value
        x[:, 1] = missing flag (0 = observed, 1 = missing)
    """

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

        # ---------------------------------------------------------
        # Initial encoder from [value, missing_flag] -> hidden state
        # ---------------------------------------------------------
        self.input_proj = nn.ModuleDict({
            node_type: nn.Sequential(
                Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for node_type in self.node_types
        })

        # ---------------------------------------------------------
        # Heterogeneous message passing encoder
        # ---------------------------------------------------------
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: SAGEConv((-1, -1), hidden_dim)
                for edge_type in self.edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr=aggr))

        # ---------------------------------------------------------
        # Node-level decoder:
        # hidden state -> predicted scalar node value
        # ---------------------------------------------------------
        self.node_decoder = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for node_type in self.node_types
        })

        # ---------------------------------------------------------
        # Re-encode filled node features for graph classification
        # ---------------------------------------------------------
        self.filled_input_proj = nn.ModuleDict({
            node_type: nn.Sequential(
                Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for node_type in self.node_types
        })

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

    def encode_node_embeddings(
        self,
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode node features into hidden embeddings using graph structure.
        """
        hidden_dict: Dict[str, torch.Tensor] = {}

        for node_type in self.node_types:
            x = data[node_type].x
            h = self.input_proj[node_type](x)
            hidden_dict[node_type] = h

        for conv in self.convs:
            updated = conv(hidden_dict, data.edge_index_dict)

            new_hidden_dict: Dict[str, torch.Tensor] = {}
            for node_type in self.node_types:
                h = updated.get(node_type, hidden_dict[node_type])
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                new_hidden_dict[node_type] = h

            hidden_dict = new_hidden_dict

        return hidden_dict

    def decode_node_values(
        self,
        hidden_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Predict scalar node values from node embeddings.
        Returns:
            node_type -> [num_nodes]
        """
        predicted_values: Dict[str, torch.Tensor] = {}

        for node_type in self.node_types:
            pred = self.node_decoder[node_type](hidden_dict[node_type]).squeeze(-1)
            predicted_values[node_type] = pred

        return predicted_values

    def fill_missing_values(
        self,
        data: HeteroData,
        predicted_values: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Use observed value if present; otherwise replace with predicted value.

        Returns:
            node_type -> filled feature matrix [num_nodes, 2]
            where col 0 = filled value
                  col 1 = original missing flag
        """
        filled_x_dict: Dict[str, torch.Tensor] = {}

        for node_type in self.node_types:
            x = data[node_type].x
            observed_value = x[:, 0]
            missing_flag = x[:, 1]

            filled_value = torch.where(
                missing_flag == 0,
                observed_value,
                predicted_values[node_type],
            )

            filled_x = torch.stack([filled_value, missing_flag], dim=1)
            filled_x_dict[node_type] = filled_x

        return filled_x_dict

    def build_graph_embedding_from_filled_values(
        self,
        data: HeteroData,
        filled_x_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Build graph-level embedding from filled node values.
        """
        pooled = []

        for node_type in self.node_types:
            h = self.filled_input_proj[node_type](filled_x_dict[node_type])

            batch = getattr(data[node_type], "batch", None)
            pooled.append(self.pool_one_type(h, batch))

        graph_emb = torch.cat(pooled, dim=-1)
        return graph_emb

    def forward(
        self,
        data: HeteroData,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "logits": ...,
                "predicted_values": {node_type: [num_nodes]},
                "filled_x_dict": {node_type: [num_nodes, 2]},
                "node_embeddings": {node_type: [num_nodes, hidden_dim]},
                "graph_embedding": [batch_size, graph_emb_dim],
            }
        """
        node_embeddings = self.encode_node_embeddings(data)
        predicted_values = self.decode_node_values(node_embeddings)
        filled_x_dict = self.fill_missing_values(data, predicted_values)
        graph_embedding = self.build_graph_embedding_from_filled_values(
            data=data,
            filled_x_dict=filled_x_dict,
        )
        logits = self.head(graph_embedding)

        return {
            "logits": logits,
            "predicted_values": predicted_values,
            "filled_x_dict": filled_x_dict,
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
        }