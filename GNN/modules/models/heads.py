import torch
import torch.nn as nn
from typing import List, Optional


class SimpleMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dims = hidden_dims or [128, 64]

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.block1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.block2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.output_layer(h)


class GatedHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = x * self.gate(x)
        return self.classifier(x)


class TypeAwareHead(nn.Module):
    """
    Learns weights for each node-type block in the concatenated embedding.
    Useful for multi-omics imbalance.
    """

    def __init__(
        self,
        num_types: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_types = num_types
        self.hidden_dim = hidden_dim

        self.type_weights = nn.Parameter(torch.ones(num_types))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_types, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        B = x.size(0)

        x = x.view(B, self.num_types, self.hidden_dim)

        weights = torch.softmax(self.type_weights, dim=0)
        x = x * weights.view(1, -1, 1)

        x = x.view(B, -1)
        return self.classifier(x)

def build_head(
    head_type: str,
    input_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_types: int,
    dropout: float = 0.1,
):
    head_type = head_type.lower()

    if head_type == "simple":
        return SimpleMLPHead(
            input_dim=input_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    elif head_type == "deep":
        return DeepMLPHead(
            input_dim=input_dim,
            out_dim=out_dim,
            hidden_dims=[128, 64],
            dropout=dropout,
        )

    elif head_type == "residual":
        return ResidualMLPHead(
            input_dim=input_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    elif head_type == "gated":
        return GatedHead(
            input_dim=input_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    elif head_type == "type_aware":
        return TypeAwareHead(
            num_types=num_types,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown head type: {head_type}")