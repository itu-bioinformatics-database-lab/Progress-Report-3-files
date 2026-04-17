# GNN/train.py

from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


def train_imputer_one_sample(
    model,
    hd: HeteroData,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mask_ratio: float = 0.2,
    min_mask_per_type: int = 5,
    device: str = "cpu",
    log_first_n_epochs: int = 25,
) -> torch.nn.Module:
    """
    Self-supervised masking on observed values:
      - each epoch: restore x from x_orig
      - randomly mask a portion of observed nodes per type
      - predict and compute MSE on masked positions only
    Requires:
      - hd[nt].x_orig exists
      - hd[nt].obs_mask exists
      - hd[nt].x exists
    """
    model = model.to(device)
    hd = hd.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        # restore unmasked inputs each epoch
        for nt in hd.node_types:
            hd[nt].x = hd[nt].x_orig.clone()

        train_masks: Dict[str, torch.Tensor] = {}
        for nt in hd.node_types:
            obs = hd[nt].obs_mask
            idx = obs.nonzero(as_tuple=False).view(-1)

            m = torch.zeros(hd[nt].num_nodes, dtype=torch.bool, device=device)
            if idx.numel() >= 1:
                k = int(mask_ratio * idx.numel())
                k = max(1, k)

                if idx.numel() >= min_mask_per_type:
                    k = max(k, min_mask_per_type)
                    k = min(k, idx.numel())

                chosen = idx[torch.randperm(idx.numel(), device=device)[:k]]
                m[chosen] = True

                # mask those inputs
                hd[nt].x[m] = 0.0

            train_masks[nt] = m

        pred = model(hd)

        loss = 0.0
        denom = 0
        for nt in hd.node_types:
            m = train_masks[nt]
            if m.any():
                loss = loss + F.mse_loss(pred[nt][m], hd[nt].x_orig[m])
                denom += 1

        if denom == 0:
            # nothing to train on (no observed values anywhere)
            break

        loss.backward()
        opt.step()

        if epoch <= log_first_n_epochs:
            print(f"Epoch {epoch:04d} | loss={loss.item():.6f}")

    return model