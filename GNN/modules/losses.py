# GNN/losses.py

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def multitask_loss(
    node_out: Dict[str, torch.Tensor],
    hd,
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    classification_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    mask_dict: Optional[Dict[str, torch.Tensor]] = None,
    target_attr: str = "x",
):
    """
    Multi-task loss:
      - classification loss on graph label
      - reconstruction loss on node values

    Parameters
    ----------
    node_out:
        dict[node_type] -> predicted node values
    hd:
        HeteroData with true node values stored in hd[nt][target_attr]
    logits:
        graph logits [batch_size, num_classes]
    y:
        graph labels [batch_size]
    mask_dict:
        optional dict[node_type] -> bool mask of nodes to reconstruct.
        If None, reconstruct all nodes.
    target_attr:
        usually 'x'

    Returns
    -------
    total_loss, loss_cls, loss_rec
    """
    loss_cls = F.cross_entropy(logits, y)

    rec_losses = []
    for nt, pred in node_out.items():
        target = hd[nt][target_attr]

        if mask_dict is not None and nt in mask_dict:
            mask = mask_dict[nt]
            if mask.sum() == 0:
                continue
            rec_losses.append(F.mse_loss(pred[mask], target[mask]))
        else:
            rec_losses.append(F.mse_loss(pred, target))

    if len(rec_losses) == 0:
        loss_rec = torch.tensor(0.0, device=logits.device)
    else:
        loss_rec = torch.stack(rec_losses).mean()

    total = classification_weight * loss_cls + reconstruction_weight * loss_rec
    return total, loss_cls, loss_rec