import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any
from torch_geometric.data import HeteroData

def compute_multitask_loss(
    model_output: Dict[str, Any],
    batch: HeteroData,
    classification_criterion: nn.Module,
    reconstruction_weight: float = 0.5,
    reconstruct_on_observed_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Computes:
    - classification loss
    - reconstruction loss
    - total loss

    Notes
    -----
    Since true values for genuinely missing nodes are unknown,
    reconstruction is usually computed on observed nodes.
    A stronger version is to artificially mask some observed nodes during training.
    """
    logits = model_output["logits"]
    predicted_values = model_output["predicted_values"]

    y = batch.y.view(-1)
    classification_loss = classification_criterion(logits, y)

    reconstruction_loss = torch.tensor(
        0.0,
        device=logits.device,
        dtype=logits.dtype,
    )

    n_terms = 0

    for node_type, pred in predicted_values.items():
        x = batch[node_type].x
        target_value = x[:, 0]
        missing_flag = x[:, 1]

        if reconstruct_on_observed_only:
            mask = (missing_flag == 0)
        else:
            mask = torch.ones_like(missing_flag, dtype=torch.bool)

        if mask.sum() > 0:
            reconstruction_loss = reconstruction_loss + F.mse_loss(
                pred[mask],
                target_value[mask],
            )
            n_terms += 1

    if n_terms > 0:
        reconstruction_loss = reconstruction_loss / n_terms

    total_loss = classification_loss + reconstruction_weight * reconstruction_loss

    return {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "reconstruction_loss": reconstruction_loss,
    }