# GNN/predict.py

from __future__ import annotations
from typing import Dict
import torch
from torch_geometric.data import HeteroData


@torch.no_grad()
def predict_nodes(
    model,
    hd: HeteroData,
    device: str = "cpu",
    mode: str = "missing",  # "missing", "all", "full"
) -> Dict[str, Dict[str, float]]:
    """
    mode:
      - "missing": return predictions only for nodes with no observed value
      - "all":     return predictions for all nodes
      - "full":    preserve observed values, fill missing with predictions (recommended)
    """
    assert mode in ["missing", "all", "full"]

    model.eval()
    hd = hd.to(device)

    out = model(hd)
    results: Dict[str, Dict[str, float]] = {}

    for nt in hd.node_types:
        node_ids = hd[nt].node_ids
        preds = out[nt].cpu()
        obs_mask = hd[nt].obs_mask.cpu()

        results[nt] = {}
        for i, node_id in enumerate(node_ids):
            if mode == "missing":
                if not obs_mask[i]:
                    results[nt][node_id] = float(preds[i, 0])
            elif mode == "all":
                results[nt][node_id] = float(preds[i, 0])
            else:  # mode == "full"
                if obs_mask[i]:
                    # keep original observed value
                    results[nt][node_id] = float(hd[nt].x_orig[i, 0].cpu())
                else:
                    results[nt][node_id] = float(preds[i, 0])

    return results


@torch.no_grad()
def predict_missing(model, hd: HeteroData, device: str = "cpu"):
    """Backwards-compatible helper: equivalent to predict_nodes(..., mode="missing")."""
    return predict_nodes(model, hd, device=device, mode="missing")