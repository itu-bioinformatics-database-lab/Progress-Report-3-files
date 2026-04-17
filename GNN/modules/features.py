# GNN/features.py

from __future__ import annotations
from typing import Dict, Any
import torch
from torch_geometric.data import HeteroData


def attach_sample_values(
    hd: HeteroData,
    sample_x: Dict[str, Dict[str, float]],
    map_gene_to_transcript: bool = True,
    map_protein_to_complex: bool = True,
    transcript_suffix: str = "_transcript",
    protein_suffix: str = "_protein",
    keep_x_orig: bool = True,
) -> HeteroData:
    """
    Attaches:
      - hd[nt].x: shape [num_nodes, 1]
      - hd[nt].obs_mask: shape [num_nodes] bool
      - hd[nt].x_orig: clone of x (if keep_x_orig)
    Then applies mapping rules:
      - gene -> transcript (strip transcript_suffix)
      - protein -> protein_complex (strip protein_suffix)
    """

    def num_nodes_of(ntype: str) -> int:
        return len(hd[ntype].node_ids)

    def build_type_features(ntype: str):
        n = int(num_nodes_of(ntype))
        x = torch.zeros((n, 1), dtype=torch.float32)
        obs = torch.zeros((n,), dtype=torch.bool)

        node_ids = hd[ntype].node_ids
        val_dict = sample_x.get(ntype, {}) or {}

        for i, vid in enumerate(node_ids):
            if vid in val_dict:
                x[i, 0] = float(val_dict[vid])
                obs[i] = True
        return x, obs

    # base attach
    for ntype in hd.node_types:
        x_t, obs_t = build_type_features(ntype)
        hd[ntype].x = x_t
        hd[ntype].obs_mask = obs_t

    # mapping: gene -> transcript
    if map_gene_to_transcript and ("gene" in sample_x) and ("transcript" in hd.node_types):
        gene_vals = sample_x["gene"] or {}
        node_ids = hd["transcript"].node_ids
        for i, vid in enumerate(node_ids):
            base = vid.replace(transcript_suffix, "")
            if base in gene_vals:
                hd["transcript"].x[i, 0] = float(gene_vals[base])
                hd["transcript"].obs_mask[i] = True

    # mapping: protein -> protein_complex
    if map_protein_to_complex and ("protein" in sample_x) and ("protein_complex" in hd.node_types):
        prot_vals = sample_x["protein"] or {}
        node_ids = hd["protein_complex"].node_ids
        for i, vid in enumerate(node_ids):
            base = vid.replace(protein_suffix, "")
            if base in prot_vals:
                hd["protein_complex"].x[i, 0] = float(prot_vals[base])
                hd["protein_complex"].obs_mask[i] = True

    if keep_x_orig:
        for ntype in hd.node_types:
            hd[ntype].x_orig = hd[ntype].x.clone()

    return hd