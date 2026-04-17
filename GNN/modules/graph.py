# modules/graph.py
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import HeteroData



EdgeKey = Tuple[str, str, str]


def build_heterodata_from_json(
    graph_json: Dict[str, Any],
    exclude_types: Set[str] | None = None,
    add_reverse: bool = True,
    default_relation: str = "interacts",
    rev_prefix: str = "rev_",
) -> HeteroData:
    """
    Build PyG HeteroData from a universal graph JSON dict.

    Parameters are notebook-configurable via function args.
    """
    if exclude_types is None:
        exclude_types = {"R"}

    vertices = graph_json["vertices"]
    edges = graph_json["edges"]

    # --- collect node ids by type (excluding excluded types) ---
    node_ids_by_type: Dict[str, list] = {}
    for vid, v in vertices.items():
        ntype = v.get("omic_type", "unknown")
        if ntype in exclude_types:
            continue
        node_ids_by_type.setdefault(ntype, []).append(vid)

    # --- map node id -> (type, local index) ---
    id2type: Dict[str, str] = {}
    id2idx: Dict[str, int] = {}
    for ntype, vids in node_ids_by_type.items():
        for i, vid in enumerate(vids):
            id2type[vid] = ntype
            id2idx[vid] = i

    hd = HeteroData()
    for ntype, vids in node_ids_by_type.items():
        hd[ntype].node_ids = vids

    # --- collect edges per relation ---
    rel_edges: Dict[Tuple[str, str, str], list] = {}
    for e in edges:
        s = e["start_vertex"]
        t = e["end_vertex"]

        # skip edges if either endpoint excluded
        if s not in id2type or t not in id2type:
            continue

        st, tt = id2type[s], id2type[t]
        si, ti = id2idx[s], id2idx[t]

        rel = e.get("int_info", {}).get("type", default_relation)
        key = (st, rel, tt)
        rel_edges.setdefault(key, []).append((si, ti))

    # --- write edge_index tensors ---
    for (st, rel, tt), pairs in rel_edges.items():
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        hd[(st, rel, tt)].edge_index = torch.stack([src, dst], dim=0)

        if add_reverse:
            hd[(tt, f"{rev_prefix}{rel}", st)].edge_index = torch.stack([dst, src], dim=0)
            
    return hd