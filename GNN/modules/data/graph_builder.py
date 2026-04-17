from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from modules.utils.dataset import build_label_encoder
from modules.utils.graph_io import (
    clone_heterodata_schema,
    get_node_names_from_backbone,
)


def build_sample_graph_from_store(
    backbone_hd: HeteroData,
    sample_id: str,
    sample_obj: Dict[str, Any],
    label_encoder: Dict[Any, int],
    store_key_to_node_type: Optional[Dict[str, str]] = None,
    missing_fill_value: float = 0.0,
) -> Tuple[HeteroData, Dict[str, List[str]]]:
    """
    sample_obj example:
    {
        "label": "AD",
        "data": {
            "gene": {"GENE1": 0.5, "GENE2": 0.0},
            "miRNA": {"mir-1": 0.8},
            ...
        }
    }

    Node feature format:
        [observed_value, missing_flag]
    """

    data = clone_heterodata_schema(backbone_hd)
    missing_nodes: Dict[str, List[str]] = {}

    sample_data = sample_obj["data"]

    if store_key_to_node_type is None:
        store_key_to_node_type = {k: k for k in sample_data.keys()}

    # reverse map: node_type -> store_key
    node_type_to_store_key = {
        node_type: store_key for store_key, node_type in store_key_to_node_type.items()
    }

    for nt in backbone_hd.node_types:
        node_names = get_node_names_from_backbone(backbone_hd, nt)

        observed_values: List[float] = []
        missing_flags: List[float] = []
        current_missing: List[str] = []

        if nt in node_type_to_store_key:
            store_key = node_type_to_store_key[nt]
            observed_dict = sample_data.get(store_key, {})
            observed_dict = {str(k): float(v) for k, v in observed_dict.items()}

            for node_name in node_names:
                if node_name in observed_dict:
                    # Present in dictionary, even if value == 0.0, means observed
                    observed_values.append(observed_dict[node_name])
                    missing_flags.append(0.0)
                else:
                    observed_values.append(missing_fill_value)
                    missing_flags.append(1.0)
                    current_missing.append(node_name)
        else:
            # Entire node type not provided for this sample
            for node_name in node_names:
                observed_values.append(missing_fill_value)
                missing_flags.append(1.0)
                current_missing.append(node_name)

        x = torch.tensor(
            list(zip(observed_values, missing_flags)),
            dtype=torch.float32,
        )

        data[nt].x = x
        data[nt].node_names = list(node_names)
        data[nt].missing_mask = torch.tensor(missing_flags, dtype=torch.float32)

        missing_nodes[nt] = current_missing

    raw_label = sample_obj["label"]
    data.y = torch.tensor([label_encoder[raw_label]], dtype=torch.long)
    data.sample_id = sample_id
    data.raw_label = raw_label

    return data, missing_nodes


def build_dataset_from_store(
    backbone_hd: HeteroData,
    store: Dict[str, Dict[str, Any]],
    store_key_to_node_type: Optional[Dict[str, str]] = None,
) -> Tuple[List[HeteroData], Dict[str, Dict[str, List[str]]], Dict[Any, int]]:
    """
    Returns:
        dataset
        missing_tracker[sample_id][node_type] = list of missing nodes
        label_encoder
    """
    dataset: List[HeteroData] = []
    missing_tracker: Dict[str, Dict[str, List[str]]] = {}
    label_encoder = build_label_encoder(store)

    for sample_id, sample_obj in store.items():
        g, missing_nodes = build_sample_graph_from_store(
            backbone_hd=backbone_hd,
            sample_id=sample_id,
            sample_obj=sample_obj,
            label_encoder=label_encoder,
            store_key_to_node_type=store_key_to_node_type,
        )
        dataset.append(g)
        missing_tracker[sample_id] = missing_nodes

    return dataset, missing_tracker, label_encoder