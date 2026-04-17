import copy
from typing import List
from torch_geometric.data import HeteroData

def clone_heterodata_schema(hd: HeteroData) -> HeteroData:
    """
    Clone node / edge structure without copying x, batch, y.
    """
    new_hd = HeteroData()

    for nt in hd.node_types:
        for key, value in hd[nt].items():
            if key in {"x", "batch", "ptr", "y", "missing_mask"}:
                continue
            new_hd[nt][key] = copy.deepcopy(value)

    for et in hd.edge_types:
        for key, value in hd[et].items():
            new_hd[et][key] = copy.deepcopy(value)

    return new_hd


def get_node_names_from_backbone(hd: HeteroData, node_type: str) -> List[str]:
    if "node_names" in hd[node_type]:
        return [str(x) for x in hd[node_type].node_names]
    if "node_ids" in hd[node_type]:
        return [str(x) for x in hd[node_type].node_ids]
    if "names" in hd[node_type]:
        return [str(x) for x in hd[node_type].names]
    if "ids" in hd[node_type]:
        return [str(x) for x in hd[node_type].ids]

    num_nodes = int(hd[node_type].num_nodes)
    return [f"{node_type}_{i}" for i in range(num_nodes)]