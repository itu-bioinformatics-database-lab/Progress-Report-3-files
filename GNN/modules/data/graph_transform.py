import copy
import random
from typing import Dict, List, Optional, Union, Any


class GraphSubsampler:
    """
    Utility class for modifying a graph JSON structure with keys:
        - "vertices": dict of node_id -> node_info
        - "edges": list of edge dicts

    Supported operations:
        1. Remove entire omics/node types from the graph
        2. Keep all known nodes and randomly subsample the remaining nodes
    """

    def __init__(self, graph_json: Dict[str, Any]) -> None:
        self.graph_json = graph_json

    def exclude_omic_types(
        self,
        omic_types_to_exclude: Optional[List[str]] = None,
        make_copy: bool = True,
    ) -> "GraphSubsampler":
        """
        Remove all vertices belonging to the specified omic types, then remove
        all edges connected to those removed vertices.

        Args:
            omic_types_to_exclude:
                List of omic types to remove from the graph.
            make_copy:
                If True, work on a deep copy of the graph.
                If False, modify the current graph in place.

        Returns:
            self
        """
        if omic_types_to_exclude is None:
            omic_types_to_exclude = []

        # Work on a copy if requested, otherwise modify in place
        working_graph = copy.deepcopy(self.graph_json) if make_copy else self.graph_json

        removed_node_ids = set()
        retained_vertices = {}

        # Keep only vertices whose omic type is not excluded
        for node_id, node_info in working_graph["vertices"].items():
            node_omic_type = node_info["omic_type"]

            if node_omic_type in omic_types_to_exclude:
                removed_node_ids.add(node_id)
            else:
                retained_vertices[node_id] = node_info

        working_graph["vertices"] = retained_vertices

        # Keep only edges whose endpoints were both retained
        retained_edges = []
        for edge in working_graph["edges"]:
            if (
                edge["start_vertex"] in removed_node_ids
                or edge["end_vertex"] in removed_node_ids
            ):
                continue
            retained_edges.append(edge)

        working_graph["edges"] = retained_edges
        self.graph_json = working_graph
        return self

    def sample_unknown_nodes(
        self,
        known_nodes_by_type: Optional[Dict[str, List[str]]] = None,
        unknown_node_keep_fraction: Optional[Union[float, Dict[str, float]]] = None,
        make_copy: bool = True,
    ) -> "GraphSubsampler":
        """
        Keep all known nodes, then randomly sample a proportion of the remaining
        unknown nodes for each omic type. Finally, keep only edges whose two
        endpoints are both retained.

        Args:
            known_nodes_by_type:
                Dict mapping omic_type -> list of node IDs that must always be kept.
            unknown_node_keep_fraction:
                Either:
                    - a single float applied to all omic types
                    - or a dict mapping omic_type -> float
                Example:
                    0.2
                    or
                    {"gene": 0.3, "miRNA": 0.5}
            make_copy:
                If True, work on a deep copy of the graph.
                If False, modify the current graph in place.

        Returns:
            self
        """
        if known_nodes_by_type is None:
            known_nodes_by_type = {}

        if unknown_node_keep_fraction is None:
            unknown_node_keep_fraction = 1.0

        # Work on a copy if requested, otherwise modify in place
        working_graph = copy.deepcopy(self.graph_json) if make_copy else self.graph_json

        vertices = working_graph["vertices"]
        edges = working_graph["edges"]

        # Default list of supported omic types
        omic_types = [
            "R",
            "transcript",
            "gene",
            "protein",
            "Enhancer",
            "Promoter",
            "Promoter/Enhancer",
            "miRNA",
            "protein_complex",
        ]

        # Initialize known node mapping for all omic types
        known_node_lookup = {omic_type: [] for omic_type in omic_types}
        for omic_type, node_list in known_nodes_by_type.items():
            known_node_lookup[omic_type] = list(node_list)

        # If a single float is provided, apply it to all omic types
        if isinstance(unknown_node_keep_fraction, float):
            unknown_node_keep_fraction = {
                omic_type: unknown_node_keep_fraction
                for omic_type in omic_types
            }

        # Collect candidate unknown nodes for each omic type
        unknown_nodes_by_type = {omic_type: [] for omic_type in omic_types}

        for node_id, node_info in vertices.items():
            omic_type = node_info["omic_type"]

            # Handle unexpected omic types gracefully
            if omic_type not in unknown_nodes_by_type:
                unknown_nodes_by_type[omic_type] = []

            if omic_type not in known_node_lookup:
                known_node_lookup[omic_type] = []

            if omic_type not in unknown_node_keep_fraction:
                unknown_node_keep_fraction[omic_type] = 1.0

            # If node is not in the known list, treat it as an unknown candidate
            if node_id not in known_node_lookup[omic_type]:
                unknown_nodes_by_type[omic_type].append(node_id)

        # Randomly sample unknown nodes per omic type
        sampled_unknown_nodes_by_type = {}
        for omic_type, node_list in unknown_nodes_by_type.items():
            keep_fraction = unknown_node_keep_fraction.get(omic_type, 1.0)

            n_to_keep = int(len(node_list) * keep_fraction)
            n_to_keep = max(0, min(n_to_keep, len(node_list)))

            sampled_unknown_nodes_by_type[omic_type] = random.sample(
                node_list,
                n_to_keep,
            )

        # Build retained vertex set:
        #   - all known nodes
        #   - sampled unknown nodes
        retained_vertices = {}
        retained_node_ids = set()

        for node_id, node_info in vertices.items():
            omic_type = node_info["omic_type"]

            if (
                node_id in known_node_lookup.get(omic_type, [])
                or node_id in sampled_unknown_nodes_by_type.get(omic_type, [])
            ):
                retained_vertices[node_id] = node_info
                retained_node_ids.add(node_id)

        # Keep only edges whose endpoints are both retained
        retained_edges = []
        for edge in edges:
            if (
                edge["start_vertex"] in retained_node_ids
                and edge["end_vertex"] in retained_node_ids
            ):
                retained_edges.append(edge)

        working_graph["vertices"] = retained_vertices
        working_graph["edges"] = retained_edges

        self.graph_json = working_graph
        return self