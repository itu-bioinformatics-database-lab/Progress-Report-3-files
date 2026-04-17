import random
from collections import defaultdict, deque

import torch

def biological_flow_positions(G):
    """
    Left-to-right biological flow layout.
    """
    ordered_types = [
        "miRNA",
        "gene",
        "transcript",
        "protein",
        "protein_complex",
        "Promoter",
        "Enhancer",
        "Promoter/Enhancer",
    ]

    node_types_present = sorted({G.nodes[n]["node_type"] for n in G.nodes})
    ordered_types = [nt for nt in ordered_types if nt in node_types_present]
    ordered_types += [nt for nt in node_types_present if nt not in ordered_types]

    pos = {}
    x_gap = 4.0
    y_gap = 1.8

    for col, nt in enumerate(ordered_types):
        nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == nt]
        nodes = sorted(nodes)

        x = col * x_gap
        y0 = -((len(nodes) - 1) * y_gap) / 2

        for i, n in enumerate(nodes):
            pos[n] = (x, y0 + i * y_gap)

    return pos
    
def sample_connected_balanced_hetero_subgraph(
    hd,
    target_nodes=50,
    seed=42,
    start_node_type="gene",
    min_per_type=3,
    max_per_type=None,
    preferred_types=None,
):
    """
    Sample a connected heterogeneous subgraph while trying to balance
    node counts across modalities/node types.

    Parameters
    ----------
    hd : HeteroData
    target_nodes : int
        Approximate total number of sampled nodes
    seed : int
    start_node_type : str
        Seed type for traversal
    min_per_type : int
        Desired minimum nodes per present type
    max_per_type : dict or None
        Optional hard caps, e.g. {"Enhancer": 10}
    preferred_types : list[str] or None
        Ordering for quota allocation

    Returns
    -------
    sampled_nodes : dict[node_type] -> sorted list of node indices
    sampled_edges : dict[edge_type] -> filtered edge_index
    """
    random.seed(seed)
    torch.manual_seed(seed)

    node_types = list(hd.node_types)
    edge_types = list(hd.edge_types)

    if preferred_types is None:
        preferred_types = [
            "miRNA",
            "gene",
            "transcript",
            "protein",
            "protein_complex",
            "Promoter",
            "Enhancer",
            "Promoter/Enhancer",
        ]

    if max_per_type is None:
        max_per_type = {}

    def num_nodes_of(nt):
        if "node_ids" in hd[nt]:
            return len(hd[nt].node_ids)
        if hasattr(hd[nt], "num_nodes") and hd[nt].num_nodes is not None:
            return int(hd[nt].num_nodes)
        raise ValueError(f"Cannot determine number of nodes for node type '{nt}'")

    # -----------------------------
    # Count availability
    # -----------------------------
    counts = {nt: num_nodes_of(nt) for nt in node_types}
    active_types = [nt for nt in node_types if counts[nt] > 0]

    ordered_types = [nt for nt in preferred_types if nt in active_types]
    ordered_types += [nt for nt in active_types if nt not in ordered_types]

    # -----------------------------
    # Build quotas
    # -----------------------------
    quota = {nt: min(min_per_type, counts[nt]) for nt in ordered_types}
    allocated = sum(quota.values())

    remaining = max(0, target_nodes - allocated)

    remaining_cap = {}
    for nt in ordered_types:
        hard_cap = max_per_type.get(nt, counts[nt])
        hard_cap = min(hard_cap, counts[nt])
        remaining_cap[nt] = max(0, hard_cap - quota[nt])

    total_remaining_cap = sum(remaining_cap.values())

    if total_remaining_cap > 0 and remaining > 0:
        for nt in ordered_types:
            extra = round(remaining * (remaining_cap[nt] / total_remaining_cap))
            quota[nt] += min(extra, remaining_cap[nt])

    # trim if slightly overshot due to rounding
    while sum(quota.values()) > target_nodes:
        for nt in reversed(ordered_types):
            if quota[nt] > 1 and sum(quota.values()) > target_nodes:
                quota[nt] -= 1

    # -----------------------------
    # Build adjacency
    # -----------------------------
    adj = defaultdict(list)
    for et in edge_types:
        src_type, rel_type, dst_type = et
        edge_index = hd[et].edge_index
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()

        for s, d in zip(srcs, dsts):
            adj[(src_type, s)].append((dst_type, d, et))
            adj[(dst_type, d)].append((src_type, s, et))  # for traversal only

    # -----------------------------
    # Seed
    # -----------------------------
    if start_node_type not in active_types:
        start_node_type = random.choice(active_types)

    start_idx = random.randrange(counts[start_node_type])

    sampled = set()
    sampled.add((start_node_type, start_idx))

    sampled_by_type = defaultdict(set)
    sampled_by_type[start_node_type].add(start_idx)

    frontier = deque()
    frontier.append((start_node_type, start_idx))

    # -----------------------------
    # Balanced connected expansion
    # -----------------------------
    while frontier and len(sampled) < target_nodes:
        cur = frontier.popleft()
        neighbors = adj.get(cur, [])

        # score neighbors: prefer under-quota node types
        scored = []
        for ntype, nidx, et in neighbors:
            if (ntype, nidx) in sampled:
                continue

            current_count = len(sampled_by_type[ntype])
            target_quota = quota.get(ntype, 0)
            hard_cap = max_per_type.get(ntype, counts[ntype])

            if current_count >= min(hard_cap, counts[ntype]):
                continue

            # higher priority if farther below quota
            deficit = max(0, target_quota - current_count)

            # small random jitter to avoid deterministic ugly shapes
            score = (deficit, random.random())
            scored.append((score, ntype, nidx))

        # sort descending by deficit
        scored.sort(reverse=True)

        for _, ntype, nidx in scored:
            current_count = len(sampled_by_type[ntype])
            hard_cap = max_per_type.get(ntype, counts[ntype])

            if current_count >= min(hard_cap, counts[ntype]):
                continue
            if len(sampled) >= target_nodes:
                break

            sampled.add((ntype, nidx))
            sampled_by_type[ntype].add(nidx)
            frontier.append((ntype, nidx))

        # if frontier empties before reaching target, reseed from sampled set
        if not frontier and len(sampled) < target_nodes:
            possible_frontier = list(sampled)
            random.shuffle(possible_frontier)
            for node in possible_frontier:
                frontier.append(node)

    # -----------------------------
    # If still missing quota for some types, try to attach them via one-hop neighbors
    # -----------------------------
    changed = True
    while changed and len(sampled) < target_nodes:
        changed = False
        for nt in ordered_types:
            while len(sampled_by_type[nt]) < quota[nt] and len(sampled) < target_nodes:
                # find a sampled node that connects to a missing-type node
                candidate = None
                for s_nt, s_idx in list(sampled):
                    neighbors = adj.get((s_nt, s_idx), [])
                    for ntype, nidx, _ in neighbors:
                        if ntype == nt and (ntype, nidx) not in sampled:
                            candidate = (ntype, nidx)
                            break
                    if candidate is not None:
                        break

                if candidate is None:
                    break

                sampled.add(candidate)
                sampled_by_type[candidate[0]].add(candidate[1])
                changed = True

    sampled_nodes = {nt: sorted(list(v)) for nt, v in sampled_by_type.items()}

    # -----------------------------
    # Filter edges among sampled nodes
    # -----------------------------
    sampled_node_sets = {nt: set(v) for nt, v in sampled_nodes.items()}
    sampled_edges = {}

    for et in edge_types:
        src_type, rel_type, dst_type = et
        edge_index = hd[et].edge_index
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()

        keep_mask = [
            (s in sampled_node_sets.get(src_type, set())) and
            (d in sampled_node_sets.get(dst_type, set()))
            for s, d in zip(srcs, dsts)
        ]

        if any(keep_mask):
            keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
            sampled_edges[et] = edge_index[:, keep_mask]

    return sampled_nodes, sampled_edges, quota


def filter_top_k_edges(G, k=3):
    new_edges = []
    for node in G.nodes():
        edges = list(G.edges(node, data=True))
        edges = edges[:k]  # simple version (or rank if you have weights)
        new_edges.extend([(u, v, d) for u, v, d in edges])
    return new_edges

def build_connected_networkx(
    hd,
    sampled_nodes,
    sampled_edges,
    graph_json=None,
    max_label_len=18,
):
    G = nx.MultiDiGraph()

    name_lookup = build_name_lookup_from_graph_json(graph_json) if graph_json is not None else {}

    # node id mapping
    for nt, idx_list in sampled_nodes.items():
        node_ids = hd[nt].node_ids if "node_ids" in hd[nt] else list(range(len(idx_list)))

        for idx in idx_list:
            raw_id = node_ids[idx] if idx < len(node_ids) else idx
            if isinstance(raw_id, torch.Tensor):
                raw_id = raw_id.item()

            display = name_lookup.get((nt, raw_id), str(raw_id))
            display = str(display)[:max_label_len]

            G.add_node(
                f"{nt}:{idx}",
                node_type=nt,
                node_index=idx,
                raw_id=raw_id,
                label=display,
            )

    # edges
    for et, edge_index in sampled_edges.items():
        src_type, rel_type, dst_type = et
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()

        for s, d in zip(srcs, dsts):
            u = f"{src_type}:{s}"
            v = f"{dst_type}:{d}"
            if u in G and v in G:
                G.add_edge(u, v, relation=rel_type)

    return G

import random
from collections import defaultdict, deque

import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# --------------------------------------------------
# 1. Build a connected sample from heterodata
# --------------------------------------------------
def sample_connected_hetero_subgraph(
    hd,
    target_nodes=50,
    seed=42,
    start_node_type=None,
):
    """
    Sample a connected-ish heterogeneous subgraph by expanding from a seed node
    through existing edges.

    Returns
    -------
    sampled_nodes : dict[node_type] -> sorted list of node indices
    sampled_edges : dict[edge_type] -> filtered edge_index
    """
    random.seed(seed)
    torch.manual_seed(seed)

    node_types = list(hd.node_types)
    edge_types = list(hd.edge_types)

    def num_nodes_of(nt):
        if "node_ids" in hd[nt]:
            return len(hd[nt].node_ids)
        if hasattr(hd[nt], "num_nodes") and hd[nt].num_nodes is not None:
            return int(hd[nt].num_nodes)
        raise ValueError(f"Cannot determine number of nodes for node type '{nt}'")

    # pick a seed node type with available nodes
    available_types = [nt for nt in node_types if num_nodes_of(nt) > 0]
    if start_node_type is None:
        start_node_type = random.choice(available_types)

    start_idx = random.randrange(num_nodes_of(start_node_type))

    # adjacency by typed node
    adj = defaultdict(list)
    for et in edge_types:
        src_type, rel_type, dst_type = et
        edge_index = hd[et].edge_index
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()

        for s, d in zip(srcs, dsts):
            adj[(src_type, s)].append((dst_type, d, et))
            adj[(dst_type, d)].append((src_type, s, et))  # undirected expansion for sampling

    sampled = set()
    q = deque()

    sampled.add((start_node_type, start_idx))
    q.append((start_node_type, start_idx))

    while q and len(sampled) < target_nodes:
        cur = q.popleft()
        neighbors = adj.get(cur, [])
        random.shuffle(neighbors)

        for ntype, nidx, _ in neighbors:
            if (ntype, nidx) not in sampled:
                sampled.add((ntype, nidx))
                q.append((ntype, nidx))
            if len(sampled) >= target_nodes:
                break

    # organize sampled nodes by type
    sampled_nodes = defaultdict(list)
    for nt, idx in sampled:
        sampled_nodes[nt].append(idx)

    sampled_nodes = {nt: sorted(v) for nt, v in sampled_nodes.items()}

    # filter edges to sampled nodes only
    sampled_node_sets = {nt: set(v) for nt, v in sampled_nodes.items()}
    sampled_edges = {}

    for et in edge_types:
        src_type, rel_type, dst_type = et
        edge_index = hd[et].edge_index

        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()

        keep_mask = [
            (s in sampled_node_sets.get(src_type, set())) and
            (d in sampled_node_sets.get(dst_type, set()))
            for s, d in zip(srcs, dsts)
        ]

        if any(keep_mask):
            keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
            sampled_edges[et] = edge_index[:, keep_mask]

    return dict(sampled_nodes), sampled_edges


# --------------------------------------------------
# 2. Optional name lookup from graph_json
# --------------------------------------------------
def build_name_lookup_from_graph_json(graph_json):
    """
    Tries to build {(node_type, node_id_or_index): display_name}
    Works best if your graph_json stores node names/labels.
    Adjust keys if your JSON schema differs.
    """
    lookup = {}

    if isinstance(graph_json, dict):
        for nt, items in graph_json.items():
            if isinstance(items, list):
                for i, item in enumerate(items):
                    if isinstance(item, dict):
                        node_id = item.get("id", i)
                        label = (
                            item.get("name")
                            or item.get("label")
                            or item.get("symbol")
                            or str(node_id)
                        )
                        lookup[(nt, node_id)] = label
    return lookup


# --------------------------------------------------
# 3. Convert to NetworkX
# --------------------------------------------------
def build_connected_networkx(
    hd,
    sampled_nodes,
    sampled_edges,
    graph_json=None,
    max_label_len=18,
):
    G = nx.MultiDiGraph()

    name_lookup = build_name_lookup_from_graph_json(graph_json) if graph_json is not None else {}

    # node id mapping
    for nt, idx_list in sampled_nodes.items():
        node_ids = hd[nt].node_ids if "node_ids" in hd[nt] else list(range(len(idx_list)))

        for idx in idx_list:
            raw_id = node_ids[idx] if idx < len(node_ids) else idx
            if isinstance(raw_id, torch.Tensor):
                raw_id = raw_id.item()

            display = name_lookup.get((nt, raw_id), str(raw_id))
            display = str(display)[:max_label_len]

            G.add_node(
                f"{nt}:{idx}",
                node_type=nt,
                node_index=idx,
                raw_id=raw_id,
                label=display,
            )

    # edges
    for et, edge_index in sampled_edges.items():
        src_type, rel_type, dst_type = et
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()

        for s, d in zip(srcs, dsts):
            u = f"{src_type}:{s}"
            v = f"{dst_type}:{d}"
            if u in G and v in G:
                G.add_edge(u, v, relation=rel_type)

    return G


# --------------------------------------------------
# 4. Layered layout by biology/entity type
# --------------------------------------------------
def layered_positions(G):
    """
    Place node types in horizontal layers for a presentation-friendly plot.
    """
    preferred_order = [
        "miRNA",
        "gene",
        "transcript",
        "protein",
        "protein_complex",
        "Promoter",
        "Enhancer",
        "Promoter/Enhancer",
    ]

    node_types_present = sorted({G.nodes[n]["node_type"] for n in G.nodes})
    ordered_types = [nt for nt in preferred_order if nt in node_types_present]
    ordered_types += [nt for nt in node_types_present if nt not in ordered_types]

    pos = {}
    y_gap = 2.5
    x_gap = 1.8

    for layer, nt in enumerate(ordered_types):
        nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == nt]
        nodes = sorted(nodes)

        y = -layer * y_gap
        x0 = -((len(nodes) - 1) * x_gap) / 2

        for i, n in enumerate(nodes):
            pos[n] = (x0 + i * x_gap, y)

    return pos


# --------------------------------------------------
# 5. Plot with shapes by node type and styles by relation

# --------------------------------------------------
def plot_connected_hetero_graph(
    G,
    figsize=(18, 10),
    title="Connected Multi-Omics Regulatory Subgraph",
    dpi=300,
    save_path=None,
):
    plt.figure(figsize=figsize)

    #pos = layered_positions(G)
    pos = biological_flow_positions(G)

    # node styles
    node_style = {
        "gene":               dict(color="#9467bd", shape="o"),
        "transcript":         dict(color="#9edae5", shape="s"),
        "protein":            dict(color="#7f7f7f", shape="^"),
        "miRNA":              dict(color="#c49c94", shape="D"),
        "protein_complex":    dict(color="#dbdb8d", shape="P"),
        "Promoter":           dict(color="#ff7f0e", shape="v"),
        "Enhancer":           dict(color="#1f77b4", shape="h"),
        "Promoter/Enhancer":  dict(color="#98df8a", shape="8"),
    }

    # fallback
    default_style = dict(color="#cccccc", shape="o")

    # draw edges grouped by relation
    relation_styles = {
        "gene - transcript": dict(color="black", style="solid", width=1.8),
        "transcript - protein": dict(color="dimgray", style="solid", width=1.6),
        "miRNA - transcript": dict(color="crimson", style="dashed", width=1.8),
        "protein - protein": dict(color="gray", style="dotted", width=1.4),
        "protein - protein_complex": dict(color="olive", style="solid", width=1.6),
        "protein_complex - protein": dict(color="olive", style="solid", width=1.6),
        "Promoter/Enhancer - gene": dict(color="green", style="solid", width=1.8),
        "Enhancer - gene": dict(color="royalblue", style="solid", width=1.7),
        "Promoter - gene": dict(color="darkorange", style="solid", width=1.7),
        "protein - Promoter/Enhancer": dict(color="purple", style="dashdot", width=1.5),
        "protein - Enhancer": dict(color="purple", style="dashdot", width=1.5),
        "protein - Promoter": dict(color="purple", style="dashdot", width=1.5),
    }

    # draw edges
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        estyle = relation_styles.get(rel, dict(color="lightgray", style="solid", width=1.0))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            edge_color=estyle["color"],
            style=estyle["style"],
            width=estyle["width"],
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            alpha=0.75,
            connectionstyle="arc3,rad=0.08",
        )

    # draw nodes by type
    node_types = sorted({G.nodes[n]["node_type"] for n in G.nodes})
    for nt in node_types:
        nodelist = [n for n in G.nodes if G.nodes[n]["node_type"] == nt]
        st = node_style.get(nt, default_style)

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_color=st["color"],
            node_shape=st["shape"],
            node_size=900,
            edgecolors="black",
            linewidths=1.0,
            alpha=0.95,
        )

    # labels
    labels = {}
    for n in G.nodes:
        nt = G.nodes[n]["node_type"]
        lbl = G.nodes[n]["label"]
        labels[n] = f"{nt}\n{lbl}"

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=7,
        font_weight="normal",
    )

    # legends
    node_legend = []
    for nt in node_types:
        st = node_style.get(nt, default_style)
        node_legend.append(
            Line2D(
                [0], [0],
                marker=st["shape"],
                color="w",
                label=nt,
                markerfacecolor=st["color"],
                markeredgecolor="black",
                markersize=10,
                linewidth=0,
            )
        )

    edge_legend = []
    shown = set()
    for rel, st in relation_styles.items():
        # show only relations present
        if any(d.get("relation") == rel for _, _, d in G.edges(data=True)):
            shown.add(rel)
            edge_legend.append(
                Line2D(
                    [0], [0],
                    color=st["color"],
                    linestyle=st["style"],
                    linewidth=st["width"],
                    label=rel,
                )
            )

    leg1 = plt.legend(
        handles=node_legend,
        title="Node types",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )
    plt.gca().add_artist(leg1)

    if edge_legend:
        plt.legend(
            handles=edge_legend,
            title="Relations",
            bbox_to_anchor=(1.02, 0.45),
            loc="upper left",
            fontsize=8,
        )

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

def main():

    G_vis = build_connected_networkx(
        hd,
        sampled_nodes,
        sampled_edges,
        graph_json=graph_json,
    )
    # ---------------------------
    # FILTER EDGES (NEW STEP)
    # ---------------------------
    filtered_edges = filter_top_k_edges(G_vis, k=2)
    
    # Create a new graph with same nodes but fewer edges
    G_filtered = G_vis.__class__()  # keeps MultiDiGraph type
    G_filtered.add_nodes_from(G_vis.nodes(data=True))
    G_filtered.add_edges_from(filtered_edges)
    
    print("Nodes:", G_filtered.number_of_nodes())
    print("Edges:", G_filtered.number_of_edges())
    
    plot_connected_hetero_graph(
        G_filtered,
        figsize = (12,6),
        title="",
        save_path = "graph.png"
    )