import random
from collections import defaultdict, deque
import torch
from torch_geometric.data import HeteroData


def get_num_nodes_safe(hd, nt):
    if "node_ids" in hd[nt]:
        return len(hd[nt].node_ids)
    if hasattr(hd[nt], "num_nodes") and hd[nt].num_nodes is not None:
        return int(hd[nt].num_nodes)
    raise ValueError(f"Cannot determine number of nodes for node type '{nt}'")


def sample_connected_balanced_node_ids(
    hd,
    target_nodes=500,
    seed=42,
    start_node_type=None,
    min_per_type=10,
    max_per_type=None,
    preferred_types=None,
):
    """
    Return sampled node indices per node type, aiming for:
      - connectedness
      - balanced modality coverage
      - total nodes <= target_nodes
    """
    random.seed(seed)
    torch.manual_seed(seed)

    node_types = list(hd.node_types)
    edge_types = list(hd.edge_types)

    if preferred_types is None:
        preferred_types = list(node_types)

    if max_per_type is None:
        max_per_type = {}

    counts = {nt: get_num_nodes_safe(hd, nt) for nt in node_types}
    active_types = [nt for nt in node_types if counts[nt] > 0]

    ordered_types = [nt for nt in preferred_types if nt in active_types]
    ordered_types += [nt for nt in active_types if nt not in ordered_types]

    # ----- quotas -----
    quota = {nt: min(min_per_type, counts[nt]) for nt in ordered_types}
    allocated = sum(quota.values())
    remaining = max(0, target_nodes - allocated)

    remaining_cap = {}
    for nt in ordered_types:
        hard_cap = min(max_per_type.get(nt, counts[nt]), counts[nt])
        remaining_cap[nt] = max(0, hard_cap - quota[nt])

    total_remaining_cap = sum(remaining_cap.values())
    if total_remaining_cap > 0 and remaining > 0:
        for nt in ordered_types:
            extra = round(remaining * (remaining_cap[nt] / total_remaining_cap))
            quota[nt] += min(extra, remaining_cap[nt])

    while sum(quota.values()) > target_nodes:
        for nt in reversed(ordered_types):
            if quota[nt] > 1 and sum(quota.values()) > target_nodes:
                quota[nt] -= 1

    # ----- adjacency for traversal -----
    adj = defaultdict(list)
    for et in edge_types:
        src_type, rel_type, dst_type = et
        edge_index = hd[et].edge_index
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()

        for s, d in zip(srcs, dsts):
            adj[(src_type, s)].append((dst_type, d, et))
            adj[(dst_type, d)].append((src_type, s, et))  # undirected traversal

    if start_node_type is None or start_node_type not in active_types:
        start_node_type = random.choice(active_types)

    start_idx = random.randrange(counts[start_node_type])

    sampled = set([(start_node_type, start_idx)])
    sampled_by_type = defaultdict(set)
    sampled_by_type[start_node_type].add(start_idx)

    frontier = deque([(start_node_type, start_idx)])

    while frontier and len(sampled) < target_nodes:
        cur = frontier.popleft()
        neighbors = adj.get(cur, [])
        random.shuffle(neighbors)

        scored = []
        for ntype, nidx, _ in neighbors:
            if (ntype, nidx) in sampled:
                continue

            current_count = len(sampled_by_type[ntype])
            target_quota = quota.get(ntype, 0)
            hard_cap = min(max_per_type.get(ntype, counts[ntype]), counts[ntype])

            if current_count >= hard_cap:
                continue

            deficit = max(0, target_quota - current_count)
            score = (deficit, random.random())
            scored.append((score, ntype, nidx))

        scored.sort(reverse=True)

        for _, ntype, nidx in scored:
            current_count = len(sampled_by_type[ntype])
            hard_cap = min(max_per_type.get(ntype, counts[ntype]), counts[ntype])

            if current_count >= hard_cap:
                continue
            if len(sampled) >= target_nodes:
                break

            sampled.add((ntype, nidx))
            sampled_by_type[ntype].add(nidx)
            frontier.append((ntype, nidx))

        if not frontier and len(sampled) < target_nodes:
            possible_frontier = list(sampled)
            random.shuffle(possible_frontier)
            frontier.extend(possible_frontier)

    sampled_nodes = {nt: sorted(list(v)) for nt, v in sampled_by_type.items()}
    return sampled_nodes, quota


def build_induced_hetero_subgraph(hd, sampled_nodes):
    """
    Build a true induced hetero subgraph from sampled_nodes.
    Keeps only sampled nodes and edges between them.
    Reindexes edge_index to local node numbering.
    """
    sub_hd = HeteroData()

    # old -> new index map for each node type
    index_maps = {}
    sampled_sets = {}

    for nt, old_indices in sampled_nodes.items():
        old_indices = sorted(old_indices)
        sampled_sets[nt] = set(old_indices)
        index_maps[nt] = {old_idx: new_idx for new_idx, old_idx in enumerate(old_indices)}

        # copy node-level attributes
        old_idx_tensor = torch.tensor(old_indices, dtype=torch.long)

        for key, value in hd[nt].items():
            if torch.is_tensor(value):
                if value.size(0) == get_num_nodes_safe(hd, nt):
                    sub_hd[nt][key] = value[old_idx_tensor]
                else:
                    sub_hd[nt][key] = value
            elif isinstance(value, list):
                if len(value) == get_num_nodes_safe(hd, nt):
                    sub_hd[nt][key] = [value[i] for i in old_indices]
                else:
                    sub_hd[nt][key] = value
            else:
                sub_hd[nt][key] = value

        # if node_ids absent, create them
        if "node_ids" not in sub_hd[nt]:
            sub_hd[nt].node_ids = old_indices

    # copy edge-level data with remapping
    for et in hd.edge_types:
        src_type, rel_type, dst_type = et

        if src_type not in sampled_nodes or dst_type not in sampled_nodes:
            continue

        edge_index = hd[et].edge_index
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()

        keep_src = sampled_sets[src_type]
        keep_dst = sampled_sets[dst_type]
        src_map = index_maps[src_type]
        dst_map = index_maps[dst_type]

        kept_positions = []
        new_srcs = []
        new_dsts = []

        for pos, (s, d) in enumerate(zip(srcs, dsts)):
            if s in keep_src and d in keep_dst:
                kept_positions.append(pos)
                new_srcs.append(src_map[s])
                new_dsts.append(dst_map[d])

        if len(new_srcs) == 0:
            continue

        sub_hd[et].edge_index = torch.tensor([new_srcs, new_dsts], dtype=torch.long)

        # copy edge attributes if they align with number of edges
        for key, value in hd[et].items():
            if key == "edge_index":
                continue
            if torch.is_tensor(value) and value.size(0) == edge_index.size(1):
                keep_pos_tensor = torch.tensor(kept_positions, dtype=torch.long)
                sub_hd[et][key] = value[keep_pos_tensor]
            else:
                sub_hd[et][key] = value

    return sub_hd

def make_small_backbone(
    hd,
    target_nodes=500,
    seed=42,
    start_node_type="gene",
    min_per_type=15,
    max_per_type=None,
    preferred_types=None,
):
    sampled_nodes, quota = sample_connected_balanced_node_ids(
        hd=hd,
        target_nodes=target_nodes,
        seed=seed,
        start_node_type=start_node_type,
        min_per_type=min_per_type,
        max_per_type=max_per_type,
        preferred_types=preferred_types,
    )

    sub_hd = build_induced_hetero_subgraph(hd, sampled_nodes)

    print("Target quota:", quota)
    print("Actual sampled counts:")
    total_nodes = 0
    for nt in sub_hd.node_types:
        n = get_num_nodes_safe(sub_hd, nt)
        total_nodes += n
        print(f"  {nt}: {n}")

    total_edges = 0
    for et in sub_hd.edge_types:
        e = sub_hd[et].edge_index.size(1)
        total_edges += e
        print(f"  {et}: {e} edges")

    print(f"\nTotal nodes: {total_nodes}")
    print(f"Total edges: {total_edges}")

    return sub_hd, sampled_nodes, quota