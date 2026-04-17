import copy
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


def _extract_logits(model_output):
    """
    Support both:
    - old models: return logits tensor
    - new multitask models: return dict with 'logits'
    """
    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output


def explain_hetero_graph_with_gradients(
    model: nn.Module,
    data: HeteroData,
    device: torch.device,
    target_class: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    data = copy.deepcopy(data).to(device)

    for node_type in data.node_types:
        data[node_type].x = data[node_type].x.clone().detach().requires_grad_(True)

    model_output = model(data)
    logits = _extract_logits(model_output)

    probs = torch.softmax(logits, dim=1).detach().cpu()
    pred_class = int(torch.argmax(probs, dim=1).item())

    if target_class is None:
        target_class = pred_class

    score = logits[0, target_class]
    model.zero_grad(set_to_none=True)
    score.backward()

    node_importance: Dict[str, torch.Tensor] = {}
    feature_importance: Dict[str, torch.Tensor] = {}

    for node_type in data.node_types:
        grad = data[node_type].x.grad
        x_abs = data[node_type].x.detach().abs()

        node_scores = (grad.abs() * x_abs).sum(dim=1).detach().cpu()
        feature_scores = (grad.abs() * x_abs).sum(dim=0).detach().cpu()

        node_importance[node_type] = node_scores
        feature_importance[node_type] = feature_scores

    return {
        "probs": probs,
        "pred_class": pred_class,
        "target_class": target_class,
        "node_importance": node_importance,
        "feature_importance": feature_importance,
    }


def summarize_top_nodes_with_missing_status(
    data: HeteroData,
    node_importance: Dict[str, torch.Tensor],
    missing_nodes_for_sample: Dict[str, List[str]],
    top_k: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    summary: Dict[str, List[Dict[str, Any]]] = {}

    for node_type, scores in node_importance.items():
        node_names = [str(x) for x in data[node_type].node_names]
        missing_set = set(missing_nodes_for_sample.get(node_type, []))

        vals, idxs = torch.topk(scores, k=min(top_k, scores.numel()))
        rows = []

        for idx, val in zip(idxs.tolist(), vals.tolist()):
            node_name = node_names[idx]
            rows.append(
                {
                    "node_name": node_name,
                    "importance": float(val),
                    "was_missing": node_name in missing_set,
                }
            )

        summary[node_type] = rows

    return summary


def explain_random_samples_and_average(
    model,
    dataset,
    device,
    k: int = 10,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Compute average node and feature importances over k random samples.

    Returns JSON-safe objects only.
    """
    if len(dataset) == 0:
        return {
            "n_samples": 0,
            "sampled_sample_ids": [],
            "node_names": {},
            "avg_node_importance": {},
            "avg_feature_importance": {},
            "avg_top_summary": {},
        }

    k = min(k, len(dataset))
    sampled_dataset = random.sample(list(dataset), k)
    reference_sample = sampled_dataset[0]

    node_importance_sum: Dict[str, torch.Tensor] = {}
    feature_importance_sum: Dict[str, torch.Tensor] = {}
    sampled_sample_ids: List[str] = []

    for sample in sampled_dataset:
        sample_id = sample.sample_id
        sampled_sample_ids.append(sample_id)

        result = explain_hetero_graph_with_gradients(
            model=model,
            data=sample,
            device=device,
        )

        for node_type, scores in result["node_importance"].items():
            scores_cpu = scores.detach().cpu()
            if node_type not in node_importance_sum:
                node_importance_sum[node_type] = scores_cpu.clone()
            else:
                node_importance_sum[node_type] += scores_cpu

        for node_type, scores in result["feature_importance"].items():
            scores_cpu = scores.detach().cpu()
            if node_type not in feature_importance_sum:
                feature_importance_sum[node_type] = scores_cpu.clone()
            else:
                feature_importance_sum[node_type] += scores_cpu

        del result

    n_samples = len(sampled_dataset)

    avg_node_importance_tensors = {
        node_type: summed_scores / n_samples
        for node_type, summed_scores in node_importance_sum.items()
    }

    avg_feature_importance_tensors = {
        node_type: summed_scores / n_samples
        for node_type, summed_scores in feature_importance_sum.items()
    }

    node_names: Dict[str, List[str]] = {}
    avg_node_importance: Dict[str, List[float]] = {}
    avg_top_summary: Dict[str, List[Dict[str, Any]]] = {}

    for node_type, scores in avg_node_importance_tensors.items():
        names = [str(x) for x in reference_sample[node_type].node_names]
        node_names[node_type] = names
        avg_node_importance[node_type] = [float(x) for x in scores.tolist()]

        vals, idxs = torch.topk(scores, k=min(top_k, scores.numel()))
        avg_top_summary[node_type] = [
            {
                "node_name": names[idx],
                "importance": float(val),
            }
            for idx, val in zip(idxs.tolist(), vals.tolist())
        ]

    avg_feature_importance: Dict[str, Dict[str, Any]] = {}
    for node_type, scores in avg_feature_importance_tensors.items():
        avg_feature_importance[node_type] = {
            "feature_names": ["observed_value", "missing_flag"],
            "importance": [float(x) for x in scores.tolist()],
        }

    return {
        "n_samples": n_samples,
        "sampled_sample_ids": sampled_sample_ids,
        "node_names": node_names,
        "avg_node_importance": avg_node_importance,
        "avg_feature_importance": avg_feature_importance,
        "avg_top_summary": avg_top_summary,
    }


def explain_all_samples_and_average(
    model,
    dataset,
    missing_tracker: Dict[str, Dict[str, List[str]]],
    device,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Explain all samples and compute average explanations.
    Supports both old and new model APIs.
    """
    per_sample_results = []

    node_importance_accumulator: Dict[str, List[torch.Tensor]] = {}
    feature_importance_accumulator: Dict[str, List[torch.Tensor]] = {}

    probs_list = []
    pred_classes = []
    raw_labels = []
    sample_ids = []

    for sample in dataset:
        sample_id = sample.sample_id
        sample_ids.append(sample_id)

        result = explain_hetero_graph_with_gradients(
            model=model,
            data=sample,
            device=device,
        )

        top_summary = summarize_top_nodes_with_missing_status(
            data=sample,
            node_importance=result["node_importance"],
            missing_nodes_for_sample=missing_tracker[sample_id],
            top_k=top_k,
        )

        probs_list.append(result["probs"].squeeze(0).detach().cpu())
        pred_classes.append(int(result["pred_class"]))
        raw_labels.append(sample.raw_label)

        for node_type, scores in result["node_importance"].items():
            node_importance_accumulator.setdefault(node_type, []).append(
                scores.detach().cpu()
            )

        for node_type, scores in result["feature_importance"].items():
            feature_importance_accumulator.setdefault(node_type, []).append(
                scores.detach().cpu()
            )

        per_sample_results.append(
            {
                "sample_id": sample_id,
                "raw_label": sample.raw_label,
                "pred_class": int(result["pred_class"]),
                "target_class": int(result["target_class"]),
                "probs": result["probs"].tolist(),
                "node_importance": {
                    node_type: tensor.tolist()
                    for node_type, tensor in result["node_importance"].items()
                },
                "feature_importance": {
                    node_type: tensor.tolist()
                    for node_type, tensor in result["feature_importance"].items()
                },
                "top_summary": top_summary,
                "missing_nodes_for_sample": missing_tracker[sample_id],
            }
        )

    avg_node_importance = {
        node_type: torch.stack(tensors, dim=0).mean(dim=0)
        for node_type, tensors in node_importance_accumulator.items()
    }

    avg_feature_importance = {
        node_type: torch.stack(tensors, dim=0).mean(dim=0)
        for node_type, tensors in feature_importance_accumulator.items()
    }

    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)

    avg_top_summary = {}
    if len(dataset) > 0:
        reference_sample = dataset[0]
        for node_type, scores in avg_node_importance.items():
            node_names = [str(x) for x in reference_sample[node_type].node_names]
            vals, idxs = torch.topk(scores, k=min(top_k, scores.numel()))

            avg_top_summary[node_type] = [
                {
                    "node_name": node_names[idx],
                    "importance": float(val),
                }
                for idx, val in zip(idxs.tolist(), vals.tolist())
            ]

    return {
        "per_sample_results": per_sample_results,
        "avg_node_importance": avg_node_importance,
        "avg_feature_importance": avg_feature_importance,
        "avg_probs": avg_probs,
        "pred_classes": pred_classes,
        "raw_labels": raw_labels,
        "avg_top_summary": avg_top_summary,
        "sample_ids": sample_ids,
    }



# import copy

# import torch
# import torch.nn as nn
# from torch_geometric.data import HeteroData

# from typing import List, Tuple, Optional, Dict, Any

# import random
# from typing import Any, Dict, List

# import torch


# def explain_random_samples_and_average(
#     model,
#     dataset,
#     device,
#     k: int = 10,
#     top_k: int = 10,
# ) -> Dict[str, Any]:
#     """
#     Compute average node and feature importances over k random samples.

#     Returns only JSON-safe objects:
#     - avg_node_importance
#     - avg_feature_importance
#     - avg_top_summary
#     - sampled_sample_ids
#     - n_samples

#     Notes
#     -----
#     - No per-sample results are stored.
#     - No probabilities are returned.
#     - Assumes all samples share the same node ordering per node type.
#     """
#     if len(dataset) == 0:
#         return {
#             "n_samples": 0,
#             "sampled_sample_ids": [],
#             "avg_node_importance": {},
#             "avg_feature_importance": {},
#             "avg_top_summary": {},
#         }

#     k = min(k, len(dataset))
#     sampled_dataset = random.sample(list(dataset), k)

#     reference_sample = sampled_dataset[0]

#     node_importance_sum: Dict[str, torch.Tensor] = {}
#     feature_importance_sum: Dict[str, torch.Tensor] = {}
#     sampled_sample_ids: List[str] = []

#     for sample in sampled_dataset:
#         sample_id = sample.sample_id
#         sampled_sample_ids.append(sample_id)

#         result = explain_hetero_graph_with_gradients(
#             model=model,
#             data=sample,
#             device=device,
#         )

#         for node_type, scores in result["node_importance"].items():
#             scores_cpu = scores.detach().cpu()
#             if node_type not in node_importance_sum:
#                 node_importance_sum[node_type] = scores_cpu.clone()
#             else:
#                 node_importance_sum[node_type] += scores_cpu

#         for node_type, scores in result["feature_importance"].items():
#             scores_cpu = scores.detach().cpu()
#             if node_type not in feature_importance_sum:
#                 feature_importance_sum[node_type] = scores_cpu.clone()
#             else:
#                 feature_importance_sum[node_type] += scores_cpu

#         del result

#     n_samples = len(sampled_dataset)

#     avg_node_importance_tensors = {
#         node_type: summed_scores / n_samples
#         for node_type, summed_scores in node_importance_sum.items()
#     }

#     avg_feature_importance_tensors = {
#         node_type: summed_scores / n_samples
#         for node_type, summed_scores in feature_importance_sum.items()
#     }

#     avg_node_importance: Dict[str, List[Dict[str, Any]]] = {}
#     avg_top_summary: Dict[str, List[Dict[str, Any]]] = {}

#     for node_type, scores in avg_node_importance_tensors.items():
#         node_names = [str(x) for x in reference_sample[node_type].node_names]

#         avg_node_importance[node_type] = [
#             {
#                 "node_name": node_names[i],
#                 "importance": float(score),
#             }
#             for i, score in enumerate(scores.tolist())
#         ]

#         vals, idxs = torch.topk(scores, k=min(top_k, scores.numel()))
#         avg_top_summary[node_type] = [
#             {
#                 "node_name": node_names[idx],
#                 "importance": float(val),
#             }
#             for idx, val in zip(idxs.tolist(), vals.tolist())
#         ]

#     avg_feature_importance: Dict[str, Dict[str, Any]] = {}
#     for node_type, scores in avg_feature_importance_tensors.items():
#         avg_feature_importance[node_type] = {
#             "feature_names": ["observed_value", "missing_flag"],
#             "importance": [float(x) for x in scores.tolist()],
#         }

#     return {
#         "n_samples": n_samples,
#         "sampled_sample_ids": sampled_sample_ids,
#         "avg_node_importance": avg_node_importance,
#         "avg_feature_importance": avg_feature_importance,
#         "avg_top_summary": avg_top_summary,
#     }

    
# def explain_all_samples_and_average(
#     model,
#     dataset,
#     missing_tracker: Dict[str, Dict[str, List[str]]],
#     device,
#     top_k: int = 10,
# ) -> Dict[str, Any]:
#     """
#     Runs gradient-based explanation for all samples and returns:
#     - per-sample explanation summaries
#     - average node importance per node type
#     - average feature importance per node type
#     - average class probabilities
#     """

#     per_sample_results = []

#     node_importance_accumulator: Dict[str, List[torch.Tensor]] = {}
#     feature_importance_accumulator: Dict[str, List[torch.Tensor]] = {}

#     probs_list = []
#     pred_classes = []
#     raw_labels = []
#     sample_ids = []
    
#     for sample in dataset:
        
#         sample_id = sample.sample_id
#         sample_ids.append(sample_id)

#         result = explain_hetero_graph_with_gradients(
#             model=model,
#             data=sample,
#             device=device,
#         )

#         top_summary = summarize_top_nodes_with_missing_status(
#             data=sample,
#             node_importance=result["node_importance"],
#             missing_nodes_for_sample=missing_tracker[sample_id],
#             top_k=top_k,
#         )

#         probs_list.append(result["probs"].squeeze(0).detach().cpu())
#         pred_classes.append(int(result["pred_class"]))
#         raw_labels.append(sample.raw_label)

#         for nt, scores in result["node_importance"].items():
#             node_importance_accumulator.setdefault(nt, []).append(scores.detach().cpu())

#         for nt, scores in result["feature_importance"].items():
#             feature_importance_accumulator.setdefault(nt, []).append(scores.detach().cpu())

#         per_sample_results.append(
#             {
#                 "sample_id": sample_id,
#                 "raw_label": sample.raw_label,
#                 "pred_class": int(result["pred_class"]),
#                 "target_class": int(result["target_class"]),
#                 "probs": result["probs"].tolist(),
#                 "node_importance": {
#                     nt: tensor.tolist()
#                     for nt, tensor in result["node_importance"].items()
#                 },
#                 "feature_importance": {
#                     nt: tensor.tolist()
#                     for nt, tensor in result["feature_importance"].items()
#                 },
#                 "top_summary": top_summary,
#                 "missing_nodes_for_sample": missing_tracker[sample_id],
#             }
#         )

#     avg_node_importance = {
#         nt: torch.stack(tensors, dim=0).mean(dim=0)
#         for nt, tensors in node_importance_accumulator.items()
#     }

#     avg_feature_importance = {
#         nt: torch.stack(tensors, dim=0).mean(dim=0)
#         for nt, tensors in feature_importance_accumulator.items()
#     }

#     avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)

#     avg_top_summary = {}
#     if len(dataset) > 0:
#         reference_sample = dataset[0]
#         for nt, scores in avg_node_importance.items():
#             node_names = [str(x) for x in reference_sample[nt].node_names]
#             vals, idxs = torch.topk(scores, k=min(top_k, scores.numel()))

#             avg_top_summary[nt] = [
#                 {
#                     "node_name": node_names[idx],
#                     "importance": float(val),
#                 }
#                 for idx, val in zip(idxs.tolist(), vals.tolist())
#             ]

#     return {
#         "per_sample_results": per_sample_results,
#         "avg_node_importance": avg_node_importance,
#         "avg_feature_importance": avg_feature_importance,
#         "avg_probs": avg_probs,
#         "pred_classes": pred_classes,
#         "raw_labels": raw_labels,
#         "avg_top_summary": avg_top_summary,
#         "sample_ids": sample_ids
#     }
    
# def explain_hetero_graph_with_gradients(
#     model: nn.Module,
#     data: HeteroData,
#     device: torch.device,
#     target_class: Optional[int] = None,
# ) -> Dict[str, Any]:
#     model.eval()
#     data = copy.deepcopy(data).to(device)

#     for nt in data.node_types:
#         data[nt].x = data[nt].x.clone().detach().requires_grad_(True)

#     logits = model(data)
#     probs = torch.softmax(logits, dim=1).detach().cpu()
#     pred_class = int(torch.argmax(probs, dim=1).item())

#     if target_class is None:
#         target_class = pred_class

#     score = logits[0, target_class]
#     model.zero_grad(set_to_none=True)
#     score.backward()

#     node_importance: Dict[str, torch.Tensor] = {}
#     feature_importance: Dict[str, torch.Tensor] = {}

#     for nt in data.node_types:
#         grad = data[nt].x.grad
#         x_abs = data[nt].x.detach().abs()

#         node_scores = (grad.abs() * x_abs).sum(dim=1).detach().cpu()
#         feat_scores = (grad.abs() * x_abs).sum(dim=0).detach().cpu()

#         node_importance[nt] = node_scores
#         feature_importance[nt] = feat_scores

#     return {
#         "probs": probs,
#         "pred_class": pred_class,
#         "target_class": target_class,
#         "node_importance": node_importance,
#         "feature_importance": feature_importance,
#     }


# def summarize_top_nodes_with_missing_status(
#     data: HeteroData,
#     node_importance: Dict[str, torch.Tensor],
#     missing_nodes_for_sample: Dict[str, List[str]],
#     top_k: int = 10,
# ) -> Dict[str, List[Dict[str, Any]]]:
#     summary: Dict[str, List[Dict[str, Any]]] = {}

#     for nt, scores in node_importance.items():
#         names = [str(x) for x in data[nt].node_names]
#         missing_set = set(missing_nodes_for_sample.get(nt, []))

#         vals, idxs = torch.topk(scores, k=min(top_k, scores.numel()))
#         rows = []

#         for idx, val in zip(idxs.tolist(), vals.tolist()):
#             node_name = names[idx]
#             rows.append({
#                 "node_name": node_name,
#                 "importance": float(val),
#                 "was_missing": node_name in missing_set,
#             })

#         summary[nt] = rows

#     return summary