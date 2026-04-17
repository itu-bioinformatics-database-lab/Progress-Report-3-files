import torch
from typing import Dict, List, Any


def _extract_logits(model_output):
    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output


def _extract_predicted_node_values(model_output):
    """
    Returns:
        predicted_values dict if present, else None
    """
    if isinstance(model_output, dict) and "predicted_values" in model_output:
        return model_output["predicted_values"]
    return None


# =========================================================
# 1. SAVE MODEL
# =========================================================
def save_model_checkpoint(
    model,
    save_path: str,
    metadata=None,
    config: dict = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "config": config or {},
    }
    torch.save(checkpoint, save_path)
    print(f"Saved model checkpoint to: {save_path}")


# =========================================================
# 2. LOAD MODEL
# =========================================================
def load_model_checkpoint(model, checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# =========================================================
# 3. PREDICT CLASSES FOR DATASET
# =========================================================
@torch.no_grad()
def predict_dataset_classes(
    model,
    dataset,
    device,
) -> List[Dict[str, Any]]:
    model.eval()
    results = []

    for sample in dataset:
        sample_device = sample.to(device)

        model_output = model(sample_device)
        logits = _extract_logits(model_output)

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        pred_class = int(torch.argmax(probs).item())

        true_label = None
        if hasattr(sample, "y"):
            try:
                true_label = int(sample.y.view(-1)[0].item())
            except Exception:
                true_label = None

        results.append(
            {
                "sample_id": sample.sample_id,
                "raw_label": getattr(sample, "raw_label", None),
                "true_label_encoded": true_label,
                "pred_class": pred_class,
                "probs": probs.tolist(),
                "is_correct": (
                    true_label == pred_class if true_label is not None else None
                ),
            }
        )

    return results


# =========================================================
# 4. COLLECT NODE VALUE PREDICTIONS
# =========================================================
@torch.no_grad()
def predict_dataset_node_values(
    model,
    dataset,
    device,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Collect node-level predicted scalar values from multitask model.

    Returns:
    {
        sample_id: {
            omic_type: {
                node_id: predicted_value
            }
        }
    }
    """
    model.eval()
    predicted_node_values: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sample in dataset:
        sample_id = sample.sample_id
        sample_device = sample.to(device)

        model_output = model(sample_device)
        pred_dict = _extract_predicted_node_values(model_output)

        predicted_node_values[sample_id] = {}

        if pred_dict is None:
            # classifier-only model: no node value prediction
            for omic_type in sample.node_types:
                predicted_node_values[sample_id][omic_type] = {}
            continue

        for omic_type in sample.node_types:
            node_names = [str(x) for x in sample[omic_type].node_names]
            pred_values = pred_dict[omic_type].detach().cpu().tolist()

            predicted_node_values[sample_id][omic_type] = {
                node_id: float(pred_values[i])
                for i, node_id in enumerate(node_names)
            }

    return predicted_node_values


# =========================================================
# 5. EXPORT NODE VALUES
# =========================================================
def export_sample_node_values(
    dataset,
    predicted_node_values: Dict[str, Dict[str, Dict[str, float]]] = None,
    use_observed_if_available: bool = True,
    missing_fill_value=None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Format:

    {
        sample_id: {
            omic_type: {
                node_id: value
            }
        }
    }

    Behavior:
    - if observed and use_observed_if_available=True -> keep original value
    - else if predicted value exists -> use predicted value
    - else -> use missing_fill_value
    """
    if predicted_node_values is None:
        predicted_node_values = {}

    exported = {}

    for sample in dataset:
        sample_id = sample.sample_id
        exported[sample_id] = {}

        for omic_type in sample.node_types:
            node_names = [str(x) for x in sample[omic_type].node_names]
            x = sample[omic_type].x.detach().cpu()

            exported[sample_id][omic_type] = {}

            sample_pred = predicted_node_values.get(sample_id, {}).get(omic_type, {})

            for i, node_id in enumerate(node_names):
                observed_value = float(x[i, 0].item())
                missing_flag = float(x[i, 1].item())

                is_observed = (missing_flag == 0.0)

                if use_observed_if_available and is_observed:
                    value = observed_value
                elif node_id in sample_pred:
                    value = float(sample_pred[node_id])
                else:
                    value = missing_fill_value

                exported[sample_id][omic_type][node_id] = value

    return exported


# =========================================================
# 6. RUN FULL INFERENCE PIPELINE
# =========================================================
def run_full_inference(
    model,
    prepared: Dict[str, Any],
    device,
    output_dir: str,
    save_json,
):
    train_dataset = prepared["train_dataset"]
    test_dataset = prepared["test_dataset"]
    full_dataset = prepared["dataset"]

    # ---- Class predictions ----
    train_preds = predict_dataset_classes(model, train_dataset, device)
    test_preds = predict_dataset_classes(model, test_dataset, device)
    all_preds = predict_dataset_classes(model, full_dataset, device)

    save_json(f"{output_dir}/train_predictions.json", train_preds)
    save_json(f"{output_dir}/test_predictions.json", test_preds)
    save_json(f"{output_dir}/all_predictions.json", all_preds)

    # ---- Node value predictions ----
    predicted_node_values = predict_dataset_node_values(
        model=model,
        dataset=full_dataset,
        device=device,
    )

    save_json(f"{output_dir}/predicted_node_values.json", predicted_node_values)

    # ---- Final exported node values ----
    # observed values kept where present, predicted values used for missing nodes
    node_values = export_sample_node_values(
        dataset=full_dataset,
        predicted_node_values=predicted_node_values,
        use_observed_if_available=True,
        missing_fill_value=None,
    )

    save_json(f"{output_dir}/all_sample_node_values.json", node_values)

    print("Inference completed and saved.")


# # modules/inference/prediction_utils.py

# import torch
# from typing import Dict, List, Any

# # =========================================================
# # 1. SAVE MODEL
# # =========================================================
# def save_model_checkpoint(
#     model,
#     save_path: str,
#     metadata=None,
#     config: dict = None,
# ) -> None:
#     checkpoint = {
#         "model_state_dict": model.state_dict(),
#         "metadata": metadata,
#         "config": config or {},
#     }
#     torch.save(checkpoint, save_path)
#     print(f"Saved model checkpoint to: {save_path}")


# # =========================================================
# # 2. LOAD MODEL (OPTIONAL BUT USEFUL)
# # =========================================================
# def load_model_checkpoint(model, checkpoint_path: str, device):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device)
#     model.eval()
#     return model, checkpoint


# # =========================================================
# # 3. PREDICT CLASSES FOR DATASET
# # =========================================================
# @torch.no_grad()
# def predict_dataset_classes(
#     model,
#     dataset,
#     device,
# ) -> List[Dict[str, Any]]:
#     model.eval()
#     results = []

#     for sample in dataset:
#         sample_device = sample.to(device)

#         logits = model(sample_device)
#         probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
#         pred_class = int(torch.argmax(probs).item())

#         true_label = None
#         if hasattr(sample, "y"):
#             try:
#                 true_label = int(sample.y.view(-1)[0].item())
#             except Exception:
#                 pass

#         results.append(
#             {
#                 "sample_id": sample.sample_id,
#                 "raw_label": getattr(sample, "raw_label", None),
#                 "true_label_encoded": true_label,
#                 "pred_class": pred_class,
#                 "probs": probs.tolist(),
#                 "is_correct": (
#                     true_label == pred_class if true_label is not None else None
#                 ),
#             }
#         )

#     return results


# # =========================================================
# # 4. EXPORT NODE VALUES (FLEXIBLE)
# # =========================================================
# def export_sample_node_values(
#     dataset,
#     predicted_node_values: Dict[str, Dict[str, Dict[str, float]]] = None,
#     use_observed_if_available: bool = True,
#     missing_fill_value=None,
# ) -> Dict[str, Dict[str, Dict[str, float]]]:
#     """
#     Format:

#     {
#         sample_id: {
#             omic_type: {
#                 node_id: value
#             }
#         }
#     }
#     """
#     if predicted_node_values is None:
#         predicted_node_values = {}

#     exported = {}

#     for sample in dataset:
#         sample_id = sample.sample_id
#         exported[sample_id] = {}

#         for omic_type in sample.node_types:
#             node_names = [str(x) for x in sample[omic_type].node_names]
#             x = sample[omic_type].x.detach().cpu()

#             exported[sample_id][omic_type] = {}

#             sample_pred = predicted_node_values.get(sample_id, {}).get(omic_type, {})

#             for i, node_id in enumerate(node_names):
#                 observed_value = float(x[i, 0].item())
#                 missing_flag = float(x[i, 1].item())

#                 is_observed = (missing_flag == 0.0)

#                 if use_observed_if_available and is_observed:
#                     value = observed_value
#                 elif node_id in sample_pred:
#                     value = float(sample_pred[node_id])
#                 else:
#                     value = missing_fill_value

#                 exported[sample_id][omic_type][node_id] = value

#     return exported


# # =========================================================
# # 5. RUN FULL INFERENCE PIPELINE
# # =========================================================
# def run_full_inference(
#     model,
#     prepared: Dict[str, Any],
#     device,
#     output_dir: str,
#     save_json,
# ):
#     """
#     Runs:
#     - predictions (train/test/all)
#     - node export
#     - saves everything
#     """

#     train_dataset = prepared["train_dataset"]
#     test_dataset = prepared["test_dataset"]
#     full_dataset = prepared["dataset"]

#     # ---- Predictions ----
#     train_preds = predict_dataset_classes(model, train_dataset, device)
#     test_preds = predict_dataset_classes(model, test_dataset, device)
#     all_preds = predict_dataset_classes(model, full_dataset, device)

#     save_json(f"{output_dir}/train_predictions.json", train_preds)
#     save_json(f"{output_dir}/test_predictions.json", test_preds)
#     save_json(f"{output_dir}/all_predictions.json", all_preds)

#     # ---- Node values ----
#     node_values = export_sample_node_values(
#         dataset=full_dataset,
#         predicted_node_values=None,
#         use_observed_if_available=True,
#         missing_fill_value=None,
#     )

#     save_json(f"{output_dir}/all_sample_node_values.json", node_values)

#     print("Inference completed and saved.")