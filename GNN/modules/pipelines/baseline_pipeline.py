import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Core torch (needed indirectly in pipeline functions)
import torch
from torch_geometric.data import HeteroData

# Your internal modules
from modules import (
    build_heterodata_from_json,
    attach_sample_values,
    HeteroImputer,
    train_imputer_one_sample,
    predict_nodes,
    generate_fake_sample_x,
)

from modules.data.graph_builder import (
    build_sample_graph_from_store,
    build_dataset_from_store,
)

from modules.utils.dataset import (
    infer_num_classes,
    split_dataset,
    make_loaders,
)

from modules.models.hetero_model import MissingAwareHeteroClassifier
from modules.models.hetero_impute_classify import HeteroImputeClassifyModel
from modules.models.losses import compute_multitask_loss

from modules.training.train import *

from modules.explain.explainer import (
    explain_hetero_graph_with_gradients,
    summarize_top_nodes_with_missing_status,
)

from modules.visualization.plots import (
    plot_training_history,
    plot_selected_node_importance_grid,
)

def load_store(store_path: str) -> Dict[str, Dict[str, Any]]:
    store_path = Path(store_path)
    if not store_path.exists():
        raise FileNotFoundError(
            f"Store file not found at {store_path}. "
            "Set STORE_PATH_GENE_miRNA correctly."
        )

    with open(store_path, "r") as f:
        store = json.load(f)

    first_sid = next(iter(store))
    print("\nFirst sample ID:", first_sid)
    print("First sample label:", store[first_sid]["label"])
    print("Available store omics keys:", list(store[first_sid]["data"].keys()))

    return store


def prepare_dataset(
    hd: HeteroData,
    store_path: str,
    store_key_to_node_type=None,
    train_frac: float = 0.8,
    batch_size: int = 4,
    seed: int = 42,
):
    store = load_store(store_path)
    first_sid = next(iter(store))

    dataset, missing_tracker, label_encoder = build_dataset_from_store(
        backbone_hd=hd,
        store=store,
        store_key_to_node_type=store_key_to_node_type,
    )

    print("\nDataset size:", len(dataset))
    print("Label encoder:", label_encoder)
    print("Metadata:", dataset[0].metadata())
    print("Num classes:", infer_num_classes(dataset))

    print("\nObserved / missing counts for first sample:")
    for nt in dataset[0].node_types:
        n_total = len(dataset[0][nt].node_names)
        n_missing = len(missing_tracker[first_sid][nt])
        n_observed = n_total - n_missing
        print(f"  {nt}: observed={n_observed}, missing={n_missing}, total={n_total}")

    train_dataset, test_dataset = split_dataset(
        dataset,
        train_frac=train_frac,
        seed=seed,
    )

    train_loader, test_loader = make_loaders(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
    )

    return {
        "hd": hd,
        "store": store,
        "dataset": dataset,
        "missing_tracker": missing_tracker,
        "label_encoder": label_encoder,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }


def build_model(
    metadata,
    num_classes: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    head = None
):
    # model = MissingAwareHeteroClassifier(
    #     metadata=metadata,
    #     hidden_dim=hidden_dim,
    #     out_dim=num_classes,
    #     num_layers=num_layers,
    #     dropout=dropout,
    #     head=head
    # )

    model = HeteroImputeClassifyModel(
        metadata=metadata,
        hidden_dim=hidden_dim,
        out_dim=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        head=head
    )
    
    return model


def train_pipeline(
    train_loader,
    test_loader,
    metadata,
    device,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 30,
    head = None
):
    num_classes = infer_num_classes(train_loader.dataset)
    model = build_model(
        metadata=metadata,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        head=head
    )
    
    # model, history = fit_model(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     lr=lr,
    #     epochs=epochs,
    # )
    model, history = fit_model_multitask(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        lr=lr,
        epochs=epochs,
        reconstruction_weight=0.5,
    )
    return model, history


def explain_sample(
    model,
    sample,
    missing_tracker: Dict[str, Dict[str, List[str]]],
    device,
    top_k: int = 10,
):
    sample_id = sample.sample_id

    result = explain_hetero_graph_with_gradients(
        model=model,
        data=sample,
        device=device,
    )

    summary = summarize_top_nodes_with_missing_status(
        data=sample,
        node_importance=result["node_importance"],
        missing_nodes_for_sample=missing_tracker[sample_id],
        top_k=top_k,
    )

    return sample_id, result, summary


def print_explanation_report(
    sample,
    sample_id: str,
    result: Dict[str, Any],
    top_summary: Dict[str, List[Dict[str, Any]]],
    preview_top_n: int = 5,
) -> None:
    print("\nExplanation result")
    print("Sample ID:", sample_id)
    print("Raw label:", sample.raw_label)
    print("Predicted probabilities:", result["probs"].numpy())
    print("Predicted class:", result["pred_class"])
    print("Explained target class:", result["target_class"])

    for nt, rows in top_summary.items():
        print(f"\nTop important nodes for node type: {nt}")
        for row in rows[:preview_top_n]:
            print(row)


def inspect_missing_nodes(
    sample,
    sample_id: str,
    missing_tracker: Dict[str, Dict[str, List[str]]],
    node_type: Optional[str] = None,
    preview_n: int = 10,
) -> None:
    print("\nExample missing-node lookup:")

    some_nt = node_type if node_type is not None else sample.node_types[0]
    print(
        f"{sample_id} | node_type={some_nt} | "
        f"n_missing={len(missing_tracker[sample_id][some_nt])}"
    )
    print(missing_tracker[sample_id][some_nt][:preview_n])


def plot_results(
    history: Dict[str, List[float]],
    sample,
    sample_id: str,
    result: Dict[str, Any],
    missing_tracker: Dict[str, Dict[str, List[str]]],
    training_save_path: str = "results/training_curve.png",
    training_figsize: Tuple[int, int] = (8, 5),
    node_types_to_plot: Optional[List[str]] = None,
    node_top_k: int = 35,
    node_plot_save_path: Optional[str] = None,
    node_plot_figsize: Tuple[int, int] = (24, 8),
    dpi: int = 300,
) -> None:
    if node_types_to_plot is None:
        node_types_to_plot = ["gene", "miRNA", "transcript"]

    if node_plot_save_path is None:
        node_plot_save_path = f"node_importance_{sample_id}_2x3.png"

    plot_training_history(
        history,
        save_path=training_save_path,
        figsize=training_figsize,
        dpi=dpi,
    )

    plot_selected_node_importance_grid(
        data=sample,
        node_importance=result["node_importance"],
        missing_nodes_for_sample=missing_tracker[sample_id],
        selected_node_types=node_types_to_plot,
        top_k=node_top_k,
        save_path=node_plot_save_path,
        dpi=dpi,
        figsize=node_plot_figsize,
    )
