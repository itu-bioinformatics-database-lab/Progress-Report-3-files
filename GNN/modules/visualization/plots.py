from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch
from torch_geometric.data import HeteroData

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 4),
    dpi: int = 600,
    show: bool = True,
) -> None:

    if "train_loss" in history:
        mode = "single_task"
        n_epochs = len(history["train_loss"])
    elif "train_total_loss" in history:
        mode = "multitask"
        n_epochs = len(history["train_total_loss"])
    else:
        raise KeyError("Invalid history format")

    epochs = range(1, n_epochs + 1)

    # ---------------- LOSS ----------------
    plt.figure(figsize=figsize)

    if mode == "single_task":
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.title("Training Loss")
        suffix = "_loss"
    else:
        plt.plot(epochs, history["train_total_loss"], label="Total Loss")
        plt.plot(epochs, history["train_classification_loss"], label="Classification")
        plt.plot(epochs, history["train_reconstruction_loss"], label="Reconstruction")
        plt.title("Training Loss Components")
        suffix = "_loss_components"

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if save_path:
        base = Path(save_path).with_suffix("")
        plt.savefig(base.with_name(base.name + suffix + ".png"), dpi=dpi, bbox_inches="tight")
        plt.savefig(base.with_name(base.name + suffix + ".pdf"), bbox_inches="tight")  # VECTOR

    if show:
        plt.show()
    else:
        plt.close()

    # ---------------- ACCURACY ----------------
    plt.figure(figsize=figsize)

    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["test_acc"], label="Test Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Test Accuracy")
    plt.legend()
    plt.tight_layout()

    if save_path:
        base = Path(save_path).with_suffix("")
        plt.savefig(base.with_name(base.name + "_accuracy.png"), dpi=dpi, bbox_inches="tight")
        plt.savefig(base.with_name(base.name + "_accuracy.pdf"), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_selected_node_importance_grid(
    data: HeteroData,
    node_importance: Dict[str, torch.Tensor],
    missing_nodes_for_sample: Dict[str, List[str]],
    selected_node_types: Optional[List[str]] = None,
    top_k: int = 15,
    save_path: str = "selected_node_importance_grid.png",
    mark_missing_with_star: bool = True,
    dpi: int = 600,
    figsize=(20, 12)
) -> None:

    if selected_node_types is None:
        selected_node_types = ["gene", "Promoter", "Enhancer", "transcript", "miRNA", "protein"]

    available = [
        nt for nt in selected_node_types
        if nt in data.node_types and nt in node_importance
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(available):
            ax.axis("off")
            continue

        nt = available[ax_idx]
        scores = node_importance[nt].detach().cpu()
        node_names = [str(x) for x in data[nt].node_names]
        missing_set = set(missing_nodes_for_sample.get(nt, []))

        k = min(top_k, scores.numel())
        vals, idxs = torch.topk(scores, k=k)

        chosen_names = []
        bar_colors = []

        for idx in idxs.tolist():
            node_name = node_names[idx]
            is_missing = node_name in missing_set

            label = f"{node_name}*" if (mark_missing_with_star and is_missing) else node_name
            chosen_names.append(label)

            bar_colors.append("orange" if is_missing else "steelblue")

        ax.bar(range(k), vals.numpy(), color=bar_colors)

        ax.set_xticks(range(k))
        ax.set_xticklabels(chosen_names, rotation=60, ha="right", fontsize=12)  # FIXED
        ax.set_title(nt, fontsize=16)
        ax.set_ylabel("Importance", fontsize=14)

    legend_handles = [
        Patch(facecolor="steelblue", label="Observed"),
        Patch(facecolor="orange", label="Missing"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = Path(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")  # FIXED

    plt.show()

    print(f"Saved figure to: {save_path}")
    
# def plot_training_history(
#     history: Dict[str, List[float]],
#     save_path: Optional[str] = None,
#     figsize: Tuple[int, int] = (7, 4),
#     dpi: int = 300,
#     show: bool = True,
# ) -> None:
#     """
#     Plot training curves.

#     Supports:
#     1. Single-task history:
#         - train_loss
#         - train_acc
#         - test_acc

#     2. Multitask history:
#         - train_total_loss
#         - train_classification_loss
#         - train_reconstruction_loss
#         - train_acc
#         - test_total_loss
#         - test_classification_loss
#         - test_reconstruction_loss
#         - test_acc
#     """
#     # ---------------------------------------------------------
#     # Detect history format
#     # ---------------------------------------------------------
#     if "train_loss" in history:
#         mode = "single_task"
#         n_epochs = len(history["train_loss"])
#     elif "train_total_loss" in history:
#         mode = "multitask"
#         n_epochs = len(history["train_total_loss"])
#     else:
#         raise KeyError(
#             "History format not recognized. Expected either "
#             "'train_loss' or 'train_total_loss' in history."
#         )

#     epochs = range(1, n_epochs + 1)

#     # ---------------------------------------------------------
#     # 1. Loss plot
#     # ---------------------------------------------------------
#     plt.figure(figsize=figsize)

#     if mode == "single_task":
#         plt.plot(epochs, history["train_loss"], label="Train Loss")
#         plt.title("Training Loss")
#         loss_suffix = "_loss.png"

#     else:
#         plt.plot(epochs, history["train_total_loss"], label="Train Total Loss")
#         plt.plot(
#             epochs,
#             history["train_classification_loss"],
#             label="Train Classification Loss",
#         )
#         plt.plot(
#             epochs,
#             history["train_reconstruction_loss"],
#             label="Train Reconstruction Loss",
#         )
#         plt.title("Training Loss Components")
#         loss_suffix = "_loss_components.png"

#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.tight_layout()

#     if save_path is not None:
#         loss_path = str(Path(save_path).with_name(Path(save_path).stem + loss_suffix))
#         plt.savefig(loss_path, dpi=dpi, bbox_inches="tight")

#     if show:
#         plt.show()
#     else:
#         plt.close()

#     # ---------------------------------------------------------
#     # 2. Accuracy plot
#     # ---------------------------------------------------------
#     plt.figure(figsize=figsize)
#     plt.plot(epochs, history["train_acc"], label="Train Acc")
#     plt.plot(epochs, history["test_acc"], label="Test Acc")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Training / Test Accuracy")
#     plt.legend()
#     plt.tight_layout()

#     if save_path is not None:
#         acc_path = str(
#             Path(save_path).with_name(Path(save_path).stem + "_accuracy.png")
#         )

#         acc_path_pdf = str(
#             Path(save_path).with_name(Path(save_path).stem + "_accuracy.pdf")
#         )
#         plt.savefig(acc_path, dpi=dpi, bbox_inches="tight")
#         plt.savefig(acc_path_pdf, dpi=dpi, bbox_inches="tight")

#     if show:
#         plt.show()
#     else:
#         plt.close()
        
    
# def plot_selected_node_importance_grid(
#     data: HeteroData,
#     node_importance: Dict[str, torch.Tensor],
#     missing_nodes_for_sample: Dict[str, List[str]],
#     selected_node_types: Optional[List[str]] = None,
#     top_k: int = 15,
#     save_path: str = "selected_node_importance_grid.png",
#     mark_missing_with_star: bool = True,
#     dpi: int = 300,
#     figsize=(18, 10)
# ) -> None:
#     """
#     Creates a 2x3 grid for selected node types and saves the figure.

#     Missing nodes:
#     - orange bars
#     - optional '*' appended to x tick labels

#     Observed nodes:
#     - steelblue bars
#     """
#     if selected_node_types is None:
#         selected_node_types = ["gene", "Promoter", "Enhancer", "transcript", "miRNA", "protein"]

#     # Keep only node types that actually exist in the data + importance dict
#     available = [
#         nt for nt in selected_node_types
#         if nt in data.node_types and nt in node_importance
#     ]

#     if len(available) == 0:
#         raise ValueError(
#             f"None of the requested node types were found. "
#             f"Requested={selected_node_types}, available={list(data.node_types)}"
#         )

#     # fixed 2x3 layout for presentation
#     fig, axes = plt.subplots(2, 3, figsize=figsize)
#     axes = axes.flatten()

#     for ax_idx, ax in enumerate(axes):
#         if ax_idx >= len(available):
#             ax.axis("off")
#             continue

#         nt = available[ax_idx]
#         scores = node_importance[nt].detach().cpu()
#         node_names = [str(x) for x in data[nt].node_names]
#         missing_set = set(missing_nodes_for_sample.get(nt, []))

#         k = min(top_k, scores.numel())
#         vals, idxs = torch.topk(scores, k=k)

#         chosen_names = []
#         bar_colors = []

#         for idx in idxs.tolist():
#             node_name = node_names[idx]
#             is_missing = node_name in missing_set

#             if mark_missing_with_star and is_missing:
#                 chosen_names.append(f"{node_name}*")
#             else:
#                 chosen_names.append(node_name)

#             bar_colors.append("orange" if is_missing else "steelblue")

#         ax.bar(range(k), vals.numpy(), color=bar_colors)
#         ax.set_xticks(range(k))
#         ax.set_xticklabels(chosen_names, rotation=60, ha="right", fontsize=8)
#         ax.set_title(nt, fontsize=12)
#         ax.set_ylabel("Importance")

#     legend_handles = [
#         Patch(facecolor="steelblue", label="Observed"),
#         Patch(facecolor="orange", label="Missing"),
#     ]
#     fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)

#     plt.tight_layout(rect=[0, 0, 1, 0.95])

#     save_path = str(Path(save_path))
#     save_path_pdf = save_path[:-4] + ".pdf"
#     plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
#     plt.savefig(save_path_pdf, dpi=dpi, bbox_inches="tight")
#     plt.show()

#     print(f"Saved figure to: {save_path}")