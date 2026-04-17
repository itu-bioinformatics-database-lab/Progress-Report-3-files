# import torch
# from typing import Dict, List, Tuple
# from torch_geometric.loader import DataLoader
# import torch.nn as nn

# def train_one_epoch(
#     model: nn.Module,
#     loader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     criterion: nn.Module,
#     device: torch.device,
# ) -> float:
#     model.train()
#     total_loss = 0.0
#     total = 0

#     for batch in loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()

#         logits = model(batch)
#         y = batch.y.view(-1)

#         loss = criterion(logits, y)
#         loss.backward()
#         optimizer.step()

#         total_loss += float(loss.item()) * y.size(0)
#         total += y.size(0)

#     return total_loss / max(total, 1)


# @torch.no_grad()
# def evaluate(
#     model: nn.Module,
#     loader: DataLoader,
#     device: torch.device,
# ) -> float:
#     model.eval()
#     correct = 0
#     total = 0

#     for batch in loader:
#         batch = batch.to(device)
#         logits = model(batch)
#         pred = logits.argmax(dim=1)
#         y = batch.y.view(-1)

#         correct += int((pred == y).sum().item())
#         total += y.size(0)

#     return correct / max(total, 1)


# def fit_model(
#     model: nn.Module,
#     train_loader: DataLoader,
#     test_loader: DataLoader,
#     device: torch.device,
#     lr: float = 1e-3,
#     epochs: int = 20,
# ) -> Tuple[nn.Module, Dict[str, List[float]]]:
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     history = {
#         "train_loss": [],
#         "train_acc": [],
#         "test_acc": [],
#     }

#     for epoch in range(1, epochs + 1):
#         train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
#         train_acc = evaluate(model, train_loader, device)
#         test_acc = evaluate(model, test_loader, device)

#         history["train_loss"].append(train_loss)
#         history["train_acc"].append(train_acc)
#         history["test_acc"].append(test_acc)

#         if epoch == 1 or epoch % 5 == 0:
#             print(
#                 f"Epoch {epoch:02d} | "
#                 f"Loss {train_loss:.4f} | "
#                 f"Train Acc {train_acc:.3f} | "
#                 f"Test Acc {test_acc:.3f}"
#             )

#     return model, history

# def train_one_epoch_multitask(
#     model,
#     loader,
#     optimizer,
#     classification_criterion,
#     device,
#     reconstruction_weight: float = 0.5,
# ):
#     model.train()
#     total_loss = 0.0
#     total_cls = 0.0
#     total_recon = 0.0
#     total = 0

#     for batch in loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()

#         output = model(batch)

#         loss_dict = compute_multitask_loss(
#             model_output=output,
#             batch=batch,
#             classification_criterion=classification_criterion,
#             reconstruction_weight=reconstruction_weight,
#         )

#         loss = loss_dict["total_loss"]
#         loss.backward()
#         optimizer.step()

#         batch_size = batch.y.view(-1).size(0)
#         total_loss += float(loss.item()) * batch_size
#         total_cls += float(loss_dict["classification_loss"].item()) * batch_size
#         total_recon += float(loss_dict["reconstruction_loss"].item()) * batch_size
#         total += batch_size

#     return {
#         "total_loss": total_loss / max(total, 1),
#         "classification_loss": total_cls / max(total, 1),
#         "reconstruction_loss": total_recon / max(total, 1),
#     }


import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from torch_geometric.loader import DataLoader

from modules.models.losses import compute_multitask_loss


# =========================================================
# Single-task training
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        y = batch.y.view(-1)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)

    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        pred = logits.argmax(dim=1)
        y = batch.y.view(-1)

        correct += int((pred == y).sum().item())
        total += y.size(0)

    return correct / max(total, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 20,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.3f} | "
                f"Test Acc {test_acc:.3f}"
            )

    return model, history


# =========================================================
# Multitask training: classification + reconstruction
# =========================================================
def train_one_epoch_multitask(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    classification_criterion: nn.Module,
    device: torch.device,
    reconstruction_weight: float = 0.5,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_recon = 0.0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        output = model(batch)

        loss_dict = compute_multitask_loss(
            model_output=output,
            batch=batch,
            classification_criterion=classification_criterion,
            reconstruction_weight=reconstruction_weight,
        )

        loss = loss_dict["total_loss"]
        loss.backward()
        optimizer.step()

        batch_size = batch.y.view(-1).size(0)

        total_loss += float(loss_dict["total_loss"].item()) * batch_size
        total_cls += float(loss_dict["classification_loss"].item()) * batch_size
        total_recon += float(loss_dict["reconstruction_loss"].item()) * batch_size
        total += batch_size

    return {
        "total_loss": total_loss / max(total, 1),
        "classification_loss": total_cls / max(total, 1),
        "reconstruction_loss": total_recon / max(total, 1),
    }


@torch.no_grad()
def evaluate_multitask(
    model: nn.Module,
    loader: DataLoader,
    classification_criterion: nn.Module,
    device: torch.device,
    reconstruction_weight: float = 0.5,
) -> Dict[str, float]:
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0
    total_cls = 0.0
    total_recon = 0.0

    for batch in loader:
        batch = batch.to(device)

        output = model(batch)
        logits = output["logits"]
        pred = logits.argmax(dim=1)
        y = batch.y.view(-1)

        loss_dict = compute_multitask_loss(
            model_output=output,
            batch=batch,
            classification_criterion=classification_criterion,
            reconstruction_weight=reconstruction_weight,
        )

        batch_size = y.size(0)

        correct += int((pred == y).sum().item())
        total += batch_size
        total_loss += float(loss_dict["total_loss"].item()) * batch_size
        total_cls += float(loss_dict["classification_loss"].item()) * batch_size
        total_recon += float(loss_dict["reconstruction_loss"].item()) * batch_size

    return {
        "accuracy": correct / max(total, 1),
        "total_loss": total_loss / max(total, 1),
        "classification_loss": total_cls / max(total, 1),
        "reconstruction_loss": total_recon / max(total, 1),
    }


def fit_model_multitask(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 20,
    reconstruction_weight: float = 0.5,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    classification_criterion = nn.CrossEntropyLoss()

    history = {
        "train_total_loss": [],
        "train_classification_loss": [],
        "train_reconstruction_loss": [],
        "train_acc": [],
        "test_total_loss": [],
        "test_classification_loss": [],
        "test_reconstruction_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_multitask(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            classification_criterion=classification_criterion,
            device=device,
            reconstruction_weight=reconstruction_weight,
        )

        train_eval = evaluate_multitask(
            model=model,
            loader=train_loader,
            classification_criterion=classification_criterion,
            device=device,
            reconstruction_weight=reconstruction_weight,
        )

        test_eval = evaluate_multitask(
            model=model,
            loader=test_loader,
            classification_criterion=classification_criterion,
            device=device,
            reconstruction_weight=reconstruction_weight,
        )

        history["train_total_loss"].append(train_metrics["total_loss"])
        history["train_classification_loss"].append(train_metrics["classification_loss"])
        history["train_reconstruction_loss"].append(train_metrics["reconstruction_loss"])
        history["train_acc"].append(train_eval["accuracy"])

        history["test_total_loss"].append(test_eval["total_loss"])
        history["test_classification_loss"].append(test_eval["classification_loss"])
        history["test_reconstruction_loss"].append(test_eval["reconstruction_loss"])
        history["test_acc"].append(test_eval["accuracy"])

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"Train Total {train_metrics['total_loss']:.4f} | "
                f"Train Cls {train_metrics['classification_loss']:.4f} | "
                f"Train Recon {train_metrics['reconstruction_loss']:.4f} | "
                f"Train Acc {train_eval['accuracy']:.3f} | "
                f"Test Total {test_eval['total_loss']:.4f} | "
                f"Test Cls {test_eval['classification_loss']:.4f} | "
                f"Test Recon {test_eval['reconstruction_loss']:.4f} | "
                f"Test Acc {test_eval['accuracy']:.3f}"
            )

    return model, history