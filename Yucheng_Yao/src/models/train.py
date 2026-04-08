# train.py
# This file provides training functions.

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from ..utils.helpers import *
from ..evaluation.evaluate import evaluate_split, evaluate_split_buddy


# This function should be sperately since ELPH and BUDDY may use a different version
def _get_data_tensors(data: Any, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Extract commonly used tensors from a PyG-style data object

    Args:
        data: PyG-style data object
        device: Target device
    Returns:
        x, edge_index, edge_label_index, edge_label on the target device
    """
    required_attrs = ["x", "edge_index", "edge_label_index", "edge_label"]
    for attr in required_attrs:
        if not hasattr(data, attr):
            raise AttributeError(f"Input data is missing required attribute: '{attr}'")

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_label_index = data.edge_label_index.to(device)
    edge_label = data.edge_label.to(device).float()

    return x, edge_index, edge_label_index, edge_label


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> float:
    """
    Train the model for one epoch on the training split

    Args:
        model: Link prediction model
        optimizer: Optimizer instance
        train_data: Training split data object
        device: Training device
        criterion: Loss function. Defaults to BCEWithLogitsLoss
    Returns:
        Average training loss for the epoch
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.train()
    optimizer.zero_grad()

    x, edge_index, edge_label_index, edge_label = _get_data_tensors(train_data, device)

    logits = model(x, edge_index, edge_label_index)
    loss = criterion(logits, edge_label)

    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data: Any,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> float:
    """
    Compute the loss on a validation or test split

    Args:
        model: Link prediction model
        data: Validation/test split data object
        device: Evaluation device
        criterion: Loss function. Defaults to BCEWithLogitsLoss
    Returns:
        Loss value on the given split
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.eval()

    x, edge_index, edge_label_index, edge_label = _get_data_tensors(data, device)

    logits = model(x, edge_index, edge_label_index)
    loss = criterion(logits, edge_label)

    return float(loss.item())


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    val_data: Any | None,
    device: torch.device,
    epochs: int,
    criterion: nn.Module | None = None,
    verbose: bool = True,
    patience: int | None = None,
    checkpoint_path: str | Path | None = None,
    restore_best_model: bool = True,
    monitor: str = "val_loss",
    monitor_hits_k: int = 100,
) -> dict[str, Any]:
    """
    Full training loop.
    Ver2

    Args:
        model: Link prediction model.
        optimizer: Optimizer instance.
        train_data: Training split data object.
        val_data: Validation split data object. Can be None.
        device: Training device.
        epochs: Number of epochs.
        criterion: Loss function. Defaults to BCEWithLogitsLoss.
        verbose: Whether to display progress.
        patience: Early stopping patience based on the monitored metric.
        checkpoint_path: Path to save the best checkpoint.
        restore_best_model: Whether to restore the best model weights at the end.
        monitor: Metric used for model selection. Supported values:
            "val_loss"
            "val_auc"
            "val_ap"
            "val_hits@K" where K is given by monitor_hits_k
        monitor_hits_k: K used when monitor is "val_hits@K".
    Returns:
        A history dictionary containing loss curves and best-model information
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "best_val_loss": None,
        "best_monitor_value": None,
        "monitor": monitor,
        "epochs_ran": 0,
        "stopped_early": False,
    }

    epoch_range = range(1, epochs + 1)
    progress_bar = tqdm(epoch_range, desc="Training", leave=True) if verbose else None

    best_state_dict: dict[str, Tensor] | None = None
    best_epoch: int | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if monitor == "val_loss":
        best_monitor_value = float("inf")
        monitor_mode = "min"
    elif monitor in {"val_auc", "val_ap", "val_hits@K"}:
        best_monitor_value = float("-inf")
        monitor_mode = "max"
    else:
        raise ValueError(
            "Unsupported monitor. Use one of: "
            "'val_loss', 'val_auc', 'val_ap', 'val_hits@K'."
        )

    iterator = progress_bar if progress_bar is not None else epoch_range

    for epoch in iterator:
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            device=device,
            criterion=criterion,
        )
        history["train_loss"].append(train_loss)

        val_loss: float | None = None
        current_monitor_value: float | None = None

        if val_data is not None:
            val_loss = evaluate_loss(
                model=model,
                data=val_data,
                device=device,
                criterion=criterion,
            )
            history["val_loss"].append(val_loss)

            if monitor == "val_loss":
                current_monitor_value = val_loss
            else:
                hits_ks = [monitor_hits_k] if monitor == "val_hits@K" else None
                val_metrics = evaluate_split(
                    model=model,
                    data=val_data,
                    device=device,
                    criterion=criterion,
                    hits_ks=hits_ks,
                )

                if monitor == "val_auc":
                    current_monitor_value = val_metrics["auc"]
                elif monitor == "val_ap":
                    current_monitor_value = val_metrics["ap"]
                else:
                    current_monitor_value = val_metrics[f"hits@{monitor_hits_k}"]

            improved = (
                current_monitor_value < best_monitor_value
                if monitor_mode == "min"
                else current_monitor_value > best_monitor_value
            )

            if improved:
                best_monitor_value = float(current_monitor_value)
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

                if checkpoint_path is not None:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        path=checkpoint_path,
                        extra={
                            "best_val_loss": best_val_loss,
                            "best_epoch": best_epoch,
                            "best_monitor_value": best_monitor_value,
                            "monitor": monitor,
                        },
                    )
            else:
                epochs_without_improvement += 1

        if progress_bar is not None:
            postfix_kwargs = {"train_loss": f"{train_loss:.4f}"}
            if val_loss is not None:
                postfix_kwargs["val_loss"] = f"{val_loss:.4f}"
            if current_monitor_value is not None:
                postfix_kwargs[monitor] = f"{current_monitor_value:.4f}"
                postfix_kwargs["best"] = f"{best_monitor_value:.4f}"
            progress_bar.set_postfix(postfix_kwargs)

        history["epochs_ran"] = epoch

        if val_data is not None and patience is not None:
            if epochs_without_improvement >= patience:
                history["stopped_early"] = True
                break

    if val_data is not None:
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        history["best_monitor_value"] = best_monitor_value

        if restore_best_model and best_state_dict is not None:
            model.load_state_dict(best_state_dict)

    return history


def _get_data_tensors_buddy(data: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Extract edge_label_index and edge_label for BUDDY-style models.

    Args:
        data: PyG-style split data object
        device: Target device
    Returns:
        edge_label_index, edge_label on the target device
    """
    required_attrs = ["edge_label_index", "edge_label"]
    for attr in required_attrs:
        if not hasattr(data, attr):
            raise AttributeError(f"Input data is missing required attribute: '{attr}'")

    edge_label_index = data.edge_label_index.to(device)
    edge_label = data.edge_label.to(device).float()

    return edge_label_index, edge_label


def train_one_epoch_buddy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    buddy_cache: dict[str, Any],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> float:
    """
    Train a BUDDY model for one epoch on the training split

    Args:
        model: BUDDY link prediction model
        optimizer: Optimizer instance
        train_data: Training split data object
        buddy_cache: Precomputed BUDDY cache dictionary
        device: Training device
        criterion: Loss function. Defaults to BCEWithLogitsLoss
    Returns:
        Average training loss for the epoch
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.train()
    optimizer.zero_grad()

    edge_label_index, edge_label = _get_data_tensors_buddy(train_data, device)

    logits = model(buddy_cache, edge_label_index)
    loss = criterion(logits, edge_label)

    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate_loss_buddy(
    model: nn.Module,
    data: Any,
    buddy_cache: dict[str, Any],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> float:
    """
    Compute the loss on a validation or test split for a BUDDY model

    Args:
        model: BUDDY link prediction model
        data: Validation/test split data object
        buddy_cache: Precomputed BUDDY cache dictionary
        device: Evaluation device
        criterion: Loss function. Defaults to BCEWithLogitsLoss
    Returns:
        Loss value on the given split
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.eval()

    edge_label_index, edge_label = _get_data_tensors_buddy(data, device)

    logits = model(buddy_cache, edge_label_index)
    loss = criterion(logits, edge_label)

    return float(loss.item())


def fit_buddy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    val_data: Any | None,
    buddy_cache: dict[str, Any],
    device: torch.device,
    epochs: int,
    criterion: nn.Module | None = None,
    verbose: bool = True,
    patience: int | None = None,
    checkpoint_path: str | Path | None = None,
    restore_best_model: bool = True,
    monitor: str = "val_loss",
    monitor_hits_k: int = 100,
) -> dict[str, Any]:
    """
    Full training loop for BUDDY with optional early stopping and checkpoint saving.
    Ver2

    Args:
        model: BUDDY link prediction model.
        optimizer: Optimizer instance.
        train_data: Training split data object.
        val_data: Validation split data object. Can be None.
        buddy_cache: Precomputed BUDDY cache dictionary.
        device: Training device.
        epochs: Number of epochs.
        criterion: Loss function. Defaults to BCEWithLogitsLoss.
        verbose: Whether to display progress.
        patience: Early stopping patience based on the monitored metric.
        checkpoint_path: Path to save the best checkpoint.
        restore_best_model: Whether to restore the best model weights at the end.
        monitor: Metric used for model selection. Supported values:
            "val_loss"
            "val_auc"
            "val_ap"
            "val_hits@K" where K is given by monitor_hits_k
        monitor_hits_k: K used when monitor is "val_hits@K"

    Returns:
        A history dictionary containing loss curves and best-model information.
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "best_val_loss": None,
        "best_monitor_value": None,
        "monitor": monitor,
        "epochs_ran": 0,
        "stopped_early": False,
    }

    epoch_range = range(1, epochs + 1)
    progress_bar = tqdm(epoch_range, desc="Training BUDDY", leave=True) if verbose else None

    best_state_dict: dict[str, Tensor] | None = None
    best_epoch: int | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if monitor == "val_loss":
        best_monitor_value = float("inf")
        monitor_mode = "min"
    elif monitor in {"val_auc", "val_ap", "val_hits@K"}:
        best_monitor_value = float("-inf")
        monitor_mode = "max"
    else:
        raise ValueError(
            "Unsupported monitor. Use one of: "
            "'val_loss', 'val_auc', 'val_ap', 'val_hits@K'."
        )

    iterator = progress_bar if progress_bar is not None else epoch_range

    for epoch in iterator:
        train_loss = train_one_epoch_buddy(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            buddy_cache=buddy_cache,
            device=device,
            criterion=criterion,
        )
        history["train_loss"].append(train_loss)

        val_loss: float | None = None
        current_monitor_value: float | None = None

        if val_data is not None:
            val_loss = evaluate_loss_buddy(
                model=model,
                data=val_data,
                buddy_cache=buddy_cache,
                device=device,
                criterion=criterion,
            )
            history["val_loss"].append(val_loss)

            if monitor == "val_loss":
                current_monitor_value = val_loss
            else:
                hits_ks = [monitor_hits_k] if monitor == "val_hits@K" else None
                val_metrics = evaluate_split_buddy(
                    model=model,
                    data=val_data,
                    buddy_cache=buddy_cache,
                    device=device,
                    criterion=criterion,
                    hits_ks=hits_ks,
                )

                if monitor == "val_auc":
                    current_monitor_value = val_metrics["auc"]
                elif monitor == "val_ap":
                    current_monitor_value = val_metrics["ap"]
                else:
                    current_monitor_value = val_metrics[f"hits@{monitor_hits_k}"]

            improved = (
                current_monitor_value < best_monitor_value
                if monitor_mode == "min"
                else current_monitor_value > best_monitor_value
            )

            if improved:
                best_monitor_value = float(current_monitor_value)
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

                if checkpoint_path is not None:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        path=checkpoint_path,
                        extra={
                            "best_val_loss": best_val_loss,
                            "best_epoch": best_epoch,
                            "best_monitor_value": best_monitor_value,
                            "monitor": monitor,
                        },
                    )
            else:
                epochs_without_improvement += 1

        if progress_bar is not None:
            postfix_kwargs = {"train_loss": f"{train_loss:.4f}"}
            if val_loss is not None:
                postfix_kwargs["val_loss"] = f"{val_loss:.4f}"
            if current_monitor_value is not None:
                postfix_kwargs[monitor] = f"{current_monitor_value:.4f}"
                postfix_kwargs["best"] = f"{best_monitor_value:.4f}"
            progress_bar.set_postfix(postfix_kwargs)

        history["epochs_ran"] = epoch

        if val_data is not None and patience is not None:
            if epochs_without_improvement >= patience:
                history["stopped_early"] = True
                break

    if val_data is not None:
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        history["best_monitor_value"] = best_monitor_value

        if restore_best_model and best_state_dict is not None:
            model.load_state_dict(best_state_dict)

    return history