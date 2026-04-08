# evaluate.py
# This file provides evaluation functions.

from __future__ import annotations
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor


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


@torch.no_grad()
def predict_logits(model: nn.Module, data: Any, device: torch.device) -> Tensor:
    """
    Predict raw logits
    """
    model.eval()
    x, edge_index, edge_label_index, _ = _get_data_tensors(data, device)
    logits = model(x, edge_index, edge_label_index)
    return logits


@torch.no_grad()
def predict_probs(model: nn.Module, data: Any, device: torch.device) -> Tensor:
    """
    Predict probabilities
    """
    logits = predict_logits(model, data, device)
    return torch.sigmoid(logits)


@torch.no_grad()
def compute_loss(
    model: nn.Module,
    data: Any,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> float:
    """
    Compute loss on a given split
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.eval()
    x, edge_index, edge_label_index, edge_label = _get_data_tensors(data, device)
    logits = model(x, edge_index, edge_label_index)
    loss = criterion(logits, edge_label)

    return float(loss.item())


def _to_numpy(tensor: Tensor) -> np.ndarray:
    """
    Convert a tensor to a NumPy array on CPU
    """
    return tensor.detach().cpu().numpy()


@torch.no_grad()
def compute_auc(model: nn.Module, data: Any, device: torch.device) -> float:
    """
    Compute ROC-AUC score.
    """
    probs = predict_probs(model, data, device)
    _, _, _, edge_label = _get_data_tensors(data, device)

    y_true = _to_numpy(edge_label)
    y_score = _to_numpy(probs)

    return float(roc_auc_score(y_true, y_score))


@torch.no_grad()
def compute_average_precision(model: nn.Module, data: Any, device: torch.device) -> float:
    """
    Compute average precision
    """
    probs = predict_probs(model, data, device)
    _, _, _, edge_label = _get_data_tensors(data, device)

    y_true = _to_numpy(edge_label)
    y_score = _to_numpy(probs)

    return float(average_precision_score(y_true, y_score))


def _split_pos_neg_scores(scores: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """
    Split scores into positive and negative subsets using binary labels

    Args:
        scores: Prediction scores of shape [num_edges]
        labels: Binary labels of shape [num_edges]
    Returns:
        pos_scores, neg_scores
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    if pos_scores.numel() == 0:
        raise ValueError("No positive samples found for Hits@K computation.")
    if neg_scores.numel() == 0:
        raise ValueError("No negative samples found for Hits@K computation.")

    return pos_scores, neg_scores


def compute_hits_at_k_from_scores(pos_scores: Tensor, neg_scores: Tensor, k: int) -> float:
    """
    Compute Hits@K from positive and negative prediction scores.

    Args:
        pos_scores: Scores for positive edges
        neg_scores: Scores for negative edges
        k: Cutoff K
    Returns:
        Hits@K score
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    num_neg = neg_scores.numel()
    effective_k = min(k, num_neg)

    topk_neg_scores = torch.topk(neg_scores, effective_k).values
    threshold = topk_neg_scores[-1]

    hits = (pos_scores > threshold).float().mean().item()
    return float(hits)


@torch.no_grad()
def compute_hits_at_k(model: nn.Module, data: Any, device: torch.device, k: int) -> float:
    """
    Compute Hits@K on a given split

    Args:
        model: Link prediction model
        data: PyG-style data object
        device: Evaluation device
        k: Cutoff K
    Returns:
        Hits@K score
    """
    probs = predict_probs(model, data, device)
    _, _, _, edge_label = _get_data_tensors(data, device)

    pos_scores, neg_scores = _split_pos_neg_scores(probs, edge_label)
    return compute_hits_at_k_from_scores(pos_scores, neg_scores, k)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    data: Any,
    device: torch.device,
    criterion: nn.Module | None = None,
    hits_ks: list[int] | None = None,
) -> dict[str, float]:
    """
    Evaluate a single data split with classification metrics and optional Hits@K.

    Args:
        model: Link prediction model
        data: PyG-style data object
        device: Evaluation device
        criterion: Loss function. Defaults to BCEWithLogitsLoss
        hits_ks: Optional list of K values for Hits@K
    Returns:
        Dictionary of evaluation metrics
    """
    metrics: dict[str, float] = {
        "loss": compute_loss(model, data, device, criterion),
        "auc": compute_auc(model, data, device),
        "ap": compute_average_precision(model, data, device),
    }

    if hits_ks is not None:
        for k in hits_ks:
            metrics[f"hits@{k}"] = compute_hits_at_k(model, data, device, k)

    return metrics


def _get_data_tensors_buddy(data: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Extract edge_label_index and edge_label for BUDDY-style evaluation.

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


@torch.no_grad()
def predict_logits_buddy(
    model: nn.Module,
    data: Any,
    buddy_cache: dict[str, Any],
    device: torch.device,
) -> Tensor:
    """
    Predict raw logits
    """
    model.eval()
    edge_label_index, _ = _get_data_tensors_buddy(data, device)
    logits = model(buddy_cache, edge_label_index)
    return logits


@torch.no_grad()
def predict_probs_buddy(
    model: nn.Module,
    data: Any,
    buddy_cache: dict[str, Any],
    device: torch.device,
) -> Tensor:
    """
    Predict probabilities
    """
    logits = predict_logits_buddy(model, data, buddy_cache, device)
    return torch.sigmoid(logits)


@torch.no_grad()
def compute_loss_buddy(
    model: nn.Module,
    data: Any,
    buddy_cache: dict[str, Any],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> float:
    """
    Compute loss on a given split
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.eval()
    edge_label_index, edge_label = _get_data_tensors_buddy(data, device)
    logits = model(buddy_cache, edge_label_index)
    loss = criterion(logits, edge_label)

    return float(loss.item())


@torch.no_grad()
def compute_auc_buddy(
    model: nn.Module,
    data: Any,
    buddy_cache: dict[str, Any],
    device: torch.device,
) -> float:
    """
    Compute ROC-AUC score
    """
    probs = predict_probs_buddy(model, data, buddy_cache, device)
    _, edge_label = _get_data_tensors_buddy(data, device)

    y_true = _to_numpy(edge_label)
    y_score = _to_numpy(probs)

    return float(roc_auc_score(y_true, y_score))


@torch.no_grad()
def compute_average_precision_buddy(
    model: nn.Module,
    data: Any,
    buddy_cache: dict[str, Any],
    device: torch.device,
) -> float:
    """
    Compute average precision
    """
    probs = predict_probs_buddy(model, data, buddy_cache, device)
    _, edge_label = _get_data_tensors_buddy(data, device)

    y_true = _to_numpy(edge_label)
    y_score = _to_numpy(probs)

    return float(average_precision_score(y_true, y_score))


@torch.no_grad()
def compute_hits_at_k_buddy(
    model: nn.Module,
    data: Any,
    buddy_cache: dict[str, Any],
    device: torch.device,
    k: int,
) -> float:
    """
    Compute Hits@K on a given split for BUDDY

    Args:
        model: BUDDY link prediction model
        data: PyG-style data object
        buddy_cache: Precomputed BUDDY cache dictionary
        device: Evaluation device
        k: Cutoff K
    Returns:
        Hits@K score
    """
    probs = predict_probs_buddy(model, data, buddy_cache, device)
    _, edge_label = _get_data_tensors_buddy(data, device)

    pos_scores, neg_scores = _split_pos_neg_scores(probs, edge_label)
    return compute_hits_at_k_from_scores(pos_scores, neg_scores, k)


@torch.no_grad()
def evaluate_split_buddy(
    model: nn.Module,
    data: Any,
    buddy_cache: dict[str, Any],
    device: torch.device,
    criterion: nn.Module | None = None,
    hits_ks: list[int] | None = None,
) -> dict[str, float]:
    """
    Evaluate a single data split for BUDDY with classification metrics and optional Hits@K

    Args:
        model: BUDDY link prediction model
        data: PyG-style data object
        buddy_cache: Precomputed BUDDY cache dictionary
        device: Evaluation device
        criterion: Loss function. Defaults to BCEWithLogitsLoss
        hits_ks: Optional list of K values for Hits@K
    Returns:
        Dictionary of evaluation metrics.
    """
    metrics: dict[str, float] = {
        "loss": compute_loss_buddy(model, data, buddy_cache, device, criterion),
        "auc": compute_auc_buddy(model, data, buddy_cache, device),
        "ap": compute_average_precision_buddy(model, data, buddy_cache, device),
    }

    if hits_ks is not None:
        for k in hits_ks:
            metrics[f"hits@{k}"] = compute_hits_at_k_buddy(
                model=model,
                data=data,
                buddy_cache=buddy_cache,
                device=device,
                k=k,
            )

    return metrics