# preprocess.py
# This file provides common preprocessing functions,
# I think BUDDY's special preprocessing should be implemented in BUDDY.py.

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected


def make_edge_index_undirected(data: Data) -> Data:
    """
    Convert graph edges to undirected format

    Args:
        data: Input PyG Data object
    Returns:
        Data object with undirected edge_index
    """
    if not hasattr(data, "edge_index"):
        raise AttributeError("Input data is missing required attribute: 'edge_index'")

    data = data.clone()
    assert data.edge_index is not None, "edge_index should not be None." # ANNOYING ERRORRRRRRRRRRRRR

    edge_index: torch.Tensor = data.edge_index
    data.edge_index = to_undirected(edge_index)

    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        data.edge_weight = None

    return data


def build_planetoid_link_splits(
    data: Data,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    neg_sampling_ratio: float = 1.0,
    is_undirected: bool = True,
    add_negative_train_samples: bool = True,
    split_labels: bool = False,
) -> tuple[Data, Data, Data]:
    """
    Build train/val/test splits for Planetoid-style datasets.

    Args:
        data: Input PyG Data object
        val_ratio: Validation edge ratio
        test_ratio: Test edge ratio
        neg_sampling_ratio: Negative-to-positive ratio
        is_undirected: Whether the graph is treated as undirected
        add_negative_train_samples: Whether to add negative train samples
        split_labels: Whether to split positive/negative labels separately
    Returns:
        train_data, val_data, test_data
    """
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=is_undirected,
        add_negative_train_samples=add_negative_train_samples,
        neg_sampling_ratio=neg_sampling_ratio,
        split_labels=split_labels,
    )
    return transform(data)


def build_ogb_link_splits(
    data: Data,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    neg_sampling_ratio: float = 1.0,
    is_undirected: bool = True,
    add_negative_train_samples: bool = True,
    split_labels: bool = False,
) -> tuple[Data, Data, Data]:
    """
    Build train/val/test splits for larger OGB-style datasets.

    This is a first-stage generic split function. Later we can replace or refine
    it with OGB-official split handling if we want stricter paper alignment

    Args:
        data: Input PyG Data object
        val_ratio: Validation edge ratio
        test_ratio: Test edge ratio
        neg_sampling_ratio: Negative-to-positive ratio
        is_undirected: Whether the graph is treated as undirected
        add_negative_train_samples: Whether to add negative train samples
        split_labels: Whether to split positive/negative labels separately
    Returns:
        train_data, val_data, test_data
    """
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=is_undirected,
        add_negative_train_samples=add_negative_train_samples,
        neg_sampling_ratio=neg_sampling_ratio,
        split_labels=split_labels,
    )
    return transform(data)


def prepare_link_prediction_data(
    dataset_name: str,
    data: Data,
) -> tuple[Data, Data, Data]:
    """
    Prepare a supported dataset for link prediction

    Args:
        dataset_name: Dataset name
        data: Raw PyG Data object
    Returns:
        train_data, val_data, test_data
    """
    normalized = dataset_name.strip().lower()
    data = make_edge_index_undirected(data)

    if normalized in {"cora", "pubmed"}:
        return build_planetoid_link_splits(data)

    if normalized == "collab":
        return build_ogb_link_splits(data)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def summarize_split(data: Data) -> dict[str, Any]:
    """
    Summarize a processed split for quick inspection/debugging

    Args:
        data: PyG split data object
    Returns:
        Dictionary of summary information
    """
    edge_index = getattr(data, "edge_index", None)
    edge_label_index = getattr(data, "edge_label_index", None)
    edge_label = getattr(data, "edge_label", None)
    num_nodes = getattr(data, "num_nodes", None)
    num_features = getattr(data, "num_features", None)

    summary: dict[str, Any] = {
        "num_nodes": int(num_nodes) if num_nodes is not None else None,
        "num_features": int(num_features) if num_features is not None else None,
        "edge_index_shape": tuple(edge_index.shape) if edge_index is not None else None,
        "edge_label_index_shape": tuple(edge_label_index.shape) if edge_label_index is not None else None,
        "edge_label_shape": tuple(edge_label.shape) if edge_label is not None else None,
    }

    if edge_label is not None:
        labels = edge_label.detach().cpu()
        summary["num_positive_labels"] = int((labels == 1).sum().item())
        summary["num_negative_labels"] = int((labels == 0).sum().item())

    return summary