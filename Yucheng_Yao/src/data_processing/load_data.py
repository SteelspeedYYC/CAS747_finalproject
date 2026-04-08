# load_data.py
# This file provides dataset loading functions, for specific info about DATASETS used here, plz check .../data/urls.txt.

from __future__ import annotations

from pathlib import Path
from typing import cast

from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid


SUPPORTED_DATASETS = {"cora", "pubmed", "collab"}


def _normalize_dataset_name(name: str) -> str:
    """
    Normalize dataset name for internal use.

    Args:
        name: Raw dataset name.

    Returns:
        Lowercase normalized dataset name.
    """
    normalized = name.strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{name}'. "
            f"Supported datasets: {sorted(SUPPORTED_DATASETS)}"
        )
    return normalized


def _get_planetoid_display_name(name: str) -> str:
    """
    Convert normalized name to the display name expected by Planetoid.

    Args:
        name: Normalized dataset name.

    Returns:
        Dataset name used by torch_geometric.datasets.Planetoid.
    """
    mapping = {
        "cora": "Cora",
        "pubmed": "PubMed",
    }
    return mapping[name]


def load_planetoid_dataset(name: str, root: str | Path) -> InMemoryDataset:
    """
    Load a Planetoid dataset.

    Supported:
        - Cora
        - Pubmed

    Args:
        name: Dataset name.
        root: Root folder for dataset storage.

    Returns:
        Loaded Planetoid dataset object.
    """
    normalized = _normalize_dataset_name(name)
    if normalized not in {"cora", "pubmed"}:
        raise ValueError(f"'{name}' is not a Planetoid dataset.")

    display_name = _get_planetoid_display_name(normalized)
    return Planetoid(root=str(Path(root)), name=display_name)


def load_ogb_dataset(name: str, root: str | Path) -> PygLinkPropPredDataset:
    """
    Load an OGB link prediction dataset.

    Supported:
        - Collab (ogbl-collab)

    Args:
        name: Dataset name.
        root: Root folder for dataset storage.

    Returns:
        Loaded OGB dataset object.
    """
    normalized = _normalize_dataset_name(name)
    if normalized != "collab":
        raise ValueError(f"'{name}' is not an OGB dataset.")

    return PygLinkPropPredDataset(name="ogbl-collab", root=str(Path(root)))


def load_dataset(name: str, root: str | Path = "data/raw") -> InMemoryDataset | PygLinkPropPredDataset:
    """
    Load a supported dataset by name.

    Args:
        name: Dataset name. Supported: 'Cora', 'Pubmed', 'Collab'
        root: Root folder for dataset storage.

    Returns:
        Loaded dataset object.
    """
    normalized = _normalize_dataset_name(name)

    if normalized in {"cora", "pubmed"}:
        return load_planetoid_dataset(normalized, root)

    if normalized == "collab":
        return load_ogb_dataset(normalized, root)

    raise ValueError(f"Unsupported dataset: {name}")


def get_data_object(name: str, root: str | Path = "data/raw") -> Data:
    """
    Return the single graph data object for a supported dataset.

    Args:
        name: Dataset name.
        root: Root folder for dataset storage.

    Returns:
        PyG Data object.
    """
    dataset = load_dataset(name, root)
    return cast(Data, dataset[0])