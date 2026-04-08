# helpers.py
# This file is used to provide basic utility functions. Some (actually most) of them are reused from previous projects.
# MOST FUNCTIONS ARE MOVED TO OTHER FILES, THIS helper.oy ONLY KEEPS BASIC FUNC's NEEDED TO BE USED

from __future__ import annotations

from pathlib import Path
from typing import Any

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # More reproducible behavior for CUDA/CUDNN.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Return the device to use for training/inference

    Args:
        prefer_cuda: Whether to prefer CUDA when available

    Returns:
        torch.device("cuda") if available and preferred, otherwise torch.device("cpu")
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists

    Args
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensure the parent directory of a file path exists

    Args:
        path: File path. # Use annotations for safer calling

    Returns:
        Parent directory Path object
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent


# 
def results_dir() -> Path:
    """
    Return the results root directory and ensure it exists

    Returns:
        Path to results directory
    """
    return ensure_dir(project_root() / "results")


def models_dir() -> Path:
    """
    Return the model-output directory and ensure it exists

    Returns:
        Path to results/models directory
    """
    return ensure_dir(results_dir() / "models")


def plots_dir() -> Path:
    """
    Return the plot-output directory and ensure it exists

    Returns:
        Path to results/plots directory
    """
    return ensure_dir(results_dir() / "plots")


def tables_dir() -> Path:
    """
    Return the table-output directory and ensure it exists

    Returns:
        Path to results/tables directory
    """
    return ensure_dir(results_dir() / "tables")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters

    Args:
        model: PyTorch model
        trainable_only: If True, count only parameters with requires_grad=True

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    return sum(param.numel() for param in model.parameters())


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save a training checkpoint

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer. Can be None
        epoch: Current epoch number
        path: Destination file path
        extra: Optional extra metadata to save
    """
    path = Path(path)
    ensure_parent_dir(path)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }

    # For saving metadata if needed
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """
    Load a training checkpoint into model and optionally optimizer

    Args:
        model: PyTorch model
        path: Checkpoint file path
        optimizer: Optimizer to restore. Can be None
        map_location: torch.load map_location argument

    Returns:
        The loaded checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    return checkpoint

# NOT USED
"""
def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    reversed_edge_index = edge_index.flip(0)
    return torch.cat([edge_index, reversed_edge_index], dim=1)
"""


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Move a tensor-like object to the given device if possible. Basically just for saving data if needed...

    Args:
        data: Input object
        device: Target device

    Returns:
        Data moved to device if supported, otherwise unchanged
    """
    if hasattr(data, "to"):
        return data.to(device)
    return data


def project_root() -> Path:
    """
    Return the project root directory based on this file's location

    Returns:
        Path to project root
    """
    return Path(__file__).resolve().parents[2]