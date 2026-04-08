# timer.py
# This file provides timing utilities for preprocessing, training, and inference.

from __future__ import annotations

import time
import torch
from torch import nn
from typing import Any, Callable


def _sync_if_needed(device: torch.device | None = None) -> None:
    """
    Synchronize CUDA operations before or after timing when needed.

    Args:
        device: Target device. If the device is CUDA, synchronization is performed.
    """
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def time_callable(
    fn: Callable[..., Any],
    *args: Any,
    sync_device: torch.device | None = None,
    **kwargs: Any,
) -> tuple[Any, float]:
    """
    Time the execution of a callable and return both its output and elapsed time

    Args:
        fn: Callable to execute
        *args: Positional arguments passed to the callable
        device: Device used for synchronization-aware timing
        **kwargs: Keyword arguments passed to the callable
    Returns:
        A tuple:
            result: Output of the callable
            elapsed_time_s: Elapsed wall time in seconds
    """
    _sync_if_needed(sync_device)
    start_time = time.perf_counter()

    result = fn(*args, **kwargs)

    _sync_if_needed(sync_device)
    end_time = time.perf_counter()

    return result, float(end_time - start_time)


def time_buddy_preprocessing(
    build_buddy_cache_fn: Callable[..., dict[str, Any]],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_hops: int,
    minhash_num_perm: int = 128,
    hll_p: int = 8,
    feature_propagation: str = "mean",
    cache_device: torch.device | None = None,
    timer_device: torch.device | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Time BUDDY preprocessing, including cache construction

    Args:
        build_buddy_cache_fn: Function used to build the BUDDY cache
        x: Node feature tensor
        edge_index: Graph connectivity tensor
        num_hops: Number of hops for preprocessing
        minhash_num_perm: Number of MinHash permutations
        hll_p: HyperLogLog precision parameter
        feature_propagation: Node feature propagation mode
        cache_device: Target device for the final cache
        timer_device: Device used for synchronization-aware timing
    Returns:
        A tuple
            buddy_cache: Constructed BUDDY cache
            elapsed_time_s: Preprocessing wall time in seconds
    """
    buddy_cache, elapsed_time_s = time_callable(
        build_buddy_cache_fn,
        x=x,
        edge_index=edge_index,
        num_hops=num_hops,
        minhash_num_perm=minhash_num_perm,
        hll_p=hll_p,
        feature_propagation=feature_propagation,
        cache_device=cache_device,
        sync_device=timer_device,
    )
    return buddy_cache, elapsed_time_s


def time_training_epoch(
    train_one_epoch_fn: Callable[..., float],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    sync_device: torch.device,
    criterion: nn.Module | None = None,
    buddy_cache: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """
    Time one training epoch for either a standard model or a BUDDY model

    Args:
        train_one_epoch_fn: Training function for one epoch
        model: Link prediction model
        optimizer: Optimizer instance
        train_data: Training split data object
        device: Training device
        criterion: Loss function
        buddy_cache: Optional BUDDY cache dictionary
    Returns:
        A tuple:
            train_loss: Training loss returned by the epoch function
            elapsed_time_s: Training wall time in seconds
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "optimizer": optimizer,
        "train_data": train_data,
        "device": sync_device,
        "criterion": criterion,
    }

    if buddy_cache is not None:
        kwargs["buddy_cache"] = buddy_cache

    train_loss, elapsed_time_s = time_callable(
        train_one_epoch_fn,
        sync_device=sync_device,
        **kwargs,
    )
    return float(train_loss), elapsed_time_s


def time_inference_full_split(
    evaluate_split_fn: Callable[..., dict[str, float]],
    model: nn.Module,
    data: Any,
    sync_device: torch.device,
    criterion: nn.Module | None = None,
    hits_ks: list[int] | None = None,
    buddy_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, float], float]:
    """
    Time full-split inference/evaluation for either a standard model or a BUDDY model

    Args:
        evaluate_split_fn: Evaluation function for a full split
        model: Link prediction model
        data: Validation/test split data object
        device: Evaluation device
        criterion: Loss function
        hits_ks: Optional list of K values for Hits@K
        buddy_cache: Optional BUDDY cache dictionary
    Returns:
        A tuple:
            metrics: Evaluation metrics dictionary
            elapsed_time_s: Inference wall time in seconds
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "data": data,
        "device": sync_device,
        "criterion": criterion,
        "hits_ks": hits_ks,
    }

    if buddy_cache is not None:
        kwargs["buddy_cache"] = buddy_cache

    metrics, elapsed_time_s = time_callable(
        evaluate_split_fn,
        sync_device=sync_device,
        **kwargs,
    )
    return metrics, elapsed_time_s