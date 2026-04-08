# buddy_helpers.py
# This file provides preprocessing and cache-building utilities for BUDDY implementation.

from __future__ import annotations

from typing import Any
import torch
from torch import Tensor
from src.utils.sketches import (
    estimate_hll_cardinality,
    initialize_hll,
    initialize_minhash,
    propagate_hll,
    propagate_minhash,
)
from src.utils.features import build_structural_features


def propagate_node_features_once(
    x: Tensor,
    edge_index: Tensor,
    aggregation: str = "mean",
) -> Tensor:
    """
    Propagate node features by one hop
    For each destination node, aggregate the features of its source neighbors
    This is used to precompute node-side propagated features for BUDDY

    Args:
        x: Node feature tensor of shape [num_nodes, num_features]
        edge_index: Graph connectivity tensor of shape [2, num_edges]
        aggregation: Aggregation method. Supported values:
            "mean"
            "sum"
    Returns:
        Propagated node feature tensor of shape [num_nodes, num_features]
    """
    if aggregation not in {"mean", "sum"}:
        raise ValueError(f"Unsupported aggregation '{aggregation}'. Use 'mean' or 'sum'.")

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    src, dst = edge_index
    num_nodes, num_features = x.shape

    out = torch.zeros(
        (num_nodes, num_features),
        dtype=x.dtype,
        device=x.device,
    )

    out.index_add_(0, dst, x[src])

    if aggregation == "mean":
        degree = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        degree = degree.clamp(min=1.0).unsqueeze(1)
        out = out / degree

    return out


def propagate_node_features(
    x: Tensor,
    edge_index: Tensor,
    num_hops: int,
    aggregation: str = "mean",
) -> list[Tensor]:
    """
    Propagate node features for multiple hops

    Args:
        x: Node feature tensor of shape [num_nodes, num_features]
        edge_index: Graph connectivity tensor of shape [2, num_edges]
        num_hops: Number of propagation hops
        aggregation: Aggregation method. Supported values:
            "mean"
            "sum"

    Returns:
        A list of propagated node feature tensors: [hop0, hop1, ..., hopK]
    """
    if num_hops < 0:
        raise ValueError("num_hops must be non-negative.")

    propagated_x_hops = [x]
    current = x

    for _ in range(num_hops):
        current = propagate_node_features_once(
            x=current,
            edge_index=edge_index,
            aggregation=aggregation,
        )
        propagated_x_hops.append(current)

    return propagated_x_hops


def estimate_cardinality_hops(hll_hops: list[Tensor], num_hops: int) -> list[Tensor]:
    """
    Estimate node neighborhood cardinalities for all hop levels

    Args:
        hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        num_hops: Maximum hop count K
    Returns:
        A list of cardinality tensors: [card_0, card_1, ..., card_K]
    """
    expected_len = num_hops + 1
    if len(hll_hops) != expected_len:
        raise ValueError(
            f"Expected {expected_len} HLL hop tensors, got {len(hll_hops)}."
        )

    cardinality_hops = [
        estimate_hll_cardinality(hll_hops[d])
        for d in range(num_hops + 1)
    ]
    return cardinality_hops


def move_buddy_cache(
    cache: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    Move all tensor-based values inside a BUDDY cache dictionary to a target device

    Args:
        cache: Cache dictionary produced by build_buddy_cache
        device: Target device
    Returns:
        A new cache dictionary with tensor values moved to the target device
    """
    moved_cache: dict[str, Any] = {}

    for key, value in cache.items():
        if isinstance(value, Tensor):
            moved_cache[key] = value.to(device)
        elif isinstance(value, list):
            moved_list = [
                item.to(device) if isinstance(item, Tensor) else item
                for item in value
            ]
            moved_cache[key] = moved_list
        else:
            moved_cache[key] = value

    return moved_cache


# FIXED
def build_buddy_cache(
    x: Tensor,
    edge_index: Tensor,
    num_hops: int,
    minhash_num_perm: int = 128,
    hll_p: int = 8,
    feature_propagation: str = "mean",
    cache_device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Build the node-side preprocessing cache for BUDDY

    Args:
        x: Node feature tensor of shape [num_nodes, num_features]
        edge_index: Graph connectivity tensor of shape [2, num_edges]
        num_hops: Number of propagation hops.
        minhash_num_perm: Number of MinHash permutations.
        hll_p: HyperLogLog precision parameter, where number of registers is 2^p
        feature_propagation: Aggregation method for node feature propagation.
            Supported values:
                "mean"
                "sum"
        cache_device: Device on which the final cache should be stored.
            If None, the cache is kept on the current device of x.
    Returns:
        A cache dictionary containing:
            "num_nodes": int
            "num_hops": int
            "propagated_x_hops": list[Tensor]
            "minhash_hops": list[Tensor]
            "hll_hops": list[Tensor]
            "cardinality_hops": list[Tensor]
    """
    if x.dim() != 2:
        raise ValueError("x must have shape [num_nodes, num_features].")

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    num_nodes = x.size(0)
    working_device = x.device

    propagated_x_hops = propagate_node_features(
        x=x,
        edge_index=edge_index,
        num_hops=num_hops,
        aggregation=feature_propagation,
    )

    minhash_0 = initialize_minhash(
        num_nodes=num_nodes,
        num_perm=minhash_num_perm,
    ).to(working_device)

    hll_0 = initialize_hll(
        num_nodes=num_nodes,
        p=hll_p,
    ).to(working_device)

    minhash_hops = propagate_minhash(
        minhash_sketches=minhash_0,
        edge_index=edge_index,
        num_hops=num_hops,
    )

    hll_hops = propagate_hll(
        hll_sketches=hll_0,
        edge_index=edge_index,
        num_hops=num_hops,
    )

    cardinality_hops = estimate_cardinality_hops(
        hll_hops=hll_hops,
        num_hops=num_hops,
    )

    cache: dict[str, Any] = {
        "num_nodes": num_nodes,
        "num_hops": num_hops,
        "propagated_x_hops": propagated_x_hops,
        "minhash_hops": minhash_hops,
        "hll_hops": hll_hops,
        "cardinality_hops": cardinality_hops,
    }

    if cache_device is not None:
        cache = move_buddy_cache(cache, cache_device)

    return cache


def build_buddy_edge_features_from_cache(
    buddy_cache: dict[str, Any],
    edge_label_index: Tensor,
    structural_use_log: bool = False,
) -> dict[str, Tensor]:
    """
    Build edge-level BUDDY inputs from a precomputed BUDDY cache.

    Args:
        buddy_cache: Cache dictionary produced by build_buddy_cache
        edge_label_index: Candidate edge tensor of shape [2, num_edges]
        structural_use_log: Whether to apply log1p scaling to structural features
    Returns:
        A dictionary containing:
            "edge_label_index": Tensor of shape [2, num_edges]
            "src_features": Tensor of shape [num_edges, node_feature_dim]
            "dst_features": Tensor of shape [num_edges, node_feature_dim]
            "pair_features": Tensor of shape [num_edges, 4 * node_feature_dim]
            "structural_features": Tensor of shape [num_edges, (num_hops + 1)^2 + 2 * num_hops]
    """
    required_keys = {
        "num_hops",
        "propagated_x_hops",
        "minhash_hops",
        "hll_hops",
    }

    missing_keys = required_keys.difference(buddy_cache.keys())
    if len(missing_keys) > 0:
        raise KeyError(f"Missing required cache keys: {sorted(missing_keys)}")

    if edge_label_index.dim() != 2 or edge_label_index.size(0) != 2:
        raise ValueError("edge_label_index must have shape [2, num_edges].")

    num_hops = int(buddy_cache["num_hops"])
    propagated_x_hops: list[Tensor] = buddy_cache["propagated_x_hops"]
    minhash_hops: list[Tensor] = buddy_cache["minhash_hops"]
    hll_hops: list[Tensor] = buddy_cache["hll_hops"]

    src, dst = edge_label_index

    final_node_features = propagated_x_hops[num_hops]
    src_features = final_node_features[src]
    dst_features = final_node_features[dst]

    hadamard = src_features * dst_features
    abs_diff = torch.abs(src_features - dst_features)
    pair_features = torch.cat([src_features, dst_features, hadamard, abs_diff], dim=-1)

    structural_features = build_structural_features(
        edge_index=edge_label_index,
        minhash_hops=minhash_hops,
        hll_hops=hll_hops,
        num_hops=num_hops,
        include_a=True,
        include_b=True,
    )

    if structural_use_log:
        structural_features = torch.log1p(structural_features)

    return {
        "edge_label_index": edge_label_index,
        "src_features": src_features,
        "dst_features": dst_features,
        "pair_features": pair_features,
        "structural_features": structural_features,
    }