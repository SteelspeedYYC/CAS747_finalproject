# features.py
# This file provides structural feature construction utilities for ELPH and BUDDY.

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, Tuple
from src.utils.sketches import estimate_hll_cardinality, estimate_intersection_size


def _validate_hop_lists(minhash_hops: list[Tensor], hll_hops: list[Tensor], num_hops: int) -> None:
    """
    Validate hop sketch lists.

    Args:
        minhash_hops: List of MinHash sketch tensors [hop0, hop1, ..., hopK]
        hll_hops: List of HLL sketch tensors [hop0, hop1, ..., hopK]
        num_hops: Maximum hop count K
    """
    expected_len = num_hops + 1

    if len(minhash_hops) != expected_len:
        raise ValueError(
            f"Expected {expected_len} MinHash hop tensors, got {len(minhash_hops)}."
        )

    if len(hll_hops) != expected_len:
        raise ValueError(
            f"Expected {expected_len} HLL hop tensors, got {len(hll_hops)}."
        )

    for d in range(expected_len):
        if minhash_hops[d].size(0) != hll_hops[d].size(0):
            raise ValueError(f"Node count mismatch at hop {d}.")


def estimate_node_cardinalities(hll_hops: list[Tensor], num_hops: int) -> list[Tensor]:
    """
    Estimate |N^d(u)| for all nodes and all hops

    Args:
        hll_hops: List of HLL sketch tensors [hop0, hop1, ..., hopK]
        num_hops: Maximum hop count K
    Returns:
        A list of cardinality tensors: [card_0, card_1, ..., card_K]
    """
    if len(hll_hops) != num_hops + 1:
        raise ValueError("The length of hll_hops must be num_hops + 1.")

    cardinalities = [estimate_hll_cardinality(hll_hops[d]) for d in range(num_hops + 1)]
    return cardinalities


def estimate_pairwise_intersections(
    edge_index: Tensor,
    minhash_hops: list[Tensor],
    hll_hops: list[Tensor],
    num_hops: int,
) -> Dict[Tuple[int, int], Tensor]:
    """
    Estimate pairwise intersection sizes for all hop pairs
    For each (du, dv), estimates: |N^du(u) ∩ N^dv(v)|

    Args:
        edge_index: Candidate edge pairs of shape [2, num_edges]
        minhash_hops: List of MinHash sketch tensors [hop0, ..., hopK]
        hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        num_hops: Maximum hop count K
    Returns:
        A dictionary mapping (du, dv) -> Tensor
    """
    _validate_hop_lists(minhash_hops, hll_hops, num_hops)

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    src, dst = edge_index
    intersections: Dict[Tuple[int, int], Tensor] = {}

    for du in range(num_hops + 1):
        for dv in range(num_hops + 1):
            intersections[(du, dv)] = estimate_intersection_size(
                minhash_a=minhash_hops[du][src],
                minhash_b=minhash_hops[dv][dst],
                hll_a=hll_hops[du][src],
                hll_b=hll_hops[dv][dst],
            )

    return intersections


def estimate_a_features(
    edge_index: Tensor,
    minhash_hops: list[Tensor],
    hll_hops: list[Tensor],
    num_hops: int,
) -> Tensor:
    """
    Estimate A-hat features for each candidate edge

    Args:
        edge_index: Candidate edge pairs of shape [2, num_edges]
        minhash_hops: List of MinHash sketch tensors [hop0, ..., hopK]
        hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        num_hops: Maximum hop count K
    Returns:
        Tensor of shape [num_edges, (K+1)*(K+1)] containing flattened A-hat features
        Flattening order is row-major over (du, dv)
    """
    intersections = estimate_pairwise_intersections(
        edge_index=edge_index,
        minhash_hops=minhash_hops,
        hll_hops=hll_hops,
        num_hops=num_hops,
    )

    num_edges = edge_index.size(1)
    a_hat = torch.zeros(
        (num_edges, num_hops + 1, num_hops + 1),
        dtype=torch.float32,
        device=edge_index.device,
    )

    for du in range(num_hops + 1):
        for dv in range(num_hops + 1):
            current = intersections[(du, dv)].float()

            if du == 0 and dv == 0:
                a_hat[:, du, dv] = current
                continue

            subtract_term = torch.zeros_like(current)

            for x in range(du + 1):
                for y in range(dv + 1):
                    if x == du and y == dv:
                        continue
                    subtract_term = subtract_term + a_hat[:, x, y]

            a_hat[:, du, dv] = torch.clamp(current - subtract_term, min=0.0)

    return a_hat.view(num_edges, -1)


def estimate_b_features(
    edge_index: Tensor,
    hll_hops: list[Tensor],
    a_hat_flat: Tensor,
    num_hops: int,
) -> Tensor:
    """
    Estimate B-hat features for each candidate edge

    Args:
        edge_index: Candidate edge pairs of shape [2, num_edges]
        hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        a_hat_flat: Flattened A-hat features of shape [num_edges, (K+1)*(K+1)]
        num_hops: Maximum hop count K

    Returns:
        Tensor of shape [num_edges, 2 * num_hops]
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    src, dst = edge_index
    num_edges = edge_index.size(1)

    a_hat = a_hat_flat.view(num_edges, num_hops + 1, num_hops + 1)
    cardinalities = estimate_node_cardinalities(hll_hops, num_hops)

    b_src = torch.zeros((num_edges, num_hops), dtype=torch.float32, device=edge_index.device)
    b_dst = torch.zeros((num_edges, num_hops), dtype=torch.float32, device=edge_index.device)

    prev_b_src = torch.zeros(num_edges, dtype=torch.float32, device=edge_index.device)
    prev_b_dst = torch.zeros(num_edges, dtype=torch.float32, device=edge_index.device)

    for d in range(1, num_hops + 1):
        card_src_d = cardinalities[d][src].float()
        card_dst_d = cardinalities[d][dst].float()

        a_block_sum = a_hat[:, : d + 1, : d + 1].sum(dim=(1, 2))

        current_b_src = torch.clamp(card_src_d - prev_b_src - a_block_sum, min=0.0)
        current_b_dst = torch.clamp(card_dst_d - prev_b_dst - a_block_sum, min=0.0)

        b_src[:, d - 1] = current_b_src
        b_dst[:, d - 1] = current_b_dst

        prev_b_src = current_b_src
        prev_b_dst = current_b_dst

    return torch.cat([b_src, b_dst], dim=1)


def build_structural_features(
    edge_index: Tensor,
    minhash_hops: list[Tensor],
    hll_hops: list[Tensor],
    num_hops: int,
    include_a: bool = True,
    include_b: bool = True,
) -> Tensor:
    """
    Build the structural feature vector used by ELPH/BUDDY

    Args:
        edge_index: Candidate edge pairs of shape [2, num_edges]
        minhash_hops: List of MinHash sketch tensors [hop0, ..., hopK]
        hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        num_hops: Maximum hop count K
        include_a: Whether to include A-hat features
        include_b: Whether to include B-hat features
    Returns:
        Structural feature tensor of shape [num_edges, feature_dim]
    """
    if not include_a and not include_b:
        raise ValueError("At least one of include_a or include_b must be True.")

    features = []

    a_hat_flat: Tensor | None = None
    if include_a or include_b:
        a_hat_flat = estimate_a_features(
            edge_index=edge_index,
            minhash_hops=minhash_hops,
            hll_hops=hll_hops,
            num_hops=num_hops,
        )

    if include_a:
        assert a_hat_flat is not None
        features.append(a_hat_flat)

    if include_b:
        assert a_hat_flat is not None
        b_hat = estimate_b_features(
            edge_index=edge_index,
            hll_hops=hll_hops,
            a_hat_flat=a_hat_flat,
            num_hops=num_hops,
        )
        features.append(b_hat)

    return torch.cat(features, dim=1)

## Edge-aware MP features
def build_layer_edge_features(
    edge_index: Tensor,
    minhash_hops: list[Tensor],
    hll_hops: list[Tensor],
    layer_idx: int,
    num_hops: int,
    include_b: bool = True,
) -> Tensor:
    """
    Build layer-specific edge features for edge-aware ELPH message passing.

    Args:
        edge_index: Candidate edge tensor of shape [2, num_edges]
        minhash_hops: List of MinHash sketch tensors [hop0, ..., hopK]
        hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        layer_idx: Current layer index l. Must satisfy 1 <= layer_idx <= num_hops.
        num_hops: Maximum hop count K
        include_b: Whether to include the B_hat_uv[layer_idx] term
    Returns:
        Layer-specific edge feature tensor of shape [num_edges, feature_dim]
    """
    if layer_idx < 1 or layer_idx > num_hops:
        raise ValueError(
            f"layer_idx must satisfy 1 <= layer_idx <= num_hops, "
            f"got layer_idx={layer_idx}, num_hops={num_hops}."
        )

    a_hat_flat = estimate_a_features(
        edge_index=edge_index,
        minhash_hops=minhash_hops,
        hll_hops=hll_hops,
        num_hops=num_hops,
    )

    num_edges = edge_index.size(1)
    a_hat = a_hat_flat.view(num_edges, num_hops + 1, num_hops + 1)

    feature_parts = []

    # A_hat[du, layer_idx] for du < layer_idx
    a_column = a_hat[:, :layer_idx, layer_idx]
    if a_column.numel() > 0:
        feature_parts.append(a_column)

    # A_hat[layer_idx, dv] for dv < layer_idx
    a_row = a_hat[:, layer_idx, :layer_idx]
    if a_row.numel() > 0:
        feature_parts.append(a_row)

    if include_b:
        b_hat = estimate_b_features(
            edge_index=edge_index,
            hll_hops=hll_hops,
            a_hat_flat=a_hat_flat,
            num_hops=num_hops,
        )

        # B_hat is concatenated as:
        # [B_src(1..K), B_dst(1..K)]
        b_src_l = b_hat[:, layer_idx - 1].unsqueeze(1)
        b_dst_l = b_hat[:, num_hops + (layer_idx - 1)].unsqueeze(1)

        feature_parts.append(b_src_l)
        feature_parts.append(b_dst_l)

    if len(feature_parts) == 0:
        return torch.zeros((num_edges, 0), dtype=torch.float32, device=edge_index.device)

    return torch.cat(feature_parts, dim=1)