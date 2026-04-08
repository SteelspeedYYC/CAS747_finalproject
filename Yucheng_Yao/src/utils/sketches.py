# sketches.py
# This file provides sketching utilities (MH and HLL tools) for ELPH and BUDDY.
# Reused several previous helper func's, I did not put them into helpers since they only will be used here once
# -----
# MAYBE SPLIT helpers.py INTO SEVERAL FILES
# -----

from __future__ import annotations

import torch
from torch import Tensor
from typing import List


def _is_power_of_two(value: int) -> bool:
    """
    Check whether a positive integer is a power of two.
    """
    return value > 0 and (value & (value - 1)) == 0


def _rho_from_hash_values(hash_values: Tensor, max_width: int = 32) -> Tensor:
    """
    Compute the position of the first 1-bit for positive integer hash values

    Args:
        hash_values: Tensor of non-negative integer hash values
        max_width: Bit width used for the hash suffix
    Returns:
        Tensor of rho values with the same shape as hash_values
    """
    hash_values = hash_values.long()

    rho = torch.ones_like(hash_values, dtype=torch.long)
    is_zero = hash_values == 0

    # For non-zero values:
    # bit_length = floor(log2(x)) + 1
    # leading_zeros = max_width - bit_length
    # rho = leading_zeros + 1

    non_zero_values = hash_values[~is_zero]
    if non_zero_values.numel() > 0:
        bit_length = torch.floor(torch.log2(non_zero_values.float())).long() + 1
        leading_zeros = max_width - bit_length
        rho_non_zero = leading_zeros + 1
        rho[~is_zero] = rho_non_zero

    # For zero values, use max_width + 1 as a safe upper bound.

    rho[is_zero] = max_width + 1

    return rho


def generate_minhash_seeds(num_perm: int, base_seed: int = 42) -> Tensor:
    """
    Generate seeds for MinHash permutations

    Args:
        num_perm: Number of MinHash permutations
        base_seed: Base random seed
    Returns:
        Tensor of shape - seeds
    """
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    seeds = torch.randint(
        low=1,
        high=2**31 - 1,
        size=(num_perm,),
        generator=generator,
        dtype=torch.long,
    )
    return seeds


def initialize_minhash(num_nodes: int, num_perm: int, base_seed: int = 42) -> Tensor:
    """
    Initialize MinHash sketches for single-node sets

    Args:
        num_nodes: Number of nodes
        num_perm: Number of MinHash permutations
        base_seed: Random seed for generating permutation seeds
    Returns:
        MinHash sketch tensor of shape [num_nodes, num_perm]
    """
    node_ids = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1)  # [N, 1]
    seeds = generate_minhash_seeds(num_perm, base_seed=base_seed).unsqueeze(0)  # [1, P]

    # A lightweight hash family:
    # h_i(u) = ((u + 1) * a_i + b_i) mod large_prime
    a = seeds
    b = (seeds * 31 + 17) % 2147483647
    large_prime = 2147483647

    sketches = ((node_ids + 1) * a + b) % large_prime
    return sketches.long()


def initialize_hll(num_nodes: int, p: int = 8, base_seed: int = 42) -> Tensor:
    """
    Initialize HyperLogLog sketches for single-node sets

    Args:
        num_nodes: Number of nodes
        p: HLL precision parameter, number of registers m = 2^p
        base_seed: Random seed for hashing
    Returns:
        HLL sketch tensor of shape [num_nodes, 2^p]
    """
    if p <= 0:
        raise ValueError("p must be positive.")

    m = 2**p
    node_ids = torch.arange(num_nodes, dtype=torch.long)

    # A lightweight 64-bit style integer hash surrogate.
    seeds = generate_minhash_seeds(1, base_seed=base_seed)
    a = seeds[0]
    b = (a * 131 + 19) % 2147483647
    large_prime = 2147483647

    hashed = ((node_ids + 1) * a + b) % large_prime  # [N]

    register_idx = (hashed % m).long()
    suffix = torch.div(hashed, m, rounding_mode="floor").long()

    rho = _rho_from_hash_values(suffix, max_width=32)

    sketches = torch.zeros((num_nodes, m), dtype=torch.long)
    sketches[torch.arange(num_nodes), register_idx] = rho

    return sketches


def minhash_union(sketch_a: Tensor, sketch_b: Tensor) -> Tensor:
    """
    Union operator for MinHash sketches using elementwise minimum
    """
    return torch.minimum(sketch_a, sketch_b)


def hll_union(sketch_a: Tensor, sketch_b: Tensor) -> Tensor:
    """
    Union operator for HyperLogLog sketches using elementwise maximum
    """
    return torch.maximum(sketch_a, sketch_b)


def hamming_similarity(sketch_a: Tensor, sketch_b: Tensor) -> Tensor:
    """
    Compute MinHash Hamming similarity, which estimates Jaccard similarity

    Args:
        sketch_a: A features in paper (tensor of shape)
        sketch_b: B features in paper (tensor of shape)
    Returns:
        Similarity (still tensor of shape)
    """
    if sketch_a.shape != sketch_b.shape:
        raise ValueError("MinHash sketches must have the same shape.")

    return (sketch_a == sketch_b).float().mean(dim=-1)


# NEED TO BE FIXED
def estimate_hll_cardinality(hll_sketch: Tensor) -> Tensor:
    """
    Estimate set cardinality from HyperLogLog sketches

    Args:
        hll_sketch: Tensor of shape [..., m], m is #reg
    Returns:
        Estimated cardinality tensor of shape [...]
    """
    if hll_sketch.dim() < 1:
        raise ValueError("hll_sketch must have at least one dimension.")

    m = hll_sketch.size(-1)
    if not _is_power_of_two(m):
        raise ValueError("The number of HLL registers must be a power of two.")

    if m == 16:
        alpha_m = 0.673
    elif m == 32:
        alpha_m = 0.697
    elif m == 64:
        alpha_m = 0.709
    else:
        alpha_m = 0.7213 / (1.0 + 1.079 / m)

    # FIXED PART for small range correction
    registers = hll_sketch.float()
    # Raw HLL estimate
    harmonic_sum = torch.sum(torch.pow(2.0, -registers), dim=-1)
    raw_estimate = alpha_m * (m**2) / harmonic_sum

    # Small range correction using linear counting
    zero_registers = torch.sum(registers == 0, dim=-1).float()

    estimate = raw_estimate.clone()

    small_range_mask = (raw_estimate <= (2.5 * m)) & (zero_registers > 0)

    if torch.any(small_range_mask):
        estimate[small_range_mask] = (
            m * torch.log(torch.tensor(float(m), device=registers.device) / zero_registers[small_range_mask])
        )

    return estimate


def estimate_intersection_size(
    minhash_a: Tensor,
    minhash_b: Tensor,
    hll_a: Tensor,
    hll_b: Tensor,
) -> Tensor:
    """
    Estimate intersection size between two sets represented by sketches.

    Args:
        minhash_a: MinHash sketch for set A, shape [..., num_perm]
        minhash_b: MinHash sketch for set B, shape [..., num_perm]
        hll_a: HLL sketch for set A, shape [..., m]
        hll_b: HLL sketch for set B, shape [..., m]
    Returns:
        Estimated intersection size tensor of shape [...]
    """
    jaccard_est = hamming_similarity(minhash_a, minhash_b)
    union_hll = hll_union(hll_a, hll_b)
    union_cardinality = estimate_hll_cardinality(union_hll)
    return jaccard_est * union_cardinality


def propagate_minhash_once(minhash_sketches: Tensor, edge_index: Tensor) -> Tensor:
    """
    Propagate MinHash sketches by one hop using elementwise minimum over neighbors.

    Args:
        minhash_sketches: Tensor of shape [num_nodes, num_perm]
        edge_index: Graph connectivity of shape [2, num_edges]
    Returns:
        Updated MinHash sketches of shape [num_nodes, num_perm]
    """
    num_nodes, num_perm = minhash_sketches.shape
    src, dst = edge_index

    out = minhash_sketches.clone()

    for node in range(num_nodes):
        neighbor_mask = dst == node
        neighbor_src = src[neighbor_mask]

        if neighbor_src.numel() > 0:
            neighbor_sketches = minhash_sketches[neighbor_src]
            aggregated = torch.min(neighbor_sketches, dim=0).values
            out[node] = aggregated

    return out


def propagate_hll_once(hll_sketches: Tensor, edge_index: Tensor) -> Tensor:
    """
    Propagate HLL sketches by one hop using elementwise maximum over neighbors

    Args:
        hll_sketches: Tensor of shape [num_nodes, num_registers]
        edge_index: Graph connectivity of shape [2, num_edges]
    Returns:
        Updated HLL sketches of shape [num_nodes, num_registers]
    """
    num_nodes, num_registers = hll_sketches.shape
    src, dst = edge_index

    out = hll_sketches.clone()

    for node in range(num_nodes):
        neighbor_mask = dst == node
        neighbor_src = src[neighbor_mask]

        if neighbor_src.numel() > 0:
            neighbor_sketches = hll_sketches[neighbor_src]
            aggregated = torch.max(neighbor_sketches, dim=0).values
            out[node] = aggregated

    return out


def propagate_minhash(
    minhash_sketches: Tensor,
    edge_index: Tensor,
    num_hops: int,
) -> List[Tensor]:
    """
    Propagate MinHash sketches for multiple hops

    Args:
        minhash_sketches: Initial sketches of shape [num_nodes, num_perm]
        edge_index: Graph connectivity of shape [2, num_edges]
        num_hops: Number of propagation hops
    Returns:
        List of sketches:
            [hop0, hop1, ..., hopK]
    """
    if num_hops < 0:
        raise ValueError("num_hops must be non-negative.")

    sketches = [minhash_sketches]
    current = minhash_sketches

    for _ in range(num_hops):
        current = propagate_minhash_once(current, edge_index)
        sketches.append(current)

    return sketches


def propagate_hll(
    hll_sketches: Tensor,
    edge_index: Tensor,
    num_hops: int,
) -> List[Tensor]:
    """
    Propagate HLL sketches for multiple hops

    Args:
        hll_sketches: Initial sketches of shape [num_nodes, num_registers]
        edge_index: Graph connectivity of shape [2, num_edges]
        num_hops: Number of propagation hops
    Returns:
        List of sketches:
            [hop0, hop1, ..., hopK]
    """
    if num_hops < 0:
        raise ValueError("num_hops must be non-negative.")

    sketches = [hll_sketches]
    current = hll_sketches

    for _ in range(num_hops):
        current = propagate_hll_once(current, edge_index)
        sketches.append(current)

    return sketches