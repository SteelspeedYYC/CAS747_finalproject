# buddy.py
# This file provides the BUDDY model for link prediction.

from __future__ import annotations

from typing import Any
from torch import Tensor, nn
from torch import cat
from src.utils.buddy_helpers import build_buddy_edge_features_from_cache


class BUDDYLinkPredictor(nn.Module):
    """
    Link predictor for BUDDY.
    """

    def __init__(
        self,
        pair_feature_dim: int,
        structural_feature_dim: int,
        hidden_channels: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the BUDDY link predictor.

        Args:
            pair_feature_dim: Dimension of pairwise node features
            structural_feature_dim: Dimension of structural features
            hidden_channels: Hidden dimension of the prediction MLP
            dropout: Dropout probability used in the MLP
        """
        super().__init__()

        input_dim = pair_feature_dim + structural_feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, pair_features: Tensor, structural_features: Tensor) -> Tensor:
        """
        Predict link logits from pairwise node features and structural features.

        Args:
            pair_features: Pairwise node feature tensor of shape [num_edges, pair_feature_dim]
            structural_features: Structural feature tensor of shape [num_edges, structural_feature_dim]

        Returns:
            Logit tensor of shape [num_edges]
        """
        predictor_input = cat([pair_features, structural_features], dim=-1)
        logits = self.mlp(predictor_input).view(-1)
        return logits


class BUDDY(nn.Module):
    """
    BUDDY model for link prediction.
    """

    def __init__(
        self,
        node_feature_dim: int,
        num_hops: int,
        predictor_hidden_channels: int,
        dropout: float = 0.0,
        structural_use_log: bool = False,
    ) -> None:
        """
        Initialize the BUDDY model

        Args:
            node_feature_dim: Dimension of the propagated node features used by BUDDY
            num_hops: Number of hops used in preprocessing and structural features
            predictor_hidden_channels: Hidden dimension of the prediction MLP
            dropout: Dropout probability used in the predictor
            structural_use_log: Whether to apply log1p scaling to structural features
                when gathering edge-level inputs from the cache
        """
        super().__init__()

        self.num_hops = num_hops
        self.structural_use_log = structural_use_log

        pair_feature_dim = 4 * node_feature_dim
        structural_feature_dim = (num_hops + 1) ** 2 + 2 * num_hops

        self.predictor = BUDDYLinkPredictor(
            pair_feature_dim=pair_feature_dim,
            structural_feature_dim=structural_feature_dim,
            hidden_channels=predictor_hidden_channels,
            dropout=dropout,
        )

    def build_edge_inputs(
        self,
        buddy_cache: dict[str, Any],
        edge_label_index: Tensor,
    ) -> dict[str, Tensor]:
        """
        Build edge-level BUDDY inputs from a precomputed cache

        Args
            buddy_cache: Cache dictionary produced by build_buddy_cache
            edge_label_index: Candidate edge tensor of shape [2, num_edges]
        Returns:
            A dictionary containing edge-level BUDDY inputs, including:
                src_features
                dst_features
                pair_features
                structural_features
        """
        return build_buddy_edge_features_from_cache(
            buddy_cache=buddy_cache,
            edge_label_index=edge_label_index,
            structural_use_log=self.structural_use_log,
        )

    def forward(
        self,
        buddy_cache: dict[str, Any],
        edge_label_index: Tensor,
    ) -> Tensor:
        """
        Run a forward pass of BUDDY

        Args:
            buddy_cache: Cache dictionary produced by build_buddy_cache
            edge_label_index: Candidate edge tensor of shape [2, num_edges]
        Returns:
            Logit tensor of shape [num_edges]
        """
        edge_inputs = self.build_edge_inputs(
            buddy_cache=buddy_cache,
            edge_label_index=edge_label_index,
        )

        pair_features = edge_inputs["pair_features"]
        structural_features = edge_inputs["structural_features"]

        return self.predictor(
            pair_features=pair_features,
            structural_features=structural_features,
        )