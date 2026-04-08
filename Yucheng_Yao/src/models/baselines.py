# baselines.py
# This file provides baseline GNN models for link prediction.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """
    GCN encoder class
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Encode node features into node embeddings

        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format with shape [2, num_edges]
        Returns:
            Node embedding matrix of shape [num_nodes, out_channels]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return x


class MLPLinkPredictor(nn.Module):
    """
    MLP decoder for link prediction
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Pair feature used: [z_u, z_v, z_u * z_v, |z_u - z_v|]
        pair_in_channels = in_channels * 4

        self.mlp = nn.Sequential(
            nn.Linear(pair_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    @staticmethod
    def build_pair_features(z_src: Tensor, z_dst: Tensor) -> Tensor:
        """
        Construct pairwise features for a node pair.

        Args:
            z_src: Source node embeddings of shape [batch_size, dim]
            z_dst: Destination node embeddings of shape [batch_size, dim]
        Returns:
            Pair feature tensor of shape [batch_size, 4 * dim]
        """
        hadamard = z_src * z_dst
        abs_diff = torch.abs(z_src - z_dst)
        return torch.cat([z_src, z_dst, hadamard, abs_diff], dim=-1)

    def forward(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        """
        Predict link logits for node pairs.

        Args:
            z_src: Source node embeddings of shape [batch_size, dim]
            z_dst: Destination node embeddings of shape [batch_size, dim]
        Returns:
            Link logits of shape [batch_size]
        """
        pair_features = self.build_pair_features(z_src, z_dst)
        logits = self.mlp(pair_features).view(-1)
        return logits


class GCNBaseline(nn.Module):
    """
    Full baseline model for link prediction. Combining GCN encoder + MLP link predictor
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        emb_channels: int,
        predictor_hidden_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.encoder = GCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=emb_channels,
            dropout=dropout,
        )

        self.predictor = MLPLinkPredictor(
            in_channels=emb_channels,
            hidden_channels=predictor_hidden_channels,
            dropout=dropout,
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Compute node embeddings

        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
        Returns:
            Node embeddings
        """
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        """
        Predict link logits for the given node pairs

        Args:
            z: Node embedding matrix of shape [num_nodes, emb_channels]
            edge_label_index: Node-pair indices of shape [2, num_pairs]
        Returns:
            Link logits of shape [num_pairs]
        """
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        return self.predictor(z_src, z_dst)

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor) -> Tensor:
        """
        Full forward pass for link prediction

        Args:
            x: Node feature matrix.
            edge_index: Graph connectivity used for message passing
            edge_label_index: Node pairs to score
        Returns:
            Link logits
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)