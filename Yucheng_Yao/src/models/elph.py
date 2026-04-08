# elph.py
# This file provides the ELPH model for link prediction.
# Includes a toy and a paper standard ELPH

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

from ..utils.features import build_structural_features, build_layer_edge_features
from ..utils.sketches import (
    initialize_hll,
    initialize_minhash,
    propagate_hll,
    propagate_minhash,
)


class ELPHNodeEncoder(nn.Module):
    """
    Node encoder for the first-stage ELPH implementation.

    This encoder learns node embeddings from node features and graph structure.
    Structural features from sketches are handled separately and later fused
    at the edge prediction stage.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the node encoder.

        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden dimension of the first GCN layer
            out_channels: Output node embedding dimension
            dropout: Dropout probability applied between layers
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Encode node features into node embeddings

        Args
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity tensor of shape [2, num_edges]
        Returns:
            Node embedding matrix of shape [num_nodes, out_channels]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class ELPHLinkPredictor(nn.Module):
    """
    Link predictor for ELPH.

    This module combines learned node-pair features with sketch-based structural
    features and produces a link prediction logit.
    """

    def __init__(
        self,
        node_emb_channels: int,
        structural_feature_dim: int,
        hidden_channels: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the link predictor

        Args:
            node_emb_channels: Dimension of node embeddings
            structural_feature_dim: Dimension of structural feature vectors
            hidden_channels: Hidden dimension of the prediction MLP
            dropout: Dropout probability used in the MLP
        """
        super().__init__()

        pair_feature_dim = node_emb_channels * 4
        input_dim = pair_feature_dim + structural_feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    @staticmethod
    def build_pair_features(z_src: Tensor, z_dst: Tensor) -> Tensor:
        """
        Build pairwise node features for edge prediction

        Args:
            z_src: Source node embeddings of shape [batch_size, dim]
            z_dst: Destination node embeddings of shape [batch_size, dim]

        Returns:
            Pairwise feature tensor of shape [batch_size, 4 * dim]
        """
        hadamard = z_src * z_dst
        abs_diff = torch.abs(z_src - z_dst)
        return torch.cat([z_src, z_dst, hadamard, abs_diff], dim=-1)

    def forward(self, z_src: Tensor, z_dst: Tensor, structural_features: Tensor) -> Tensor:
        """
        Predict link logits from node embeddings and structural features

        Args:
            z_src: Source node embeddings of shape [batch_size, dim]
            z_dst: Destination node embeddings of shape [batch_size, dim]
            structural_features: Structural feature tensor of shape [batch_size, structural_feature_dim]
        Returns:
            Logit tensor of shape [batch_size]
        """
        pair_features = self.build_pair_features(z_src, z_dst)
        predictor_input = torch.cat([pair_features, structural_features], dim=-1)
        logits = self.mlp(predictor_input).view(-1)
        return logits


class ELPH(nn.Module):
    """
    Final ELPH model for link prediction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        emb_channels: int,
        predictor_hidden_channels: int,
        num_hops: int = 2,
        minhash_num_perm: int = 128,
        hll_p: int = 8,
        dropout: float = 0.0,
        use_log_features: bool = False,
    ) -> None:
        """
        Initialize the ELPH model

        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden dimension of the node encoder
            emb_channels: Output node embedding dimension
            predictor_hidden_channels: Hidden dimension of the link predictor MLP
            num_hops: Number of sketch propagation hops
            minhash_num_perm: Number of MinHash permutations
            hll_p: HyperLogLog precision parameter, where number of registers is 2^p
            dropout: Dropout probability
            use_log_features: Whether to apply log1p scaling to structural features
        """
        super().__init__()

        self.num_hops = num_hops
        self.minhash_num_perm = minhash_num_perm
        self.hll_p = hll_p
        self.use_log_features = use_log_features

        self.encoder = ELPHNodeEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=emb_channels,
            dropout=dropout,
        )

        structural_feature_dim = (num_hops + 1) ** 2 + 2 * num_hops

        self.predictor = ELPHLinkPredictor(
            node_emb_channels=emb_channels,
            structural_feature_dim=structural_feature_dim,
            hidden_channels=predictor_hidden_channels,
            dropout=dropout,
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Encode node features into node embeddings

        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity tensor of shape [2, num_edges]
        Returns:
            Node embedding matrix of shape [num_nodes, emb_channels]
        """
        return self.encoder(x, edge_index)

    def build_sketch_hops(self, num_nodes: int, edge_index: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """
        Build multi-hop MinHash and HyperLogLog sketches.

        Args:
            num_nodes: Number of nodes in the graph
            edge_index: Graph connectivity tensor of shape [2, num_edges]
        Returns:
            A tuple:
                - minhash_hops
                - hll_hops
        """
        minhash_0 = initialize_minhash(
            num_nodes=num_nodes,
            num_perm=self.minhash_num_perm,
        ).to(edge_index.device)

        hll_0 = initialize_hll(
            num_nodes=num_nodes,
            p=self.hll_p,
        ).to(edge_index.device)

        minhash_hops = propagate_minhash(
            minhash_sketches=minhash_0,
            edge_index=edge_index,
            num_hops=self.num_hops,
        )

        hll_hops = propagate_hll(
            hll_sketches=hll_0,
            edge_index=edge_index,
            num_hops=self.num_hops,
        )

        return minhash_hops, hll_hops

    def build_structural_features(
        self,
        edge_label_index: Tensor,
        minhash_hops: list[Tensor],
        hll_hops: list[Tensor],
    ) -> Tensor:
        """
        Build structural feature vectors for candidate edges.

        Args:
            edge_label_index: Candidate edge tensor of shape [2, num_edges]
            minhash_hops: List of MinHash sketch tensors [hop0, ..., hopK]
            hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        Returns:
            Structural feature tensor of shape [num_edges, structural_feature_dim]
        """
        structural_features = build_structural_features(
            edge_index=edge_label_index,
            minhash_hops=minhash_hops,
            hll_hops=hll_hops,
            num_hops=self.num_hops,
            include_a=True,
            include_b=True,
        )

        if self.use_log_features:
            structural_features = torch.log1p(structural_features)

        return structural_features

    def decode(self, z: Tensor, edge_label_index: Tensor, structural_features: Tensor) -> Tensor:
        """
        Predict link logits for candidate edges.

        Args:
            z: Node embedding matrix of shape [num_nodes, emb_channels]
            edge_label_index: Candidate edge tensor of shape [2, num_edges]
            structural_features: Structural feature tensor of shape [num_edges, structural_feature_dim]
        Returns:
            Logit tensor of shape [num_edges]
        """
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        return self.predictor(z_src, z_dst, structural_features)

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor) -> Tensor:
        """
        Run a full forward pass of ELPH.

        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity tensor of shape [2, num_edges]
            edge_label_index: Candidate edge tensor of shape [2, num_edges_to_score]
        Returns:
            Logit tensor of shape [num_edges_to_score]
        """
        num_nodes = x.size(0)

        z = self.encode(x, edge_index)
        minhash_hops, hll_hops = self.build_sketch_hops(num_nodes, edge_index)
        structural_features = self.build_structural_features(
            edge_label_index=edge_label_index,
            minhash_hops=minhash_hops,
            hll_hops=hll_hops,
        )

        return self.decode(z, edge_label_index, structural_features)
    

# Final Version is following content
class ELPHEdgeAwareLayerLog1p(nn.Module):
    """
    Edge-aware message passing layer for a more faithful ELPH implementation.
    Bug Fixed.
    Adding Log1p.
    """

    def __init__(
        self,
        node_channels: int,
        edge_feature_dim: int,
        message_hidden_channels: int,
        update_hidden_channels: int,
        dropout: float = 0.0,
        use_log_edge_features: bool = True,
    ) -> None:
        """
        Initialize the edge-aware message passing layer

        Args:
            node_channels: Dimension of node features
            edge_feature_dim: Dimension of layer-specific edge features
            message_hidden_channels: Hidden dimension of the message MLP
            update_hidden_channels: Hidden dimension of the update MLP
            dropout: Dropout probability used in the MLPs
            use_log_edge_features: Whether to apply log1p scaling to edge features before message computation
        """
        super().__init__()

        self.use_log_edge_features = use_log_edge_features

        message_input_dim = 2 * node_channels + edge_feature_dim
        update_input_dim = 2 * node_channels

        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, message_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(message_hidden_channels, node_channels),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(update_input_dim, update_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(update_hidden_channels, node_channels),
        )

    def compute_messages(
        self,
        x_src: Tensor,
        x_dst: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        """
        Compute edge messages

        Args:
            x_src: Source node features of shape [num_edges, node_channels]
            x_dst: Destination node features of shape [num_edges, node_channels]
            edge_features: Edge feature tensor of shape [num_edges, edge_feature_dim]
        Returns:
            Message tensor of shape [num_edges, node_channels]
        """
        if self.use_log_edge_features:
            edge_features = torch.log1p(edge_features)

        message_input = torch.cat([x_src, x_dst, edge_features], dim=-1)
        messages = self.message_mlp(message_input)
        return messages

    def aggregate_messages(
        self,
        messages: Tensor,
        dst_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """
        Aggregate messages at destination nodes using sum aggregation

        Args:
            messages: Message tensor of shape [num_edges, node_channels]
            dst_index: Destination node indices of shape [num_edges]
            num_nodes: Total number of nodes
        Returns:
            Aggregated message tensor of shape [num_nodes, node_channels]
        """
        aggregated = torch.zeros(
            (num_nodes, messages.size(1)),
            dtype=messages.dtype,
            device=messages.device,
        )

        aggregated.index_add_(0, dst_index, messages)
        return aggregated

    def update_nodes(self, x: Tensor, aggregated_messages: Tensor) -> Tensor:
        """
        Update node representations using current node states and aggregated messages

        Args:
            x: Current node feature tensor of shape [num_nodes, node_channels]
            aggregated_messages: Aggregated message tensor of shape [num_nodes, node_channels]
        Returns
            Updated node feature tensor of shape [num_nodes, node_channels]
        """
        update_input = torch.cat([x, aggregated_messages], dim=-1)
        updated_x = self.update_mlp(update_input)
        return updated_x

    def forward(self, x: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        """
        Run one edge-aware message passing step

        Args:
            x: Node feature tensor of shape [num_nodes, node_channels]
            edge_index: Graph connectivity tensor of shape [2, num_edges]
            edge_features: Edge feature tensor of shape [num_edges, edge_feature_dim]
        Returns:
            Updated node feature tensor of shape [num_nodes, node_channels]
        """
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")

        src_index, dst_index = edge_index

        if edge_features.size(0) != edge_index.size(1):
            raise ValueError(
                "edge_features must have the same number of rows as the number of edges."
            )

        x_src = x[src_index]
        x_dst = x[dst_index]

        messages = self.compute_messages(x_src, x_dst, edge_features)
        aggregated_messages = self.aggregate_messages(
            messages=messages,
            dst_index=dst_index,
            num_nodes=x.size(0),
        )
        updated_x = self.update_nodes(x, aggregated_messages)

        return updated_x
    


class ELPHEdgeAwareEncoder(nn.Module):
    """
    Edge-aware encoder for a more faithful ELPH implementation.
    CHANGING:
    This encoder performs multi-layer message passing where each layer uses
    sketch-derived, layer-specific edge features. Compared with the baseline
    ELPH encoder, this version more closely follows the paper's idea that
    edge features should explicitly participate in message passing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_hops: int,
        message_hidden_channels: int,
        update_hidden_channels: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the edge-aware encoder

        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden node embedding dimension used throughout the encoder
            num_hops: Number of edge-aware message passing layers
            message_hidden_channels: Hidden dimension of the message MLP in each layer
            update_hidden_channels: Hidden dimension of the update MLP in each layer
            dropout: Dropout probability used in the edge-aware layers
        """
        super().__init__()

        self.num_hops = num_hops
        self.hidden_channels = hidden_channels

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        for layer_idx in range(1, num_hops + 1):
            edge_feature_dim = 2 * layer_idx + 2  # A-column + A-row + B_src + B_dst

            layer = ELPHEdgeAwareLayerLog1p(
                node_channels=hidden_channels,
                edge_feature_dim=edge_feature_dim,
                message_hidden_channels=message_hidden_channels,
                update_hidden_channels=update_hidden_channels,
                dropout=dropout,
            )

            self.layers.append(layer)

    def project_input(self, x: Tensor) -> Tensor:
        """
        Project raw node features into the hidden embedding space

        Args:
            x: Node feature tensor of shape [num_nodes, in_channels]
        Returns:
            Projected node feature tensor of shape [num_nodes, hidden_channels]
        """
        return self.input_proj(x)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        minhash_hops: list[Tensor],
        hll_hops: list[Tensor],
    ) -> Tensor:
        """
        Run the edge-aware encoder

        Args:
            x: Node feature tensor of shape [num_nodes, in_channels]
            edge_index: Graph connectivity tensor of shape [2, num_edges]
            minhash_hops: List of MinHash sketch tensors [hop0, ..., hopK]
            hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        Returns:
            Encoded node embeddings of shape [num_nodes, hidden_channels]
        """
        h = self.project_input(x)

        for layer_idx, layer in enumerate(self.layers, start=1):
            layer_edge_features = build_layer_edge_features(
                edge_index=edge_index,
                minhash_hops=minhash_hops,
                hll_hops=hll_hops,
                layer_idx=layer_idx,
                num_hops=self.num_hops,
                include_b=True,
            )

            h = layer(
                x=h,
                edge_index=edge_index,
                edge_features=layer_edge_features,
            )

        return h
    


class ELPHEdgeAware(nn.Module):
    """
    More faithful edge-aware ELPH model for link prediction.
    CHANGING:
    This model preserves the sketch-based structural feature construction
    of ELPH and additionally uses layer-specific edge features inside the
    node encoder through edge-aware message passing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        predictor_hidden_channels: int,
        num_hops: int = 2,
        minhash_num_perm: int = 128,
        hll_p: int = 8,
        message_hidden_channels: int = 64,
        update_hidden_channels: int = 64,
        dropout: float = 0.0,
        use_log_features: bool = False,
    ) -> None:
        """
        Initialize the edge-aware ELPH model

        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden dimension of node embeddings
            predictor_hidden_channels: Hidden dimension of the final link predictor MLP
            num_hops: Number of sketch propagation hops and edge-aware message passing layers
            minhash_num_perm: Number of MinHash permutations
            hll_p: HyperLogLog precision parameter, where the number of registers is 2^p
            message_hidden_channels: Hidden dimension of the message MLP in each edge-aware layer
            update_hidden_channels: Hidden dimension of the update MLP in each edge-aware layer
            dropout: Dropout probability used in encoder and predictor components
            use_log_features: Whether to apply log1p scaling to the final full structural 
                features before link prediction.
        """
        super().__init__()

        self.num_hops = num_hops
        self.minhash_num_perm = minhash_num_perm
        self.hll_p = hll_p
        self.use_log_features = use_log_features

        self.encoder = ELPHEdgeAwareEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_hops=num_hops,
            message_hidden_channels=message_hidden_channels,
            update_hidden_channels=update_hidden_channels,
            dropout=dropout,
        )

        structural_feature_dim = (num_hops + 1) ** 2 + 2 * num_hops

        self.predictor = ELPHLinkPredictor(
            node_emb_channels=hidden_channels,
            structural_feature_dim=structural_feature_dim,
            hidden_channels=predictor_hidden_channels,
            dropout=dropout,
        )

    def build_sketch_hops(
        self,
        num_nodes: int,
        edge_index: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Build multi-hop MinHash and HyperLogLog sketches

        Args:
            num_nodes: Number of nodes in the graph
            edge_index: Graph connectivity tensor of shape [2, num_edges]
        Returns:
            A tuple:
                - minhash_hops
                - hll_hops
        """
        minhash_0 = initialize_minhash(
            num_nodes=num_nodes,
            num_perm=self.minhash_num_perm,
        ).to(edge_index.device)

        hll_0 = initialize_hll(
            num_nodes=num_nodes,
            p=self.hll_p,
        ).to(edge_index.device)

        minhash_hops = propagate_minhash(
            minhash_sketches=minhash_0,
            edge_index=edge_index,
            num_hops=self.num_hops,
        )

        hll_hops = propagate_hll(
            hll_sketches=hll_0,
            edge_index=edge_index,
            num_hops=self.num_hops,
        )

        return minhash_hops, hll_hops

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        minhash_hops: list[Tensor],
        hll_hops: list[Tensor],
    ) -> Tensor:
        """
        Encode node features into node embeddings using the edge-aware encoder.

        Args:
            x: Node feature tensor of shape [num_nodes, in_channels]
            edge_index: Graph connectivity tensor of shape [2, num_edges]
            minhash_hops: List of MinHash sketch tensors [hop0, ..., hopK]
            hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        Returns:
            Node embedding tensor of shape [num_nodes, hidden_channels]
        """
        return self.encoder(
            x=x,
            edge_index=edge_index,
            minhash_hops=minhash_hops,
            hll_hops=hll_hops,
        )

    def build_full_structural_features(
        self,
        edge_label_index: Tensor,
        minhash_hops: list[Tensor],
        hll_hops: list[Tensor],
    ) -> Tensor:
        """
        Build the full structural feature vector for candidate edges

        Args:
            edge_label_index: Candidate edge tensor of shape [2, num_edges_to_score]
            minhash_hops: List of MinHash sketch tensors [hop0, ..., hopK]
            hll_hops: List of HLL sketch tensors [hop0, ..., hopK]
        Returns:
            Structural feature tensor of shape
            [num_edges_to_score, (num_hops + 1)^2 + 2 * num_hops]
        """
        structural_features = build_structural_features(
            edge_index=edge_label_index,
            minhash_hops=minhash_hops,
            hll_hops=hll_hops,
            num_hops=self.num_hops,
            include_a=True,
            include_b=True,
        )

        if self.use_log_features:
            structural_features = torch.log1p(structural_features)

        return structural_features

    def decode(
        self,
        z: Tensor,
        edge_label_index: Tensor,
        structural_features: Tensor,
    ) -> Tensor:
        """
        Predict link logits for candidate edges

        Args:
            z: Node embedding tensor of shape [num_nodes, hidden_channels]
            edge_label_index: Candidate edge tensor of shape [2, num_edges_to_score]
            structural_features: Structural feature tensor of shape [num_edges_to_score, structural_feature_dim]
        Returns:
            Logit tensor of shape [num_edges_to_score]
        """
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        return self.predictor(z_src, z_dst, structural_features)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_label_index: Tensor,
    ) -> Tensor:
        """
        Run a full forward pass of the edge-aware ELPH model.

        Args:
            x: Node feature tensor of shape [num_nodes, in_channels]
            edge_index: Graph connectivity tensor of shape [2, num_edges]
            edge_label_index: Candidate edge tensor of shape [2, num_edges_to_score]
        Returns:
            Logit tensor of shape [num_edges_to_score]
        """
        num_nodes = x.size(0)

        minhash_hops, hll_hops = self.build_sketch_hops(
            num_nodes=num_nodes,
            edge_index=edge_index,
        )

        z = self.encode(
            x=x,
            edge_index=edge_index,
            minhash_hops=minhash_hops,
            hll_hops=hll_hops,
        )

        structural_features = self.build_full_structural_features(
            edge_label_index=edge_label_index,
            minhash_hops=minhash_hops,
            hll_hops=hll_hops,
        )

        return self.decode(
            z=z,
            edge_label_index=edge_label_index,
            structural_features=structural_features,
        )