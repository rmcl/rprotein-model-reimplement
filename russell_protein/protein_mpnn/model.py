from typing import Optional

import torch
import torch.nn as nn

from .encoding import (
    build_features_for_network_input
)


class ProteinMPNN(nn.Module):
    def __init__(
        self,
        num_letters : int,
        num_node_features : int,
        num_edge_features : int,
        num_hidden_dimensions :    int,
        num_encoder_layers : Optional[int] = 3,
        num_decoder_layers : Optional[int] = 3,
        vocab_size : Optional[int] = 21,
        k_neighbors : Optional[int] = 64,
        augment_eps : Optional[float] = 0.05,
        dropout : Optional[float] = 0.1,
    ):
        """The ProteinMPNN model

        Args:
            num_letters (int):
            num_node_features (int): The number of node features
            num_edge_features (int): The number of edge features
            num_hidden_dim (int): The hidden dimension of the model
            num_encoder_layers (Optional[int]): The number of encoder layers. Defaults to 3.
            num_decoder_layers (Optional[int]): The number of decoder layers. Defaults to 3.
            vocab (Optional[int], optional): The size of the vocabulary. Defaults to 21.
                There are 20 amino acids and 1 for ??????
            k_neighbors (Optional[int]): The number of neighbors to consider. Defaults to 64.
            augment_eps (Optional[float]): The augmentation epsilon. Defaults to 0.05.
            dropout (Optional[float]): The dropout rate. Defaults to 0.1.

        """
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_hidden_dimensions = num_hidden_dimensions
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
        self.k_neighbors = k_neighbors
        self.augment_eps = augment_eps
        self.dropout = dropout

        self.features = ProteinEncoding(
            num_edge_features=num_edge_features,
            num_rbfs=16,
            top_k=k_neighbors,
            num_positional_embeddings=16
        )

        layers = [
            nn.Linear(num_edge_features, num_hidden_dimensions, bias=True),
            nn.Embedding(vocab_size, num_hidden_dimensions)
        ]



# Think about this in another file

class PositionalEncodings(nn.Module):
    """This class takes in a tensor of offsets and outputs a tensor of positional encodings.

    ChatGPT thinks this could be done better with
    nn.Embedding(2 * max_relative_feature + 2, num_embeddings), but
    keeping the same as original for now
    """

    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


class ProteinEncoding(nn.Module):
    """Process the input data to create the node and edge features for the MPNN"""

    def __init__(
        self,
        num_edge_features,
        num_rbfs=16,
        top_k=30,
        num_positional_embeddings=16,
    ):
        super().__init__()
        self.num_rbfs = num_rbfs
        self.top_k = top_k
        self.num_positional_embeddings = num_positional_embeddings

        #num_node_in = 6 # this seems to be unused in the Protein but is used in Ca ProtEnc encoding
        num_edge_in = num_positional_embeddings + num_rbfs * 25 # what's 25?

        self.positional_embedding = PositionalEncodings(
            num_embeddings = self.num_positional_embeddings,
            max_relative_feature = 32
        )

        self.edge_embedding = nn.Linear(num_edge_in, num_edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(num_edge_features)


    def forward(self, X, mask, residue_indexes, chain_encodings):
        top_k_neighbor_indexes, rbfs, offset, E_chains = build_features_for_network_input(
            X,
            mask,
            residue_indexes,
            chain_encodings,
            self.num_rbfs,
            self.top_k
        )

        # Pass the offset and chain encodings to the positional embedding layer
        E_positional = self.positional_embedding(offset, E_chains)
        E = torch.cat((E_positional, rbfs), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, top_k_neighbor_indexes
