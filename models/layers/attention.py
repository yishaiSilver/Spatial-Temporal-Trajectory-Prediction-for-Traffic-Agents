"""
Module used for attention mechanism
"""

import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Attention mechanism used in the seq2seq model
    """

    def __init__(self, hidden_size: int, query_size: int):
        """
        Constructor for the Attention class

        Args:
            hidden_size (int): The hidden size of the attention mechanism.
            query_size (int): The size of the query tensor.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.query_size = query_size

        self.attention = nn.Sequential(
            nn.Linear(hidden_size + query_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden: torch.Tensor, query: torch.Tensor):
        """
        Forward pass for the attention mechanism.

        Args:
            hidden (torch.Tensor): The hidden state of the encoder.
            query (torch.Tensor): The query tensor.

        Returns:
            torch.Tensor: The attention weights.
        """
        # repeat the query tensor to match the hidden tensor
        query = query.unsqueeze(1).repeat(1, hidden.size(1), 1)

        # concatenate the hidden and query tensors
        combined = torch.cat((hidden, query), dim=-1)

        # calculate the attention weights
        attention_weights = self.attention(combined)

        return attention_weights

    def get_attention_weights(self, hidden: torch.Tensor, query: torch.Tensor):
        """
        Get the attention weights for the given hidden and query tensors.

        Args:
            hidden (torch.Tensor): The hidden state of the encoder.
            query (torch.Tensor): The query tensor.

        Returns:
            torch.Tensor: The attention weights.
        """
        return self.forward(hidden, query)