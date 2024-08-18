import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple multi-layer perceptron.

    Methods:
        __init__(input_dim, hidden_dims, output_dim):
            Constructor for the MLP class.
        forward(x):
            Forward pass through the network.
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Constructor for the MLP class.

        args:
            input_dim (int): input dimension.
            hidden_dims (list): list of hidden dimensions.
            output_dim (int): output dimension.
        """
        super(MLP, self).__init__()
        layers = []

        dims = [input_dim] + hidden_dims + [output_dim]

        # Create the layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        # Remove the last ReLU
        layers.pop()

        # Create the network
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        args:
            x (torch.Tensor): input tensor.
        """

        return self.layers(x)
