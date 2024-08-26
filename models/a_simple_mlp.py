"""
This file contains a wrapper for the MLP model. Starting simple.
"""

import torch
import torch.nn as nn

import logging

from models.layers.mlp import MLP

logger = logging.getLogger(__name__)

class SimpleMLP(nn.Module):
    """
    A wrapper for the MLP that .
    """

    def __init__(self, model_config: dict, data_config: dict):
        """
        Constructor for the SimpleMLP class.

        args:
            model_config (dict): dictionary containing the model configuration.
            data_config (dict): dictionary containing the data configuration.
        """

        super().__init__()
        self.device = model_config["device"]

        # want to use an n-dimensional space just in case :)
        coord_dims = data_config["coord_dims"]

        # get the number of outputs the mlp should have
        input_timesteps = data_config["input_timesteps"]
        input_size = 0
        output_timesteps = data_config["output_timesteps"]
        output_size = output_timesteps * coord_dims

        # get the hidden size(s)
        hidden_size = model_config["hidden_size"]

        # modify the input size in accordance with the inputs being used
        features = data_config["features"]
        p_in = features["p_in"] + 1 # neighbors plus target
        v_in = features["v_in"] # v is same # of agents as p
        lane = features["lane"]
        positional_embeddings = features["positional_embeddings"]

        input_size += p_in * coord_dims * input_timesteps
        input_size += v_in * coord_dims * input_timesteps
        input_size += lane * 4 # 4: x, y, dx, dy

        # add the positional embeddings *if* they are being used
        input_size *= positional_embeddings * 2 if positional_embeddings else 1

        # create the mlp
        self.mlp = MLP(input_size, hidden_size, output_size).float()

        self.mlp.to(self.device)

        print(f"Created MLP with input size: {input_size}")

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = torch.stack(x)

        x = x.float() # only one input, despite collate returning tensor

        outputs = self.mlp(x)

        return outputs
