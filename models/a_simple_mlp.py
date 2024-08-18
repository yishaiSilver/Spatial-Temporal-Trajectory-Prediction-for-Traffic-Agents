"""
This file contains a wrapper for the MLP model. Starting simple.
"""

import torch
import torch.nn as nn

from models.layers.mlp import MLP


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
        self.p_in = features["p_in"] + 1 # neighbors plus target
        self.v_in = features["v_in"] # v is same # of agents as p
        self.lane = features["lane"]

        input_size += self.p_in * coord_dims * input_timesteps
        input_size += self.v_in * coord_dims * input_timesteps
        input_size += self.lane * 4 * input_timesteps # 4: x, y, dx, dy

        # create the mlp
        self.mlp = MLP(input_size, hidden_size, output_size).float()


    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = torch.stack(x)

        x = x.float() # only one input, despite collate returning tensor

        outputs = self.mlp(x)

        print(outputs.shape)

        return outputs
