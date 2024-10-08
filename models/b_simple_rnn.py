"""
This file contains a wrapper for the MLP model. Starting simple.
"""

import torch
import torch.nn as nn

from models.layers.mlp import MLP

from utils.logger_config import logger

class SimpleRNN(nn.Module):
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
        self.input_timesteps = data_config["input_timesteps"]
        input_size = 0
        self.output_timesteps = data_config["output_timesteps"]
        output_size = coord_dims

        # get the hidden size(s)
        hidden_size = model_config["hidden_size"]
        num_layers = model_config["num_layers"]
        dropout = model_config["dropout"]

        # modify the input size in accordance with the inputs being used
        features = data_config["features"]
        p_in = features["p_in"] + 1  # neighbors plus target
        v_in = features["v_in"]  # v is same # of agents as p
        lane = features["lane"]
        positional_embeddings = features["positional_embeddings"]

        input_size += p_in * coord_dims
        input_size += v_in * coord_dims
        input_size += lane * 4  # 4: x, y, dx, dy

        # add the positional embeddings *if* they are being used
        input_size *= positional_embeddings * 2 if positional_embeddings else 1

        self.input_size = input_size
        self.hidden_size = hidden_size

        # create the recurrent network
        # TODO change to config spec: RNN, GRU, LSTM
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            device=self.device,
        )
        self.fc = nn.Linear(hidden_size, output_size, device=self.device)

        logger.debug(" Created RNN with input size: %d", input_size)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = torch.stack(x) # b x input_size

        # initialize the hidden state
        hidden = None

        outputs = []

        for t in range(self.output_timesteps):
            # get the output
            out, hidden = self.rnn(x, hidden)

            # get the last output
            out = out[:, -1, :]
            out = self.fc(out)

            # append the output
            outputs.append(out)

            # add the output to the input, replacing the first element
            out = out.unsqueeze(1)
            x = torch.cat((x, out), dim=1)
            x = x[:, 1:, :]

        # stack the outputs
        outputs = torch.stack(outputs, dim=1)

        return outputs