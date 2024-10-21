"""
This file contains a wrapper for the MLP model. Starting simple.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo

from models.lanes.lane_encoder import LaneEncoder
from models.lanes.lane_preprocess import LanePreprocess

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

        # get the hidden size(s)
        hidden_size = model_config["hidden_size"]
        num_layers = model_config["num_layers"]
        dropout = model_config["dropout"]

        # modify the input size in accordance with the inputs being used
        features = data_config["features"]
        p_in = features["p_in"] + 1  # neighbors plus target
        v_in = features["v_in"]  # v is same # of agents as p
        # lane = features["lane"]
        self.positional_embeddings = features["positional_embeddings"]

        input_size += p_in * coord_dims
        input_size += v_in * coord_dims
        # input_size += lane * 2  # 4: x, y, dx, dy

        # add the positional embeddings *if* they are being used
        input_size *= (
            self.positional_embeddings * 2
            if self.positional_embeddings > 0
            else 1
        )

        input_size += 128

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
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2, device=self.device)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size, device=self.device)
        self.fc3 = nn.Linear(hidden_size, coord_dims, device=self.device)

        self.lane_preprocess = LanePreprocess()
        self.lane_encoder = LaneEncoder()
        self.lane_encoder.cuda()

        logger.debug(
            "\n Created RNN: \n\t Input size: %d \n\t Device: %s \n\t Parameters: %d",
            input_size,
            self.device,
            sum(p.numel() for p in self.parameters()),
        )

    @torch.compile()
    def get_positional_embeddings(self, x):
        """
        Get the positional embeddings for the input vector.
        """
        if self.positional_embeddings == 0:
            return x

        x_positional = torch.zeros(x.shape[0], x.shape[1], 0, device=x.device)
        for i in range(self.positional_embeddings):
            s = torch.sin(2 ** (i) * np.pi * x)
            c = torch.cos(2 ** (i) * np.pi * x)

            if x.is_cuda:
                s = s.cuda()
                c = c.cuda()

            x_positional = torch.cat((x_positional, s), dim=2)
            x_positional = torch.cat((x_positional, c), dim=2)

        # get rid of any gradient tracking
        x_positional = x_positional.detach()

        # change to device
        x_positional = x_positional.to(self.device)

        return x_positional

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x, lanes, _ = x

        # to make better use of parallelism, we preprocess lanes belonging
        # to input timesteps as part of the transformation pipeline
        # therefore, we just unpack the lanes here
        lanes, lanes_t = lanes
        lane_embeddings, ortho_loss = self.lane_encoder(x, lanes)

        # get the positional embeddings
        x = self.get_positional_embeddings(x)

        # combine x and lanes_f
        x = torch.cat((x, lane_embeddings), dim=2)

        # initialize the hidden state
        hidden = None

        outputs = []

        for _ in range(self.output_timesteps):
            # get the output
            x_t, hidden = self.rnn(x, hidden)

            # get the last output
            # x_t = x_t[:, -1, :]
            x_t = hidden[-1]
            x_t = F.leaky_relu(self.fc1(x_t))  # b x coord dims
            x_t = F.leaky_relu(self.fc2(x_t))
            x_t = self.fc3(x_t)

            # append the output
            outputs.append(x_t)

            # add the output to the input, replacing the first element
            x_t = x_t.unsqueeze(1)

            # get the positional embeddings
            x_t = self.get_positional_embeddings(x_t)

            # now we need to update the lane information
            lanes, lanes_t = self.lane_preprocess(x_t, lanes_t)
            lane_embeddings, o_loss = self.lane_encoder(x_t, lanes)

            x = torch.cat((x_t, lane_embeddings), dim=2)
            ortho_loss += o_loss

            x = x.detach()

        # stack the outputs
        outputs = torch.stack(outputs, dim=1)

        return outputs, ortho_loss
