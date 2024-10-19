"""
This file contains a wrapper for the MLP model. Starting simple.
"""

import torch
import torch.nn as nn

import numpy as np

from models.lanes.pointnet import PointNet
from models.layers.mlp import MLP

from utils.logger_config import logger

class Seq2Seq(nn.Module):
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
        self.positional_embeddings = features["positional_embeddings"]

        input_size += p_in * coord_dims
        input_size += v_in * coord_dims
        # input_size += lane * 2  # 4: x, y, dx, dy

        # add the positional embeddings *if* they are being used
        input_size *= (
            self.positional_embeddings * 2 if self.positional_embeddings else 1
        )

        # input_size += 128

        # add the positional embeddings *if* they are being used
        # input_size *= positional_embeddings * 2 if positional_embeddings else 1

        self.input_size = input_size
        self.hidden_size = hidden_size

        # create the recurrent network
        # TODO change to config spec: RNN, GRU, LSTM
        self.encoder_rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            device=self.device,
        )

        self.decoder_rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            device=self.device,
        )

        self.fc = nn.Linear(hidden_size, output_size, device=self.device)

        self.pointnet = PointNet().to(self.device)

        logger.debug(" Created Seq2Seq with input size: %d", input_size)

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

        # change to device
        x_positional = x_positional.to(self.device)

        return x_positional

    @torch.compile()
    def embed_lanes(self, lanes):
        """
        takes in 2d lanes and generates an embedding vector
        """

        # convert from batchsize x timesteps x points x dims
        # to:          (batchsize * timesteps) x points x dims

        b, t, p, d = lanes.shape

        lanes = lanes.view(b * t, d, p)  # reordering d and p

        embeddings, matrix = self.pointnet(lanes)

        # convert back to batchsize x timesteps x embedddings
        embeddings = embeddings.view(b, t, -1)

        if lanes.is_cuda:
            embeddings = embeddings.cuda()

        return embeddings  # TODO return matrix, use normalization to encourage orthogonality

    @torch.compile()
    def forward(self, input):
        """
        Forward pass through the network.
        """
        x, lanes, other = input

        # lane_t = lanes[:, -1]
        # lanes_embedded = self.embed_lanes(lanes)

        # get the positional embeddings
        x = self.get_positional_embeddings(x)

        # combine x and lanes_f
        # x = torch.cat((x, lanes_embedded), dim=2)

        # encode the input
        hidden = None
        _, hidden = self.encoder_rnn(x, hidden)

        # get the last position
        x = x[:, -1, :].unsqueeze(1)

        # decode the output
        outputs = []
        for _ in range(self.output_timesteps):
            # get the output
            x_t, hidden = self.decoder_rnn(x, hidden)

            # get the last output
            x_t = x_t[:, -1, :]
            x_t = self.fc(x_t)  # b x coord dims

            # append the output
            outputs.append(x_t)

            # add the output to the input, replacing the first element
            x_t = x_t.unsqueeze(1)

            # # move lane_t by x_t, then add to the new vector
            # lane_t = lane_t - x_t
            # lane_t = lane_t.unsqueeze(1)
            # lane_t_embedded = self.embed_lanes(lane_t)

            # get the positional embeddings
            x_t = self.get_positional_embeddings(x_t)

            # x_t = torch.cat((x_t, lane_t_embedded), dim=2)
            # lane_t = lane_t.view(lane_t.size(0), 15, 2)

            # update to the new input
            x = x_t

        # stack the outputs
        outputs = torch.stack(outputs, dim=1)

        return outputs