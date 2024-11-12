"""
A module used to generate encodings for a list of lanes.
"""

import torch.nn as nn

from models.lanes.resnet import ResNet

from utils.logger_config import logger

class LaneEncoder(nn.Module):
    """
    A module that manages how the lanes get encoded.
    """

    def __init__(self, lane_config):
        """
        Initializes the LaneEncoder module.
        """
        super().__init__()

        self.angle_filter = True
        self.distance_filter = True
        self.embedding_size = lane_config["embedding_size"]

        self.ortho_lambda = 0.001

        num_points = lane_config["num_points"]

        # . TODO will eventually want to use some sort of network
        # to evaluate which lanes are important, but for now the general
        # direction will have to suffice

        # . TODO possibly use zero padding to begin with for optimization

        # self.pointnet = PointNet(num_points, 4, self.embedding_size)  # 4 input dims

        self.resnet = ResNet(self.embedding_size)

    def forward(self, x, lanes):
        """
        Encodes the lanes.

        args:
            x (torch.Tensor): the position of the agent
                batches x timesteps x dims
            lanes (list): the lanes to encode
                list of torch.Tensors: batches x lanes x dims

        returns:
            torch.Tensor: the encoded lanes
                batches x timesteps x encoded_dims
        """

        # TODO improve the way per model reshaping is done

        # since lanes are irrelevant to time, just flatten them into batches
        # b, t, p, d = lanes.shape
        # lanes = lanes.view(b * t, d, p)  # reordering d and p

        b, t, n, m, d = lanes.shape
        lanes = lanes.view(b * t, n, m, d)

        if x.is_cuda:
            lanes = lanes.cuda()

        # track gradients only for the pointnet call
        lanes = lanes.detach()

        # logger.debug(lanes.shape)

        embeddings = self.resnet(lanes)
        ortho_loss = 0

        # get the embeddings
        # embeddings, ortho_loss = self.pointnet(lanes)

        # convert back to batchsize x timesteps x embedddings
        embeddings = embeddings.view(b, t, -1)

        return embeddings, ortho_loss * self.ortho_lambda
