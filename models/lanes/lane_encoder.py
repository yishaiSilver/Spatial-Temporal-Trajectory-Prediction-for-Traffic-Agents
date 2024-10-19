import torch
import torch.nn as nn

from models.lanes.pointnet import PointNet
from models.lanes.angle_filter import angle_filter
from models.lanes.distance_filter import distance_filter_and_pad
from models.lanes.zero_pad import zero_pad

from utils.logger_config import logger


class LaneEncoder(nn.Module):
    """
    A module that manages how the lanes get encoded.
    """

    def __init__(self):
        """
        Initializes the LaneEncoder module.
        """
        super().__init__()

        self.angle_filter = True
        self.distance_filter = True
        self.distance_threshold = 10
        self.num_padded = 20

        # . TODO will eventually want to use some sort of network
        # to evaluate which lanes are important, but for now the general
        # direction will have to suffice

        # . TODO possibly use zero padding to begin with for optimization

        self.pointnet = PointNet(4) # 4 input dims


    def add_timestep_dim(self, x, lanes):
        """
        Adds a timestep dimension to the lanes.

        args:
            x (torch.Tensor): the position of the agent
                batches x timesteps x dims
            lanes (list): the lanes to encode
                list of torch.Tensors: batches x lanes x dims

        returns:
            list: the lanes with a timestep dimension added
                list of torch.Tensors: batches x timesteps x lanes x dims
        """

        # get the number of timesteps
        timesteps = x.shape[1]

        # add a timestep dimension to the lanes
        lanes = [lane.unsqueeze(0).repeat(timesteps, 1, 1) for lane in lanes]

        return lanes
    
    def shift_lanes(self, x, lanes):
        """
        Shifts the lanes to simulate the agent moving through space.

        args:
            x (torch.Tensor): the position of the agent
                batches x timesteps x dims
            lanes (list): the lanes to encode
                list of torch.Tensors: batches x lanes x dims

        returns:
            list: the shifted lanes
                list of torch.Tensors: batches x timesteps x lanes x dims
        """

        # get the agent offsets
        agent_offsets = torch.cumsum(x, dim=1)

        shifted_lanes = []
        for i, lane in enumerate(lanes):
            agent_offsets_t = agent_offsets[i].unsqueeze(1)
            lane[:, :, :2] -= agent_offsets_t
            shifted_lanes.append(lane)

        return shifted_lanes

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

        # since angle doesn't change, do it first to minimize other points
        # considered.
        if self.angle_filter:
            lanes = angle_filter(lanes)

        # add a timestep dimension to the lanes:
        # list of batches x timesteps x lanes x dims
        lanes = self.add_timestep_dim(x, lanes)

        # shift the lanes to simulate the agent moving through space
        lanes = self.shift_lanes(x, lanes)
        final_lanes = [lane[-1] for lane in lanes]
        # FIXME can use unsqueeze here to avoid add_timestep_dim w/ minimal
        # changes

        if self.distance_filter:
            lanes = distance_filter_and_pad(
                lanes, self.distance_threshold, self.num_padded
            )
        # else:
        #     lanes = zero_pad(lanes, self.num_padded)

        # since lanes are irrelevant to time, just flatten them into batches
        b, t, p, d = lanes.shape
        lanes = lanes.view(b * t, d, p)  # reordering d and p


        # track gradients only for the pointnet call
        with torch.no_grad():
            lanes = lanes.clone()

        # get the embeddings
        embeddings, matrix = self.pointnet(lanes)

        # convert back to batchsize x timesteps x embedddings
        embeddings = embeddings.view(b, t, -1)

        return embeddings, matrix, final_lanes
