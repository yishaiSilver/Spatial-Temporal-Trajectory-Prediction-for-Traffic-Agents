"""
A module that is used to filter out irrelevant lanes and prepare
the lanes for encoding.
"""

import torch

from models.lanes.angle_filter import angle_filter
from models.lanes.distance_filter import distance_filter_and_pad
from models.lanes.zero_pad import zero_pad

from utils.logger_config import logger


class LanePreprocess:
    """
    A module that manages how the lanes get preprocessed.
    """

    def __init__(self):
        """
        Initializes the LanePreprocess module.
        """
        self.angle_filter = True
        self.distance_filter = True
        self.distance_threshold = 10
        self.num_padded = 20

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

        # this will only be the case if the lanes have timesteps
        if len(lanes[0].shape) == 3:
            return lanes

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

    def __call__(self, x, lanes):
        """
        Preprocesses the lanes by filtering out irrelevant lanes and
        padding them.
        """

        # since angle doesn't change, do it first to minimize other points
        # considered. Only do it at the beginning, when lanes
        # don't have a timestep dimension
        if self.angle_filter and len(lanes[0].shape) == 2:
            lanes = angle_filter(lanes)

        # add a timestep dimension to the lanes:
        # list of batches x timesteps x lanes x dims
        lanes = self.add_timestep_dim(x, lanes)

        # shift the lanes to simulate the agent moving through space
        lanes = self.shift_lanes(x, lanes)

        # save the last lanes for future offsets
        final_lanes = [lane[-1].unsqueeze(0) for lane in lanes]

        # if self.distance_filter:
        lanes = distance_filter_and_pad(
            lanes, self.distance_threshold, self.num_padded
        )
        # else:
        #     lanes = zero_pad(lanes, self.num_padded)

        return lanes, final_lanes
