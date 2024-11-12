"""
A module that is used to filter out irrelevant lanes and prepare
the lanes for encoding.
"""

import torch
import numpy as np

from models.lanes.rear_filter import rear_filter
from models.lanes.distance_filter import distance_filter_and_pad
from models.lanes.angle_filter import angle_filter
from models.lanes.generate_map_matrix import generate_map


class LanePreprocess:
    """
    A module that manages how the lanes get preprocessed.
    """

    def __init__(self, lane_config):
        """
        Initializes the LanePreprocess module.
        """
        self.angle_filter = True
        self.distance_filter = True
        self.num_points = lane_config["num_points"]
        self.min_y_filter = lane_config["min_y_filter"]

    def add_timestep_dim(self, x, lanes):
        """
        Adds a timestep dimension to the lanes.

        args:
            x (torch.Tensor or np.ndarray): the position of the agent
                batches x timesteps x dims
            lanes (list): the lanes to encode
                list of torch.Tensors or np.ndarrays: batches x lanes x dims

        returns:
            list: the lanes with a timestep dimension added
                list of torch.Tensors or np.ndarrays: batches x timesteps x lanes x dims
        """

        # Get the number of timesteps from x
        timesteps = x.shape[1]

        # Add a timestep dimension to the lanes and repeat along that dimension
        if isinstance(x, np.ndarray):
            # For NumPy, use expand_dims and tile
            lanes = [
                np.expand_dims(lane, axis=0).repeat(timesteps, axis=0)
                for lane in lanes
            ]
        elif isinstance(x, torch.Tensor):
            # For PyTorch, use unsqueeze and repeat
            lanes = [
                lane.unsqueeze(0).repeat(timesteps, 1, 1) for lane in lanes
            ]
        else:
            raise TypeError(
                "Input x must be either a torch.Tensor or np.ndarray"
            )

        return lanes

    def shift_lanes(self, x, lanes):
        """
        Shifts the lanes to simulate the agent moving through space.

        args:
            x (torch.Tensor or np.ndarray): the position of the agent
                batches x timesteps x dims
            lanes (list): the lanes to encode
                list of torch.Tensors or np.ndarrays: batches x lanes x dims

        returns:
            list: the shifted lanes
                list of torch.Tensors or np.ndarrays: batches x timesteps x lanes x dims
        """

        # Check if input is a NumPy array or PyTorch tensor
        if isinstance(x, np.ndarray):
            # For NumPy, use np.cumsum
            agent_offsets = np.cumsum(x, axis=1)
        elif isinstance(x, torch.Tensor):
            # For PyTorch, use torch.cumsum
            agent_offsets = torch.cumsum(x, dim=1)
        else:
            raise TypeError("Input x must be a torch.Tensor or np.ndarray")

        shifted_lanes = []
        for i, lane in enumerate(lanes):
            agent_offsets_t = (
                agent_offsets[i][:, np.newaxis]
                if isinstance(agent_offsets, np.ndarray)
                else agent_offsets[i].unsqueeze(1)
            )

            # Subtract the agent offsets from the lanes
            lane[:, :, :2] -= agent_offsets_t

            shifted_lanes.append(lane)

        return shifted_lanes

    def __call__(self, x, lanes):
        """
        Preprocesses the lanes by filtering out irrelevant lanes and
        padding them.
        """

        # FIXME
        # this should happen after lane shifting
        # filter out lanes in the rear
        lanes = rear_filter(lanes, self.min_y_filter)

        # since angle doesn't change, do it first to minimize other points
        # considered. Only do it at the beginning, when lanes
        # don't have a timestep dimension
        if len(lanes[0].shape) == 2:
            lanes = angle_filter(lanes)

            # add a timestep dimension to the lanes:
            # list of batches x timesteps x lanes x dims
            lanes = self.add_timestep_dim(x, lanes)

        # shift the lanes to simulate the agent moving through space
        lanes = self.shift_lanes(x, lanes)

        # save the last lanes for future offsets
        if isinstance(lanes[0], np.ndarray):
            final_lanes = [lane[-1][np.newaxis] for lane in lanes]
        elif isinstance(lanes[0], torch.Tensor):
            final_lanes = [lane[-1].unsqueeze(0) for lane in lanes]
        else:
            raise TypeError(
                "Each batch of lanes must be either a torch.Tensor or \
                    np.ndarray"
            )

        lanes = distance_filter_and_pad(
            lanes, self.num_points
        )

        lanes = generate_map(lanes, 20, 0.5)

        return lanes, final_lanes
