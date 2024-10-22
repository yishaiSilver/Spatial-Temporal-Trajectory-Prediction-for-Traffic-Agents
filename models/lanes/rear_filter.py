"""
This is a simple module to filter out lanes that are behind the agent.
"""

import numpy as np
import torch

"""
Module for filtering out irrelevant lanes based on the angle of the lane.
"""
import torch
import numpy as np

from utils.logger_config import logger

def rear_filter(lanes):
    """
    Takes in a list of lanes and filters out irrelevant lanes based
    on whether the lanes are behind the agent.

    args:
        lanes (list): A list where each element is either a torch.Tensor or np.ndarray
            of shape (batches, dims), where dims includes the lane norms in the last two columns.

    returns:
        list: A filtered list with irrelevant lanes removed.
    """

    output = []
    for batch_lanes in lanes:
        if isinstance(batch_lanes, np.ndarray):
            lane_positions_y = batch_lanes[:, 1]
            # Filter lanes where the angle is greater than 0 (moving up)
            filtered_lanes = batch_lanes[lane_positions_y > 0]

        elif isinstance(batch_lanes, torch.Tensor):
            lane_positions_y = batch_lanes[:, :, 1]
            # Filter lanes where the angle is greater than 0 (moving up)
            filtered_lanes = batch_lanes[lane_positions_y > 0]

            filtered_lanes = filtered_lanes.unsqueeze(0)

        output.append(filtered_lanes)

    return output
