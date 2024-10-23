"""
This is a simple module to filter out lanes that are behind the agent.
"""

import torch
import numpy as np


def rear_filter(lanes, min_y=-5):
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
            filtered_lanes = batch_lanes[lane_positions_y > min_y]

        elif isinstance(batch_lanes, torch.Tensor):
            lane_positions_y = batch_lanes[:, :, 1]
            # Filter lanes where the angle is greater than 0 (moving up)
            filtered_lanes = batch_lanes[lane_positions_y > min_y]

            filtered_lanes = filtered_lanes.unsqueeze(0)

        output.append(filtered_lanes)

    return output
