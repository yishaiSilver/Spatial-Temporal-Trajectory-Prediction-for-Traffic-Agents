"""
Module for filtering out irrelevant lanes based on the angle of the lane.
"""
import torch
import numpy as np

def angle_filter(lanes):
    """
    Takes in a list of lanes and filters out irrelevant lanes.
    Irrelevant lanes are lanes going in the opposite direction.
    Since the agent is generally moving up, we filter out down lanes.

    args:
        lanes (list): A list where each element is either a torch.Tensor or np.ndarray
            of shape (batches, dims), where dims includes the lane norms in the last two columns.

    returns:
        list: A filtered list with irrelevant lanes removed.
    """

    output = []
    for batch_lanes in lanes:
        lane_norms = batch_lanes[:, 2:]

        # Check if the data is torch.Tensor or np.ndarray
        if isinstance(batch_lanes, np.ndarray):
            # For NumPy, use np.arctan2
            lane_angles = np.arctan2(lane_norms[:, 1], lane_norms[:, 0])
            # Filter lanes where the angle is greater than 0 (moving up)
            filtered_lanes = batch_lanes[lane_angles > 0]
        elif isinstance(batch_lanes, torch.Tensor):
            # For PyTorch, use torch.atan2
            lane_angles = torch.atan2(lane_norms[:, 1], lane_norms[:, 0])
            # Filter lanes where the angle is greater than 0 (moving up)
            filtered_lanes = batch_lanes[lane_angles > 0]
        else:
            raise TypeError(
                "Each batch of lanes must be either a torch.Tensor or np.ndarray"
            )

        output.append(filtered_lanes)

    return output
