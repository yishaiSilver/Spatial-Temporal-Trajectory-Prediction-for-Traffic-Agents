"""
A module used to filter out irrelevant lanes via angle.
"""
import torch

def angle_filter(lanes):
    """
    Takes in list of lanes and filters out irrelevant lanes.
    Irrelevant lanes are lanes going in the opposite direction.
    Since the agent is generally moving up, we filter out down lanes.
    """

    output = []
    for batch_lanes in lanes:
        lane_norms = batch_lanes[:, 2:]
        
        # get the angle of the lane
        lane_angles = torch.atan2(lane_norms[:, 1], lane_norms[:, 0])

        # since the agent is moving up, we want to filter out lanes that are
        # going down (more than 90 degrees from 90 degrees)
        mask = lane_angles > 0

        # apply the mask
        batch_lanes = batch_lanes[mask]

        output.append(batch_lanes)

    return output