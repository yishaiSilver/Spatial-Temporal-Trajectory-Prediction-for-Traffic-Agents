"""
A module used to filter out irrelevant lanes by distance.
"""

import torch

from utils.logger_config import logger


def distance_filter_and_pad(lanes, distance, padded=30):
    """
    Takes in list of lanes and filters out irrelevant lanes.
    Irrelevant lanes are lanes that are too far to be relevant to the agent.
    """

    output = []
    for batch_lanes in lanes:
        lane_norms = batch_lanes[:, :, :2]

        # get the distance from the agent to the lane
        distances = torch.norm(lane_norms, dim=2)

        # get the indices of the lanes that are close enough
        # timestep x lanes
        mask = distances < distance

        # for each timestep, get the lanes that are close enough
        close_and_padded_lanes = []
        for t in range(mask.shape[0]):

            # pull out the lanes that are close enough
            close_enough = batch_lanes[t][mask[t]]

            # pad the lanes with zeros
            if close_enough.shape[0] < padded:
                zero_pads = torch.zeros(
                    padded - close_enough.shape[0],
                    close_enough.shape[1],
                    device=close_enough.device,
                )
                close_enough = torch.cat((close_enough, zero_pads), dim=0)
            else:
                # if we have more lanes than we need, take the closest to origin
                distances_t = distances[t][mask[t]]
                sorted_indices = torch.argsort(distances_t)[:padded]
                close_enough = close_enough[sorted_indices]

            close_and_padded_lanes.append(close_enough)

        # convert to tensor
        batch_lanes = torch.stack(close_and_padded_lanes)

        output.append(batch_lanes)

    # convert from list of tensors to tensor
    output = torch.stack(output)

    return output
