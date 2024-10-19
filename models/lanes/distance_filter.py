"""
A module used to filter out irrelevant lanes by distance.
"""

import torch

from utils.logger_config import logger


def distance_filter_and_pad(lanes, distance, padded=30):
    """
    Takes in list of lanes and filters out irrelevant lanes.
    Irrelevant lanes are lanes that are too far to be relevant to the agent.
    
    IMPORTANT: this function doesn't actually filter out lanes, it just sorts
    them by distance and pads them with zeros. Actually filtering out the lanes
    more than doubled the time it took to run the model, so we're just going to
    
    """

    output = []
    for batch_lanes in lanes:
        lane_norms = batch_lanes[:, :, :2]

        # get the distance from the agent to the lane
        distances = torch.norm(lane_norms, dim=2)

        # for each timestep, get the lanes that are close enough
        close_and_padded_lanes = []

        # Get the number of timesteps and lanes
        num_timesteps, num_lanes, _ = lane_norms.shape

        # Pad lanes with zeros if necessary
        if num_lanes < padded:
            zero_pads = torch.zeros(
            (num_timesteps, padded - num_lanes, batch_lanes.shape[2]),
            device=batch_lanes.device,
            )
            batch_lanes = torch.cat((batch_lanes, zero_pads), dim=1)
        else:
            # Sort distances and get indices of the closest lanes
            sorted_indices = torch.argsort(distances, dim=1)[:, :padded]

            # Get the closest lanes
            batch_lanes = torch.gather(batch_lanes, 1, sorted_indices.unsqueeze(2).expand(-1, -1, batch_lanes.shape[2]))



        # # apply distance threshold
        # mask = distances < distance

        # logger.debug(f"mask.shape: {mask.shape}")
        # logger.debug(f"batch_lanes.shape: {batch_lanes.shape}")

        # masked_lanes = batch_lanes * mask.unsqueeze(2)


        # convert to tensor
        # batch_lanes = torch.stack(close_and_padded_lanes)

        output.append(batch_lanes)

    # convert from list of tensors to tensor
    output = torch.stack(output)

    return output
