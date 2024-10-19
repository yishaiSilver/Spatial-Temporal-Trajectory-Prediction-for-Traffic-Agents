"""
A module used to filter out irrelevant lanes by distance.
"""

import torch
import numpy as np

from utils.logger_config import logger


def distance_filter_and_pad(lanes, padded=30):
    """
    Takes in list of lanes and filters out irrelevant lanes.
    Irrelevant lanes are lanes that are too far to be relevant to the agent.

    IMPORTANT: this function doesn't actually filter out lanes, it just sorts
    them by distance and pads them with zeros. Actually filtering out the lanes
    more than doubled the time it took to run the model, so we're just going to
    sort and pad.
    """

    output = []
    for batch_lanes in lanes:
        lane_norms = batch_lanes[:, :, :2]

        # Check if the input is a NumPy array or a PyTorch tensor
        if isinstance(batch_lanes, np.ndarray):
            # NumPy version of norm (equivalent to torch.norm)
            distances = np.linalg.norm(lane_norms, axis=2)
            num_timesteps, num_lanes, _ = lane_norms.shape

            # Pad lanes with zeros if necessary
            if num_lanes < padded:
                zero_pads = np.zeros(
                    (num_timesteps, padded - num_lanes, batch_lanes.shape[2])
                )
                batch_lanes = np.concatenate((batch_lanes, zero_pads), axis=1)
            else:
                # Sort distances and get indices of the closest lanes
                sorted_indices = np.argsort(distances, axis=1)[:, :padded]

                # Gather the closest lanes
                batch_lanes = np.take_along_axis(
                    batch_lanes, sorted_indices[:, :, np.newaxis], axis=1
                )

        elif isinstance(batch_lanes, torch.Tensor):
            # PyTorch version of norm
            distances = torch.norm(lane_norms, dim=2)
            num_timesteps, num_lanes, _ = lane_norms.shape

            # Pad lanes with zeros if necessary
            if num_lanes < padded:
                zero_pads = torch.zeros(
                    (num_timesteps, padded - num_lanes, batch_lanes.shape[2]),
                    device=batch_lanes.device,
                )
                batch_lanes = torch.cat((batch_lanes, zero_pads), dim=1)
            else:
                # FIXME this is the slowest part of the model
                # Sort distances and get indices of the closest lanes
                # sorted_indices_arg = torch.argsort(distances, dim=1)[:, :padded]

                # the following is faster than argsort
                sorted_distances, sorted_indices = torch.sort(distances, dim=1)
                sorted_indices = sorted_indices[:, :padded]

                # Gather the closest lanes
                batch_lanes = torch.gather(
                    batch_lanes,
                    1,
                    sorted_indices.unsqueeze(2).expand(
                        -1, -1, batch_lanes.shape[2]
                    ),
                )

        else:
            raise TypeError(
                "Each batch of lanes must be either a torch.Tensor or np.ndarray"
            )

        output.append(batch_lanes)

    # Convert the list of arrays/tensors to the final stacked output
    if isinstance(output[0], np.ndarray):
        output = np.stack(output)
    elif isinstance(output[0], torch.Tensor):
        output = torch.stack(output)
    else:
        raise TypeError(
            "Output data must be either NumPy arrays or PyTorch tensors."
        )

    return output
