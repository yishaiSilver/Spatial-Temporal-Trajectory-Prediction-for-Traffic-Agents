"""
Zero pads the lanes to a fixed number of lanes.
"""

import torch


def zero_pad(lanes, num_lanes):
    """
    Zero pads the lanes to a fixed number of lanes.
    """

    output = []
    for batch_lanes in lanes:
        if len(batch_lanes) >= num_lanes:
            # if we have more lanes than we need, take the closest to origin
            distances = torch.norm(batch_lanes[:, :2], dim=1)
            sorted_indices = torch.argsort(distances)[:num_lanes]
            batch_lanes = batch_lanes[sorted_indices]
            output.append(batch_lanes)
        else:
            zero_pads = torch.zeros(
                num_lanes - len(batch_lanes),
                batch_lanes.shape[1],
                device=batch_lanes.device,
            )
            output.append(torch.cat((batch_lanes, zero_pads), dim=0))

    return output
