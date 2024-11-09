"""
A module to generate a map matrix/image from a list of lanes.
"""

import numpy as np
import torch

from utils.logger_config import logger

def generate_numpy(lanes, size=10, granularity=0.5):
    """
    Generate a map matrix from the lanes.

    lanes are of shape (batches, n points, dims (pos, norm))
    """

    # create an empty map
    map_size = int(size / granularity)
    half_size = map_size / 2 - 1
    scale_factor = 1 / granularity

    batch_maps = []

    for batch in lanes:
        time_maps = []

        for lanes_t in batch:
            map_matrix = np.zeros((2, map_size, map_size))

            # get the coordinates of the lanes
            for lane in lanes_t:
                # extract and scale coordinates, then shift them
                x = int((lane[0] * scale_factor) + half_size)
                y = int((lane[1] * scale_factor) + half_size)

                # check if the coordinates are within the map bounds
                if 0 <= x < map_size and 0 <= y < map_size:
                    map_matrix[:, x, y] = lane[2:]

            time_maps.append(map_matrix)
        
        batch_maps.append(time_maps)

    batch_maps = np.array(batch_maps)

    return batch_maps

def generate_torch(lanes, size=10, granularity=0.5):
    """
    Generate a map matrix from the lanes.

    lanes are of shape (batches, n points, dims (pos, norm))
    """

    # create an empty map
    map_size = int(size / granularity)
    half_size = map_size / 2 - 1
    scale_factor = 1 / granularity

    batch_maps = []

    for batch in lanes:
        map_matrix = torch.zeros((2, map_size, map_size), device="cuda")

        lanes_t = batch[0]

        # get the coordinates of the lanes
        coords = ((lanes_t[:, :2] * scale_factor) + half_size).long()
        valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < map_size) & (coords[:, 1] >= 0) & (coords[:, 1] < map_size)
        valid_coords = coords[valid_mask]
        valid_lanes = lanes_t[valid_mask]

        # fill the map matrix
        map_matrix[:, valid_coords[:, 0], valid_coords[:, 1]] = valid_lanes[:, 2:].T

        # add the time dimention
        map_matrix = map_matrix.unsqueeze(0)

        batch_maps.append(map_matrix)

    batch_maps = torch.stack(batch_maps)

    return batch_maps
    

def generate_map(lanes, size, granularity):
    """
    Takes in points and generates map
    """

    if isinstance(lanes, np.ndarray):
        lanes = generate_numpy(lanes, size, granularity)
    elif isinstance(lanes, torch.Tensor):
        lanes = generate_torch(lanes, size, granularity)

    return lanes
