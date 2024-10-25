"""
A module to generate a map matrix/image from a list of lanes.
"""

import torch
import numpy as np

from utils.logger_config import logger

# def get_coord)

def generate_numpy(x, lanes, size=10, granularity=0.5):
    """
    Generate a map matrix from the lanes.

    lanes are of shape (batches, n points, dims (pos, norm))
    """

    # create an empty map
    map_size = size / granularity
    map_matrix = np.zeros((map_size, map_size, 2))

    # get the coordinates of the lanes
    for lane in lanes:
        logger.debug(f"lane: {lane.shape}")
        # lane = lane.squeeze()
        x = lane[:, 0]
        y = lane[:, 1]

        # get the coordinates of the lane
        x = x / granularity
        y = y / granularity

        # get the coordinates of the lane
        x = x + size / 2
        y = y + size / 2

        # get the coordinates of the lane
        x = x.astype(int)
        y = y.astype(int)

        # get the coordinates of the lane
        map_matrix[x, y] =  lane[:, 2:]

    return map_matrix