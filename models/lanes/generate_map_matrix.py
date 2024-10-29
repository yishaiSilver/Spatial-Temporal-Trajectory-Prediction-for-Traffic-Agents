"""
A module to generate a map matrix/image from a list of lanes.
"""

import numpy as np

from utils.logger_config import logger

def generate_numpy(lanes, size=10, granularity=0.5):
    """
    Generate a map matrix from the lanes.

    lanes are of shape (batches, n points, dims (pos, norm))
    """
    # create an empty map
    map_size = int(size / granularity)
    map_matrix = np.zeros((map_size, map_size, 2))

    # pre-calculate values for scaling and centering
    half_size = size / 2 - 1
    scale_factor = 1 / granularity

    # get the coordinates of the lanes
    for lane in lanes:
        # extract and scale coordinates, then shift them
        x = int((lane[0] * scale_factor) + half_size)
        y = int((lane[1] * scale_factor) + half_size)

        # check if the coordinates are within the map bounds
        if 0 <= x < map_size and 0 <= y < map_size:
            map_matrix[x, y] = lane[2:]

    return map_matrix
