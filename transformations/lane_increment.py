"""
A simple module to manage lane information over timesteps
"""

import numpy as np
import torch
from utils.logger_config import logger


def increment_lanes_numpy(lanes, x):
    """
    increment the lanes using numpy

    params:
        lanes: lane positions
        x: timesteps x agent positions
    """
    timesteps, _ = x.shape

    # cumsum x across timesteps
    x_translation = np.cumsum(x, 0)


    # TODO make this its own transformation
    # sort lanes by distance from the last position
    last_position = x_translation[-1]
    distances = np.linalg.norm(lanes - last_position, axis=1)
    sorted_indices = np.argsort(distances)
    lanes = lanes[sorted_indices]

    # insert a new axis on lanes
    lanes = lanes[np.newaxis, :, :]

    # repeat lanes across the number of timesteps:
    lanes = np.repeat(lanes, timesteps, axis=0)

    # move lanes to be relative to x
    lanes = lanes - x_translation[:, np.newaxis, :]

    return lanes