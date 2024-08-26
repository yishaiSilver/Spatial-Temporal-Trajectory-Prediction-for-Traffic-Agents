"""

This module contains the `AgentCenter` class which applies agent-centered
transformation to the given batch data.

"""

import numpy as np
import torch


def get_rotation_matrix(positions):
    """
    Gets the rotation matrix for the given positions.
    """
    rotation_transforms = np.eye(2)

    # get the angle from the target agent's first input position to the
    # final input position
    first_position = positions[0]
    last_position = positions[-1]

    # get the angle
    theta = (
        np.arctan2(
            last_position[1] - first_position[1],
            last_position[0] - first_position[0],
        )
        - np.pi / 2
    )

    rotation_transforms[0, 0] = np.cos(theta)
    rotation_transforms[0, 1] = -np.sin(theta)
    rotation_transforms[1, 0] = np.sin(theta)
    rotation_transforms[1, 1] = np.cos(theta)

    return rotation_transforms


def apply(datum):
    """
    We really only want to use lanes that are relevant to the agent.
    This means that lanes going in the opposite direction are not relevant and
    should be removed.

    Args:
        datum (dict): Dictionary representing a single data point.

    Returns:
        dict: datum with irrelevant lanes removed.
    """

    # get the angle of the p_in agent
    agent_id = datum["agent_id"]
    agent_index = np.where(datum["track_id"] == agent_id)[0][0]

    p_in = datum["p_in"]
    positions = p_in[agent_index]
    first_position = positions[0]
    last_position = positions[-1]

    # get the angle
    theta = np.arctan2(
        last_position[1] - first_position[1],
        last_position[0] - first_position[0],
    )

    new_lanes = []

    # get the lane angles based on their norm
    lane_norms = datum["lane_norm"]

    angles = np.arctan2(lane_norms[:, 1], lane_norms[:, 0])

    # get the angle difference
    angle_diffs = np.abs(angles - theta)

    # get the lane positions
    lane_positions = datum["lane"]


    return datum


def inverse(predictions, metadata):
    """
    Lane removal is not relevant to the predicted outcome of the model, so we
    fo nothing here.

    TODO: perhaps base.py can just skip over the inverse of data removal
    transformations like this one.
    """
    return predictions
