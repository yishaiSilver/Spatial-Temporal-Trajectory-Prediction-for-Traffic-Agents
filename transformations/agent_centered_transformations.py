"""

This module contains the `AgentCenter` class which applies agent-centered
transformation to the given batch data.

"""

import numpy as np
import torch


def homogenize_matrix(matrix):
    """
    Homogenize a 2D matrix by adding a column of ones.

    Args:
        matrix (np.ndarray): 2D matrix.

    Returns:
        np.ndarray: Homogenized matrix with an additional column of ones.
    """

    # get the original shape
    original_shape = matrix.shape

    # get the non-numerical dimensions
    non_numerical_dims = original_shape[:-1]

    # add the '1' layer/row
    shape = non_numerical_dims + (1,)
    ones = np.ones(shape)

    homogenized_matrix = np.concatenate(
        [matrix, ones],
        axis=-1,
    )
    return homogenized_matrix


def get_translation_matrix(positions):
    """
    Gets the translation matrix for the given positions.
    """
    num_timesteps = positions.shape[0]

    translation_transforms = np.eye(3)[np.newaxis].repeat(num_timesteps, axis=0)

    # set the translation component of the transformation matrices
    translation_transforms[:, :2, 2] -= positions

    return translation_transforms


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
        -np.arctan2(
            last_position[1] - first_position[1],
            last_position[0] - first_position[0],
        )
        + np.pi / 2
    )

    rotation_transforms[0, 0] = np.cos(theta)
    rotation_transforms[0, 1] = -np.sin(theta)
    rotation_transforms[1, 0] = np.sin(theta)
    rotation_transforms[1, 1] = np.cos(theta)

    return rotation_transforms


def apply(datum):
    """
    Apply agent-centered transformation to the given datum.

    Args:
        datum (dict): Dictionary representing a single data point.

    Returns:
        dict: Transformed datum with updated positions.
    """
    # get all of the ids for the agents being tracked
    # renaming due to bad naming in the dataset
    agent_ids = datum["track_id"]

    # extract the agent_id from the datum
    target_id = datum["agent_id"]

    # get the index of the target agent
    agent_index = np.where(agent_ids == target_id)[0][0]

    # get the input and output data
    positions_in = np.array(datum["p_in"])
    velocities_in = np.array(datum["v_in"])
    positions_out = np.array(datum["p_out"])
    velocities_out = np.array(datum["v_out"])

    # FIXME:
    # save the input length before we extend it
    input_length = positions_in.shape[1]

    # extend by the output data
    positions = np.concatenate([positions_in, positions_out], axis=1)
    velocities = np.concatenate([velocities_in, velocities_out], axis=1)

    # ccenter the positions around the target agent
    target_positions = positions[agent_index]
    positions = positions - target_positions

    # create the rotation transform (key difference: only one needed)
    rotation_transforms = get_rotation_matrix(positions_in[agent_index])
    positions = positions @ rotation_transforms

    offsets = np.diff(target_positions, axis=0)
    first_offset = np.array([0, 0])
    offsets = np.vstack([first_offset, offsets])
    offsets = offsets @ rotation_transforms
    positions[agent_index] = offsets

    # update the positions in the datum
    datum["p_in"] = positions[:, :input_length]
    datum["v_in"] = velocities[:, :input_length]
    datum["v_out"] = velocities[:, input_length:]

    # update the prediction correction
    datum["prediction_correction"] = inverse

    # get the inverses of the translation and rotation matrices
    # translation_transforms_inv = np.linalg.inv(translation_transforms)
    rotation_transforms_inv = np.linalg.inv(rotation_transforms)

    metadata = {
        "target_offset": target_positions,
        "rotation_transforms": rotation_transforms_inv,
    }

    datum["batch_correction_metadata"] = metadata

    return datum


def inverse(predictions, metadata):
    """TODO: correct_predictions"""

    # IMPORTANT: inputs are batched
    batch_size = predictions.shape[0]

    # cumsum along the time dimension
    predictions = torch.cumsum(predictions, axis=1)

    # get the translation and rotation matrices
    target_positions = [
        metadata[i]["target_offset"][18] for i in range(batch_size)
    ]
    rotation_transforms = [
        metadata[i]["rotation_transforms"] for i in range(batch_size)
    ]

    # convert to numpy
    target_positions = np.array(target_positions)
    rotation_transforms = np.array(rotation_transforms)

    # convert to tensors
    target_positions = torch.tensor(target_positions, dtype=torch.float32)
    rotation_transforms = torch.tensor(rotation_transforms, dtype=torch.float32)

    # apply to all timestamps (batch, timestamp, 2)
    target_positions = target_positions.unsqueeze(1)

    # put on device
    device = predictions.device
    target_positions = target_positions.to(device)
    rotation_transforms = rotation_transforms.to(device)

    # apply the inverse transformations
    # (30, 2) @ (2, 2) -> (30, 2)
    predictions = predictions @ rotation_transforms

    # (30, 2) + (2) -> (30, 2)
    predictions = predictions + target_positions

    return predictions
