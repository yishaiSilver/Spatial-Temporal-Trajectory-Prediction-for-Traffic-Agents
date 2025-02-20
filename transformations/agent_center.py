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
    Apply agent-centered transformation to the given datum.

    Args:
        datum (dict): Dictionary representing a single data point.

    Returns:
        dict: Transformed datum with updated positions.

    TODO: 
        - [ ] Add support for which datum modifications are used
              I.e. are we using the lane data? do we need to update it?
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
    if "p_out" not in datum:
        datum["p_out"] = np.zeros_like(positions_in)
    positions_out = np.array(datum["p_out"])

    # save the final positions for inverse purposes
    final_known = positions_in[agent_index, -1]

    velocities_in = np.array(datum["v_in"])
    if "v_out" not in datum:
        datum["v_out"] = np.zeros_like(velocities_in)
    velocities_out = np.array(datum["v_out"])

    lane_positions = np.array(datum["lane"])
    lane_norms = np.array(datum["lane_norm"])

    # save the input length before we extend it
    input_length = positions_in.shape[1]

    # extend by the output data
    positions = np.concatenate([positions_in, positions_out], axis=1)
    velocities = np.concatenate([velocities_in, velocities_out], axis=1)

    # center the positions around the target agent
    target_positions = positions[agent_index]
    positions = positions - target_positions
    lane_positions = lane_positions - target_positions[0] # only using 0th ts

    # create the rotation transform (key difference: only one needed)
    rotation_transforms = get_rotation_matrix(positions_in[agent_index])
    # rotation_transforms = np.eye(2)
    positions = positions @ rotation_transforms
    lane_positions = lane_positions @ rotation_transforms
    lane_norms = lane_norms @ rotation_transforms

    # print(target_positions.shape)

    # set the agent indices to be a displacement from ts to ts
    offsets = np.diff(target_positions, axis=0)
    first_offset = np.array([0, 0])
    offsets = np.vstack([first_offset, offsets])
    offsets = offsets @ rotation_transforms
    positions[agent_index] = offsets

    # update the positions in the datum
    datum["p_in"] = positions[:, :input_length]
    datum["v_in"] = velocities[:, :input_length]

    # save the outputs (transformed, for visualization purposes)
    datum["p_out_transformed"] = positions[:, input_length:]
    datum["v_out"] = velocities[:, input_length:]

    datum["lane"] = lane_positions
    datum["lane_norm"] = lane_norms

    # update the prediction correction
    datum["prediction_correction"] = inverse

    # get the inverses of the translation and rotation matrices
    # translation_transforms_inv = np.linalg.inv(translation_transforms)
    rotation_transforms_inv = np.linalg.inv(rotation_transforms)

    metadata = {
        "final_known": final_known,
        "rotation_transforms": rotation_transforms_inv,
    }

    datum["metadata"] = metadata

    return datum


def inverse(predictions, metadata):
    """
    Inverse the agent-centered transformation applied to the predictions:
    takes in predicted offsets from network, returns them into the original 
    world coordinate system.

    # TODO: Perhaps cumsum belongs to the model, not the transformation.

    # IMPORTANT: inputs are batched

    Args:
        predictions (torch.Tensor): The predictions to be transformed.
        metadata (dict): The metadata containing the target offset and rotation transforms.
    """

    # IMPORTANT: inputs are batched
    batch_size = predictions.shape[0]

    # print(predictions.shape)

    # cumsum along the time dimension
    predictions = torch.cumsum(predictions, axis=1)

    # # get the translation and rotation matrices
    final_known = torch.tensor(np.array([
        metadata[i]["final_known"] for i in range(batch_size)
    ]), dtype=torch.float32)
    rotation_transforms = torch.tensor(np.array([
        metadata[i]["rotation_transforms"] for i in range(batch_size)
    ]), dtype=torch.float32)

    # # put on device
    device = predictions.device
    final_known = final_known.to(device)
    rotation_transforms = rotation_transforms.to(device)

    # apply the inverse transformations
    # (b, 30, 2) @ (b, 2, 2) -> (b, 30, 2)
    predictions = predictions @ rotation_transforms

    # print(predictions.shape)
    # print(final_known.shape)

    # print(predictions)

    # # (b, 30, 2) + (b, 2) -> (b, 30, 2)
    predictions = predictions + final_known.unsqueeze(1)

    return predictions
