"""

This module contains the `AgentCenter` class which applies agent-centered
transformation to the given batch data.

"""

import numpy as np
import torch

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

    # get the input and output data
    positions_in = np.array(datum["p_in"])
    positions_out = np.array(datum["p_out"])

    velocities_in = np.array(datum["v_in"])
    velocities_out = np.array(datum["v_out"])

    lane_positions = np.array(datum["lane"])
    lane_norms = np.array(datum["lane_norm"])

    # save the input length before we extend it
    input_length = positions_in.shape[1]

    # extend by the output data
    positions = np.concatenate([positions_in, positions_out], axis=1)
    velocities = np.concatenate([velocities_in, velocities_out], axis=1)

    # add random noise to the positions
    positions += np.random.normal(0, 0.1, positions.shape)
    # velocities += np.random.normal(0, 0.1, velocities.shape)
    lane_positions += np.random.normal(0, 0.1, lane_positions.shape)
    lane_norms += np.random.normal(0, 0.05, lane_norms.shape)

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

    # can't truly recover the original positions
    return predictions