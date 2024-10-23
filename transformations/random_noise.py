"""

This module contains the `AgentCenter` class which applies agent-centered
transformation to the given batch data.

"""

import numpy as np


def apply(datum, position_noise=0.1, norm_noise=0.05):
    """
    Apply random noise to the given datum.
    """

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
    positions += np.random.normal(0, position_noise, positions.shape)
    # velocities += np.random.normal(0, norm_noise, velocities.shape)
    lane_positions += np.random.normal(0, position_noise, lane_positions.shape)
    lane_norms += np.random.normal(0, norm_noise, lane_norms.shape)

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
    Do nothing for the inverse. The original positions are lost.
    """

    # get rid of metadata not used warning
    __ = metadata

    return predictions
