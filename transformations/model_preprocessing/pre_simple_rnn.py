"""
Module used to prepare data for the SimpleRNN model.
"""

import numpy as np
# from utils.logger_config import logger

from models.lanes.lane_preprocess import LanePreprocess

class preSimpleRNN:
    """
    SimpleMLP class for the SimpleMLP model.
    """

    @staticmethod
    def apply(datum, data_config):
        """
        Extract whatever data is needed from the datum and encapsulate it in a
        tuple.

        Args:
            datum (dict): Dictionary containing the data to be transformed.
        """

        features = data_config["features"]
        # feat_agent_positions = features["p_in"]
        # feat_agent_velocities = features["v_in"]
        lane_config = features["lane"]
        # feat_positional_embeddings = features["positional_embeddings"]

        # get the index of the target agent
        target_index = np.where(datum["track_id"] == datum["agent_id"])[0][0]

        # get the agent's position
        p_in = datum["p_in"]

        # TODO:
        # positional embeddings expansion
        # lanes?

        # inputs, labels, correction_function, and metadata
        p_in = datum["p_in"][target_index]
        p_out = datum["p_out"][target_index]

        # input positions, lane information, neighbors, teacher forcing
        inputs = [p_in, None, None, p_out]
        labels = p_out
        correction = datum["inverse"]
        metadata = datum["metadata"]

        # logger.debug(f"inputs: {inputs[0].shape}")

        if lane_config:
            lanes = np.array(datum["lane"])
            lane_norms = np.array(datum["lane_norm"])


            lanes = [np.hstack((lanes, lane_norms))]
            # stack the lanes and lane norms TODO
            # lanes = [lanes]

            # add batch dimension
            x = inputs[0][np.newaxis, :, :]

            # # preprocess the lanes
            #. FIXME add to config spec
            lanes, last_lane = LanePreprocess(lane_config)(x, lanes)

            # logger.debug(f"lanes: {lanes.shape}")

            inputs[1] = (lanes[0], last_lane[0])

        metadata["scene_id"] = datum["scene_idx"]

        return inputs, labels, correction, metadata

    @staticmethod
    def inverse(batch_predictions, batch_metadata):
        """
        Apply the prediction correction to the data.
        """
        _ = batch_metadata # unused warning

        num_batches = len(batch_predictions)

        # reshape to num_batches x num_timesteps x num_features
        # TODO this is a magic number. Use metadata?
        batch_predictions = batch_predictions.reshape(num_batches, 30, 2)

        return batch_predictions
