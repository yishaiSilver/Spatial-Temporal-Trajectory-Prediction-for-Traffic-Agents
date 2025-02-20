import numpy as np


class preSeq2Seq:
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

        # features = data_config["features"]
        # feat_agent_positions = features["p_in"]
        # feat_agent_velocities = features["v_in"]
        # feat_lanes = features["lane"]
        # feat_positional_embeddings = features["positional_embeddings"]

        # get the index of the target agent
        target_index = np.where(datum["track_id"] == datum["agent_id"])[0][0]

        # get the agent's position
        # p_in = datum["p_in"]
        # target_p_in = p_in[target_index]  # (t, 2)

        # TODO:
        # positional embeddings expansion
        # lanes?

        # inputs, labels, correction_function, and metadata
        inputs = datum["p_in"][target_index]
        labels = datum["p_out"][target_index]
        correction = datum["inverse"]
        metadata = datum["metadata"]

        return inputs, labels, correction, metadata

    @staticmethod
    def inverse(batch_predictions, batch_metadata):
        """
        Apply the prediction correction to the data.
        """
        _ = batch_metadata # unused

        num_batches = len(batch_predictions)

        # reshape to num_batches x num_timesteps x num_features
        # TODO this is a magic number. Use metadata?
        batch_predictions = batch_predictions.reshape(num_batches, 30, 2)

        return batch_predictions
