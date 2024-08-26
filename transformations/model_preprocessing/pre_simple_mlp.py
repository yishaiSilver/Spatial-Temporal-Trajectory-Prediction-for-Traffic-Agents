import numpy as np


class preSimpleMLP():
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
        feat_agent_positions = features["p_in"]
        feat_agent_velocities = features["v_in"]
        feat_lanes = features["lane"]


        # the MLP can only take in a flattened vector
        vector = []

        agent_ids = datum["track_id"]
        target_id = datum["agent_id"]

        target_index = np.where(agent_ids == target_id)[0][0]

        # get the agent's position
        p_in = datum["p_in"]
        target_p_in = p_in[target_index]

        # add the target agent's position, flattened
        vector.extend(target_p_in.flatten())

        # add velocities of the n nearest agents
        if feat_agent_positions > 0:
            # sort the agents by distance to the target agent
            sorted_indices = np.argsort(
                np.linalg.norm(p_in - target_p_in, axis=1)
            )

            for i in range(1, p_in + 1):  # skip 0 because it's the target
                # get the position of the ith closest agent
                agent_p_in = p_in[sorted_indices[i]]

                # add the position to the vector, flattened
                vector.extend(agent_p_in.flatten())

                # add velocities if needed
                if feat_agent_velocities > 0:
                    # get the velocity of the ith closest agent
                    agent_v_in = datum["v_in"][sorted_indices[i]]
                    vector.extend(agent_v_in.flatten())

        if feat_lanes > 0:
            # get the lane of the target agent
            lane = datum["lane"]
            lane_norms = datum["lane_norm"]

            # sort the lane positions by distance to the target agent's
            # final position
            final_position = target_p_in[-1]

            # sort the lanes by distance to the target agent
            sorted_indices = np.argsort(
                np.linalg.norm(lane - final_position, axis=1)
            )

            for i in range(lane):
                lane_position = lane[sorted_indices[i]]
                lane_norm = lane_norms[sorted_indices[i]]

                vector.extend(lane_position)
                vector.extend(lane_norm)

        # inputs, labels, correction_function, and metadata
        inputs = vector
        labels = datum["p_out"][target_index]
        correction = datum["inverse"]
        metadata = datum["metadata"]
        return inputs, labels, correction, metadata

    @staticmethod
    def inverse(batch_predictions, batch_metadata):
        """
        Apply the prediction correction to the data.
        """
        num_batches = len(batch_predictions)

        # reshape to num_batches x num_timesteps x num_features
        # TODO this is a magic number. Use metadata?
        batch_predictions = batch_predictions.reshape(
            num_batches, 30, 2
        )

        return batch_predictions