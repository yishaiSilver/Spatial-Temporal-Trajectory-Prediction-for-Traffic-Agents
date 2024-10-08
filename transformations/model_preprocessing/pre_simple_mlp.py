import numpy as np
from utils.logger_config import logger

class preSimpleMLP:
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
        feat_positional_embeddings = features["positional_embeddings"]

        # the MLP can only take in a flattened vector
        vector = []

        agent_ids = datum["track_id"]
        target_id = datum["agent_id"]

        target_index = np.where(agent_ids == target_id)[0][0]

        # get the agent's position
        p_in = datum["p_in"]
        target_p_in = p_in[target_index]

        # get the number of agents (non-zero)
        agent_mask = datum["car_mask"]
        num_agents = np.sum(agent_mask)


        # add the target agent's position, flattened
        vector.extend(target_p_in.flatten())

        # add velocities of the n nearest agents
        if feat_agent_positions > 0:
            # sort the agents by distance to the target agent
            # since target agent is so close to 0, just use 0
            sorted_indices = np.argsort(
                np.linalg.norm(p_in[:, :, :], axis=(1, 2))
            )

            for i in range(
                1, feat_agent_positions + 1
            ):  
                # skip 0 because it's the target
                # get the position of the ith closest agent
                agent_p_in = p_in[sorted_indices[i]]

                # change to zero-padding if non-agent
                if i >= num_agents:
                    agent_p_in = np.zeros_like(agent_p_in)

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

            for i in range(feat_lanes):
                if i >= len(sorted_indices):
                    lane_position = np.zeros_like(lane[0])
                    lane_norm = np.zeros_like(lane_norms[0])
                else:
                    lane_position = lane[sorted_indices[i]]
                    lane_norm = lane_norms[sorted_indices[i]]

                vector.extend(lane_position)
                vector.extend(lane_norm)

        # now that we've added all the features, we can add the positional 
        # embeddings if needed
        if feat_positional_embeddings > 0:
            # get the positional embeddings
            vector = np.array(vector)

            new_vector = []

            for i in range(feat_positional_embeddings):
                s = np.sin(2**(i) * np.pi * vector)
                c = np.cos(2**(i) * np.pi * vector)

                # append the sin and cos to the new vector
                new_vector.extend(s)
                new_vector.extend(c)

            new_vector = np.array(new_vector)

            vector = new_vector.flatten()

        # inputs, labels, correction_function, and metadata
        inputs = vector
        labels = datum["p_out"][target_index]
        correction = datum["inverse"]
        metadata = datum["metadata"]

        # logger.debug(" Inputs shape: %s", inputs.shape)

        return inputs, labels, correction, metadata

    @staticmethod
    def inverse(batch_predictions, batch_metadata):
        """
        Apply the prediction correction to the data.
        """
        num_batches = len(batch_predictions)

        # reshape to num_batches x num_timesteps x num_features
        # TODO this is a magic number. Use metadata?
        batch_predictions = batch_predictions.reshape(num_batches, 30, 2)

        return batch_predictions
