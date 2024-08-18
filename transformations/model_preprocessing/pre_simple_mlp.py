import numpy as np

from transformations.base import BaseTransformation


class preSimpleMLP(BaseTransformation):
    """
    SimpleMLP class for the SimpleMLP model.
    """

    def __init__(self, data_config: dict):
        """
        Constructor for the SimpleMLP class.

        Acceptable inputs are p_in, v_in, and lane.

        args:
            model_config (dict): dictionary containing the model configuration.
        """
        features = data_config["features"]
        self.p_in = features["p_in"]
        self.v_in = features["v_in"]
        self.lane = features["lane"]

        return

    def apply(self, datum):
        """
        Extract whatever data is needed from the datum and encapsulate it in a
        tuple.

        Args:
            datum (dict): Dictionary containing the data to be transformed.
        """

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
        if self.p_in > 0:
            # sort the agents by distance to the target agent
            sorted_indices = np.argsort(
                np.linalg.norm(p_in - target_p_in, axis=1)
            )

            for i in range(1, self.p_in + 1):  # skip 0 because it's the target
                # get the position of the ith closest agent
                agent_p_in = p_in[sorted_indices[i]]

                # add the position to the vector, flattened
                vector.extend(agent_p_in.flatten())

                # add velocities if needed
                if self.v_in > 0:
                    # get the velocity of the ith closest agent
                    agent_v_in = datum["v_in"][sorted_indices[i]]
                    vector.extend(agent_v_in.flatten())

        if self.lane > 0:
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

            for i in range(self.lane):
                lane_position = lane[sorted_indices[i]]
                lane_norm = lane_norms[sorted_indices[i]]

                vector.extend(lane_position)
                vector.extend(lane_norm)

        # TODO: perhaps move this to be part of __init__?
        # save the prior prediction correction
        self.prior_prediction_correction = datum["prediction_correction"]

        # inputs, labels, correction_function, and metadata
        inputs = vector
        labels = datum["p_out"][target_index].flatten()
        correction_function = self.prediction_correction
        metadata = None
        return inputs, labels, correction_function, metadata

    def prediction_correction(self, batch_predictions, batch_metadata):
        """
        Apply the prediction correction to the data.
        """

        return self.prior_prediction_correction(
            batch_predictions, batch_metadata
        )
