"""

This module contains the `AgentCenter` class which applies agent-centered
transformation to the given batch data.

"""

import numpy as np


class AgentCenter:
    """
    Applies agent-centered transformation to the given batch_data.

    Methods:
        apply(batch_data):
            Apply agent-centered transformation to the given batch_data.

        invert(batch_data):
            Inverts the position inputs and outputs in the batch data by
            removing the offsets.
    """

    def __init__(self):
        super().__init__()

        self.input_offsets = None
        self.output_offsets = None

        return

    def homogenize_matrix(self, matrix):
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

    def apply(self, datum):
        """
        Apply agent-centered transformation to the given datum.

        Args:
            datum (dict): Dictionary representing a single data point.

        Returns:
            dict: Transformed datum with updated positions.
        """
        # get all of the ids for the agents being tracked
        agent_ids = datum["track_id"]

        # extract the agent_id from the datum
        target_id = datum["agent_id"]

        # get the index of the target agent
        agent_index = np.where(agent_ids == target_id)[0][0]

        # get the lanes and norms for the target agent
        lanes = np.array(datum["lane"])
        lane_norms = np.array(datum["lane_norm"])

        # get the input and output data
        positions = np.array(datum["p_in"])
        velocities = np.array(datum["v_in"])

        # save the cutoff for the input data before we extend it
        input_length = positions.shape[1]

        # extend by the output data
        positions = np.concatenate(
            [positions, np.array(datum["p_out"])], axis=1
        )
        velocities = np.concatenate(
            [velocities, np.array(datum["v_out"])], axis=1
        )

        # get the number of timesteps
        _, num_timesteps, _ = positions.shape

        # homogenize the 2D data
        positions_homogenous = self.homogenize_matrix(positions)
        velocities_homogenous = self.homogenize_matrix(velocities)
        lanes_homogenous = self.homogenize_matrix(lanes)
        lane_norms_homogenous = self.homogenize_matrix(lane_norms)

        # translation of the lanes needs to be done differently:
        # - better RAM usage
        # - possible use during inference
        lanes_homogenous -= positions_homogenous[agent_index, 0]

        # create a list of transformation matrices that center all points
        # around the target agent
        translation_transforms = np.eye(3)[np.newaxis].repeat(
            num_timesteps, axis=0
        )

        # get the target agent's positions
        target_positions = positions[agent_index]

        # get the offsets that should be experienced by lanes (which are not
        # updated at every timestamp).
        # needs to be done before centering positions around agent (diff -> 0)
        offsets_homogenous = np.diff(positions_homogenous[agent_index], axis=0)
        first_offset = np.array([0, 0, 0])
        offsets_homogenous = np.vstack([first_offset, offsets_homogenous])

        # set the translation component of the transformation matrices
        translation_transforms[:, :2, 2] -= target_positions

        # apply the translation transformation to all points
        positions_homogenous = np.matmul(
            translation_transforms, positions_homogenous[:, :, :, np.newaxis]
        )

        # get rid of the last dimension
        # TODO: why does the extra dimension exist?
        positions_homogenous = positions_homogenous[:, :, :, 0]

        # create the rotation transform (key difference: only one needed)
        rotation_transforms = np.eye(3)

        # get the angle from the target agent's first input position to the
        # final input position
        first_position = positions[agent_index, 0]
        last_position = positions[agent_index, -1]

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

        # apply the rotation transformation to:
        # positions, velocities, lanes, lane_norms
        positions_homogenous = np.matmul(
            rotation_transforms, positions_homogenous[:, :, :, np.newaxis]
        )
        velocities_homogenous = np.matmul(
            rotation_transforms, velocities_homogenous[:, :, :, np.newaxis]
        )
        lanes_homogenous = np.matmul(
            rotation_transforms, lanes_homogenous[:, :, np.newaxis]
        )
        lane_norms_homogenous = np.matmul(
            rotation_transforms, lane_norms_homogenous[:, :, np.newaxis]
        )
        offsets_homogenous = np.matmul(
            rotation_transforms, offsets_homogenous[:, :, np.newaxis]
        )

        # dehomogenize the data
        positions = positions_homogenous[:, :, :2, 0]
        velocities = velocities_homogenous[:, :, :2, 0]
        lanes = lanes_homogenous[:, :2, 0]  # one less dimension b/c no times
        lane_norms = lane_norms_homogenous[:, :2, 0]
        offsets = offsets_homogenous[:, :2, 0]

        # print(lanes)

        # update the positions in the datum
        datum["p_in"] = positions[:, :input_length]
        datum["v_in"] = velocities[:, :input_length]
        datum["p_out"] = positions[:, input_length:]
        datum["v_out"] = velocities[:, input_length:]

        # update the lane positions
        datum["lane"] = lanes
        datum["lane_norm"] = lane_norms

        # save the scene offsets to invert the transformation
        datum["offsets"] = offsets

        # we want to save the offsets to invert the transformation as well as
        # update the lane positions

        return datum

    def invert(self, datum):
        """
        TODO: implement lol
        Inverts the position inputs and outputs in the datum by removing the
        offsets.

        Args:
            datum (dict): Dictionary representing a single data point.

        Returns:
            dict: Updated datum with inverted position inputs and outputs.
        """
        # get the input and output data
        position_inputs = datum["p_in"]
        position_outputs = datum["p_out"]

        # get the mask
        agent_mask = datum["car_mask"]

        # remove the offset iff the agent is present
        position_inputs[agent_mask] -= self.input_offsets
        position_outputs[agent_mask] -= self.output_offsets

        # update the positions in the datum
        datum["p_in"] = position_inputs
        datum["p_out"] = position_outputs

        return datum
