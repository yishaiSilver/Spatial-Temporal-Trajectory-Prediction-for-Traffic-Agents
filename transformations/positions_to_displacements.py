import numpy as np


# TODO: FIXME: this basically gives the velocity?

TIMESTAMP_DIM = 2

class PositionToDisplacement():
    """
    Applies agent-centered transformation to the given batch_data.

    Methods:
        apply(batch_data):
            Apply transformation to the given batch_data.

        invert(batch_data):
            Undo the transformation in the batch data.
    """

    def __init__(self):
        super().__init__()
        return

    def apply(self, datum):
        """ apply """

        # get the input and output data
        position_inputs = np.array([datum["p_in"]])
        position_outputs = np.array([datum["p_out"]])

        # save the length of the input data:
        # shape: _, num_agents, num_timesteps, num_dims TODO: find meaning of 0
        num_input_timestamps = position_inputs.shape[TIMESTAMP_DIM]

        # concatenate the input and output
        all_positions = np.concatenate((position_inputs,
                                        position_outputs),
                                        axis=TIMESTAMP_DIM)

        # calculate the displacement
        displacement = np.diff(all_positions, axis=TIMESTAMP_DIM)

        # split the displacement back into input and output
        displacement_inputs = displacement[:, :num_input_timestamps]
        displacement_outputs = displacement[:, num_input_timestamps:]

        # update the batch data
        datum["p_in"] = displacement_inputs
        datum["p_out"] = displacement_outputs

        return datum

    def invert(self, datum):
        """ invert """

        # get the input and output data
        displacement_inputs = np.array([datum["p_in"]])
        displacement_outputs = np.array([datum["p_out"]])

        # save the length of the input data
        num_input_timestamps = displacement_inputs.shape[TIMESTAMP_DIM]

        # concatenate the input and output
        all_displacements = np.concatenate((displacement_inputs,
                                            displacement_outputs),
                                            axis=TIMESTAMP_DIM)

        # calculate the displacement
        positions = np.cumsum(all_displacements, axis=TIMESTAMP_DIM)

        # split the displacement back into input and output
        position_inputs = positions[:, :num_input_timestamps]
        position_outputs = positions[:, num_input_timestamps:]

        # update the batch data
        datum["p_in"] = position_inputs
        datum["p_out"] = position_outputs

        return datum
