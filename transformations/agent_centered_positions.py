import sys
# from typing import Any
# import torchvision.transforms as T

sys.path.append(
    ".."
)  # Add the parent directory of 'transformations' to the Python path
sys.path.append(
    "../transformations"
)  # Add the 'transformations' directory to the Python path
from base import BaseTransformation

import numpy as np


# TODO: FIXME: IMPORTANT: this will basically zero-out the agent's positions

class AgentCenter(BaseTransformation):
    """
    Applies agent-centered transformation to the given batch_data.

    Methods:
        apply(batch_data):
            Apply agent-centered transformation to the given batch_data.

        invert(batch_data):
            Inverts the position inputs and outputs in the batch data by removing the offsets.
    """

    def __init__(self):
        super().__init__()

        self.input_offsets = None
        self.output_offsets = None

        return

    def apply(self, batch_data):
        """
        Apply agent-centered transformation to the given batch_data.

        Args:
            batch_data (list): List of dictionaries representing the batch data.

        Returns:
            list: Transformed batch_data with updated positions.
        """

        # extract all of the agent_ids from the batch_data
        agent_ids = np.array([datum["agent_id"] for datum in batch_data])

        # get the input and output data
        position_inputs = np.array([datum["p_in"] for datum in batch_data])
        position_outputs = np.array([datum["p_out"] for datum in batch_data])

        # get the offsets
        self.input_offsets = -1 * position_inputs[:, agent_ids]
        self.output_offsets = -1 * position_outputs[:, agent_ids]

        # perform the offset
        position_inputs += self.input_offsets
        position_outputs += self.output_offsets

        # update the positions in the batch_data
        for i, datum in enumerate(batch_data):
            datum["p_in"] = position_inputs[i]
            datum["p_out"] = position_outputs[i]

        return batch_data

    def invert(self, batch_data):
        """
        Inverts the position inputs and outputs in the batch data by removing the offsets.

        Args:
            batch_data (list): A list of dictionaries representing the batch data.
        Returns:
            list: Updated batch data with inverted position inputs and outputs.
        """
        # get the input and output data
        position_inputs = np.array([datum["p_in"] for datum in batch_data])
        position_outputs = np.array([datum["p_out"] for datum in batch_data])

        # remove the offset
        position_inputs -= self.input_offsets
        position_outputs -= self.output_offsets

        # update the positions in the batch_data
        for i, datum in enumerate(batch_data):
            datum["p_in"] = position_inputs[i]
            datum["p_out"] = position_outputs[i]

        return batch_data
