import torch
import numpy as np

class Collate():
    """
    collate class (convert batch -> input tensors)
    """

    def __init__(self, collate_keys: list[str]):
        """ init """

        self.keys = collate_keys

        return

    def apply(self, batch_data):
        """
        Apply the collate transformation to the given batch_data.

        Args:
            batch_data (list): List of dictionaries representing the batch data.

        Returns:
            Tensor: Transformed batch data, ready for input to the model.
        """

        # aggregate what we want
        desired_input_data = []

        # get the desired keys (converted to tensors)
        for key in self.keys:
            np_array = np.array([datum[key] for datum in batch_data])
            tensors = torch.tensor(np_array)
            desired_input_data.append(tensors)

        # combine the tensors
        desired_data = torch.stack(desired_input_data)
        
        # remove the first dimension
        desired_data = desired_data.squeeze(0)

        # chosen output data
        output_np = np.array([datum["p_out"] for datum in batch_data])
        output_tensor = torch.tensor(output_np)

        return desired_data, output_tensor

    # TODO: implement
    # def invert(self, batch_data):
    #     """
    #     TODO: implement
    #     Inverts the collate transformation in the batch data.

    #     Args:
    #         Tensor: List of dictionaries representing the batch data.

    #     Returns:
    #         list(dict): Convert a tensor to a batch of data
    #     """
        
    #     # convert the tensor back to a batch of data
    #     batch_data = batch_data.tolist()

    #     return