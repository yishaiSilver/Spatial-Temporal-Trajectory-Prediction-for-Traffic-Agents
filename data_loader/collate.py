import torch
import numpy as np


class Collate:
    """
    collate class (convert batch -> input tensors)
    """

    def __init__(self):
        """init"""
        return

    def apply(self, batch_data):
        """
        Apply the collate transformation to the given batch_data.

        Args:
            batch_data (list): List of dictionaries representing the batch data.

        Returns:
            Tensor: Transformed batch data, ready for input to the model.
        """

        batch_inputs = []
        batch_labels = []
        batch_prediction_correction = []
        batch_correction_metadata = []

        for datum in batch_data:
            model_input, label, prediction_correction, metadata = datum

            batch_inputs.append(model_input)
            batch_labels.append(label)
            batch_prediction_correction.append(prediction_correction)
            batch_correction_metadata.append(metadata)

        # print(batch_inputs)

        inputs = np.array(batch_inputs)
        labels = np.array(batch_labels)
    
        # we only need one function reference because all data should have 
        # the same correction function
        prediction_correction = batch_prediction_correction[0]

        input = torch.tensor(inputs, dtype=torch.float32)

        # TODO device
        input.to('cuda')

        # convert all inputs to tensors
        inputs = tuple(torch.tensor(inputs, dtype=torch.float32))

        # convert all labels to tensors
        labels = torch.tensor(labels, dtype=torch.float32)

        return inputs, labels, prediction_correction, batch_correction_metadata
