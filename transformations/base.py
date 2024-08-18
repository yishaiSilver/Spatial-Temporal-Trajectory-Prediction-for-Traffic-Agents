# transforms.py
# from typing import Any
# import torchvision.transforms as T

class BaseTransformation():
    """ class
    """
    def __init__(self):
        """ init """

        self.prior_prediction_correction = None

        return

    def apply(self, datum):
        """ apply """

        # datum["labels"] = datum["p_out"]
        datum["prediction_correction"] = self.prediction_correction

        return datum

    def prediction_correction(self, batch_predictions, batch_metadata):
        """ prediction_correction """
        _ = batch_metadata # unused at this base level

        return batch_predictions
