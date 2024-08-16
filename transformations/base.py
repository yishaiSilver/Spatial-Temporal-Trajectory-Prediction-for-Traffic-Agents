# transforms.py
# from typing import Any
# import torchvision.transforms as T

class BaseTransformation():
    """ class
    """
    def __init__(self):
        """ init """
        return

    def apply(self, datum):
        """ apply """

        datum["labels"] = datum["p_out"]
        datum["prediction_correction"] = None

        return datum

    def prediction_correction(self, predictions, labels):
        """ prediction_correction """
        return predictions, labels
