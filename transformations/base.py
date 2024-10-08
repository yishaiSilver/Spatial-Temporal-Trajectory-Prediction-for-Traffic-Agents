# transforms.py
# from typing import Any
# import torchvision.transforms as T


import transformations.agent_centered_transformations as AgentCenter
from transformations.model_preprocessing.pre_simple_mlp import preSimpleMLP
from transformations.model_preprocessing.pre_simple_rnn import preSimpleRNN


class BaseTransformation:
    """class"""

    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config

    def __call__(self, x):
        return self.forward_transform(x, self.model_config, self.data_config)

    def inverse_transform(self, batch_predictions, meta):
        """
        Apply the prediction correction to the data.
        """

        model_name = self.model_config["name"]
        transforms = self.data_config["transforms"]

        # print("inverse_transform")

        # # updating name for readability
        x = batch_predictions
        # meta = batch_metadata

        # inverse pass through whatever model-specific transformations are needed
        if model_name == "SimpleMLP":
            x = preSimpleMLP.inverse(x, meta)
        elif model_name == "SimpleRNN":
            x = preSimpleRNN.inverse(x, meta)

        # inverse pass through whatever model-agnostic transformations are needed
        if transforms is not None:
            # perform whatever additional transformations are needed
            if "AgentCenter" in transforms:
                x = AgentCenter.inverse(x, meta)

        return x

    def forward_transform(self, x, model_config, data_config):
        """
        Apply the forward transformation to the data.

        """

        # save configs for later use
        self.model_config = model_config
        self.data_config = data_config

        model_name = model_config["name"]
        transforms = data_config["transforms"]

        x["inverse"] = self.inverse_transform

        # forward pass through whatever model-agnostic transformations are needed
        if transforms is not None:
            # perform whatever additional transformations are needed
            if "AgentCenter" in transforms:
                x = AgentCenter.apply(x)
            

        # forward pass through whatever model-specific transformations are needed
        if model_name == "SimpleMLP":
            x = preSimpleMLP.apply(x, data_config)
        elif model_name == "SimpleRNN":
            x = preSimpleRNN.apply(x, data_config)
        return x


# class TransformFunction:

#     @staticmethod
#     def inverse(batch_predictions, batch_metadata):
#         """prediction_correction"""
#         _ = batch_metadata  # unused at this base level

#         return batch_predictions
