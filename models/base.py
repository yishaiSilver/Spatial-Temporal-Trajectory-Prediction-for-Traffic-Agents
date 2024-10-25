"""
Module used to construct the base model class.
"""

import os

import torch
import torch.nn as nn

from models.a_simple_mlp import SimpleMLP
from models.b_simple_rnn import SimpleRNN
from models.c_seq2seq import Seq2Seq


class BaseModel(nn.Module):
    """
    Base class for all models.
    """

    def __init__(
        self,
        model_config: dict,
        data_config: dict,
        load_model=False,
        testing=False,
    ):
        """
        Constructor for the SimpleMLP class.

        args:
            model_config (dict): dictionary containing the model configuration.
            data_config (dict): dictionary containing the data configuration.
        """

        super().__init__()

        self.device = model_config["device"]
        self.model_config = model_config
        self.data_config = data_config

        model_type = model_config["name"]

        if model_type == "SimpleMLP":
            self.model = SimpleMLP(model_config, data_config)
        elif model_type == "SimpleRNN":
            self.model = SimpleRNN(model_config, data_config)
            if testing:
                self.model.teacher_forcing_freq = 0
        elif model_type == "Seq2Seq":
            self.model = Seq2Seq(model_config, data_config)
            if testing:
                self.model.teacher_forcing_freq = 0

        if load_model:
            path = f"models/saved_weights/{model_type}.pth"

            # print whether the path exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model weights not found at {path}")

            self.model.load_state_dict(torch.load(path, weights_only=True))

    def forward(self, x):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """

        return self.model(x)
