"""
Contains functions to load data in an organized manner.
"""

import os
import os.path

# import numpy
import pickle
from glob import glob

from torch.utils.data import Dataset, DataLoader

from data_loader.collate import Collate
from transformations.agent_centered_transformations import AgentCenter
from transformations.base import BaseTransformation

from transformations.model_preprocessing.pre_simple_mlp import preSimpleMLP

# from transformations.positions_to_displacements import PositionToDisplacement

# number of sequences in each dataset
# train:205942  val:3200 test: 36272
# sequences sampled at 10HZ rate


class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""

    def __init__(self, data_path: str, transform=None):
        """TODO: init"""
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform

        self.pkl_list = glob(os.path.join(self.data_path, "*"))
        self.pkl_list.sort()

    def __len__(self):
        """TODO: len"""
        return len(self.pkl_list)

    def __getitem__(self, idx):
        """getitem"""
        pkl_path = self.pkl_list[idx]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if self.transform:
            data = self.transform(data)

        return data


def create_data_loader(model_config, data_config, train=True, examine=False):
    """TODO: create_data_loader"""
    #   data_path: str, transforms=None, batch_size=4, shuffle=False, val_split=0.0, num_workers=1
    if train:
        data_path = data_config["train_path"]
    else:
        data_path = data_config["val_path"]

    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    model_name = model_config["name"]
    transforms = data_config["transforms"]

    def transform_fn(x):
        tf = BaseTransformation()
        x = tf.apply(x)

        if transforms is not None:
            # perform whatever additional transformations are needed
            if "AgentCenter" in transforms:
                tf = AgentCenter()
                x = tf.apply(x)

        # add per-model transformations here
        if model_name == "SimpleMLP":
            tf = preSimpleMLP(data_config)
            x = tf.apply(x)

        return x

    collate = Collate()
    collate_fn = collate.apply

    if examine:

        def noop(x):
            return x

        collate_fn = noop

    dataset = ArgoverseDataset(data_path, transform=transform_fn)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
