"""
Module used to display test information of matrices
"""
import sys
sys.path.append("../")

import os

import yaml
import tqdm
import torch
import numpy as np
import data_loader.data_loaders as data

from models.base import BaseModel

import transformations.agent_center as AgentCenter
from transformations.model_preprocessing.pre_simple_rnn import preSimpleRNN

from models.lanes.generate_map_matrix import generate_numpy

# convert from n x n x 2 to n x n and plot with matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# open the config file
with open("../config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# get the configs
data_config = config["data"]
data_config["xps15"]["shuffle"] = False
data_config["xps15"]["batch_size"] = 1
data_config["xps15"]["num_workers"] = 1
model_config = config["model"]

# get the data
train_loader, val_loader = data.create_data_loader(
    model_config, data_config, train=True
)

inputs, _, _, _ = next(iter(train_loader))

x, lanes_w_loss, neighbors, teacher_forcing = inputs

# lanes_w_loss [(batch, ts, n, 2), ortho_loss]
lanes = lanes_w_loss[0][0]

# single ts
def plot_ts(ts, lanes, ax):
    """
    Plots a single timestep on ax
    """
    ax.clear()

    lane = lanes[ts]
    map_matrix = generate_numpy(lane, size=20, granularity=0.5)

    map_matrix = map_matrix[:, :, 0] + map_matrix[:, :, 1]

    # round to 1
    map_matrix = np.round(map_matrix)

    ax.imshow(map_matrix, origin='lower')

fig, ax = plt.subplots(figsize=(7, 7))

ani = animation.FuncAnimation(
    fig, plot_ts, frames=range(0, 19), fargs=(lanes, ax), interval=100
)

ani.save("test.gif", writer="pillow", fps=10)
