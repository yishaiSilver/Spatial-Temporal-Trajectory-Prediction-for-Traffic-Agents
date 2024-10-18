"""
Module for visualizing the predictions of the model.
"""

import sys

sys.path.append("../")

import yaml
import tqdm
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import torch


import data_loader.data_loaders as data
from models.b_simple_rnn import SimpleRNN
import transformations.agent_centered_transformations as AgentCenter
from transformations.model_preprocessing.pre_simple_rnn import preSimpleRNN

# open the config file
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


def update_plot(timestep, scenes, axs, preds=None):
    """
    Plots a scene to an axis at a given timestamp.
    """

    for scene, ax in zip(scenes, axs):
        ax.clear()  # Clear the current plot

        timestamp = timestep

        scene_timestamp = timestamp
        agent_index = np.where(scene["track_id"] == scene["agent_id"])[0][0]

        num_agents = len(scene["track_id"])

        if timestamp < len(scene["p_in"][0]):
            predicting = "Input Data"
            positions = np.array(scene["p_in"])

            prior_offset = np.zeros(2)
        else:
            predicting = "Output Data"
            positions = np.array(scene["p_out_transformed"])
            timestamp -= len(scene["p_in"][0])

            # get the total sum of p_in[agent_index] to get the offset
            prior_offset = np.sum(scene["p_in"][agent_index], axis=0)

        agent_positions = positions[agent_index]

        positions = positions[:num_agents, timestamp, :]

        offset = np.cumsum(agent_positions, axis=0)
        total_offset = offset[timestamp] + prior_offset

        lane_positions = scene["lane"] - total_offset
        lane_norms = scene["lane_norm"]

        # Plot the lanes
        for lane_position, lane_norm in zip(lane_positions, lane_norms):
            ax.arrow(
                lane_position[0],
                lane_position[1],
                lane_norm[0],
                lane_norm[1],
                width=0.05,
                color="black",
            )

        # Plot the agents
        colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "pink",
            "olive",
            "cyan",
        ]
        for i, position in enumerate(positions):
            color = colors[i % len(colors)]
            x, y = position
            s = 25
            if i == agent_index:
                x, y = 0, 0
                s = 50

            ax.scatter(x, y, color=color, s=s)

        # if the predictions are provided, plot them
        if preds is not None:
            if scene_timestamp >= len(scene["p_in"][0]):
                pred_x, pred_y = preds[timestamp] - offset[timestamp]
                ax.scatter(pred_x, pred_y, color="black", s=50)

        # prepend space if in single digits
        scene_timestamp = str(scene_timestamp).rjust(2, "0")

        # Set the axis limits
        lim = 30
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # set the aspect ratio of the plot to be equal
        ax.set_aspect("equal")


def animate(scene, preds=None, filename="animation.gif"):
    """
    Animates a scene with optional predictions.
    """

    fig, axs = plt.subplots(ncols=2)
    num_timestamps = len(scene["p_in"][0]) + len(scene["p_out"][0])

    scenes = [scene, scene]

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=range(num_timestamps),
        fargs=(scenes, axs, preds),
        interval=10,  # Time in milliseconds between frames
        repeat=True,
    )

    # Save the animation as a GIF
    ani.save(filename, writer="pillow", fps=10)


model_config = config["model"]
data_config = config["data"]
model_input_loader, _ = data.create_data_loader(
    model_config, data_config, train=True, examine=False
)


def transform(x):
    """
    transforms x, TODO needs to be 'modernized'
    """
    x["inverse"] = None
    x = AgentCenter.apply(x)
    x = preSimpleRNN.apply(x, data_config)
    return x


model_input_dataset = data.ArgoverseDataset(
    data_config["train_path"], transform
)

# ground truth dataset for visualization
visualization_dataset = data.ArgoverseDataset(
    data_config["train_path"], AgentCenter.apply
)


def get_prediction(model_cfg, data_cfg, idx):
    """
    Get the prediction for a given index.
    """
    model_input = model_input_dataset[idx]

    # collate into batch, TODO fix prediction_correction
    inputs, _, _, _ = data.collate([model_input])

    model = SimpleRNN(model_cfg, data_cfg)
    path = f"../models/saved_weights/{model_cfg['name']}.pth"
    model.load_state_dict(torch.load(path, weights_only=True))
    model.to("cpu")

    inputs = tuple(
        input_tensor.to("cpu") if input_tensor is not None else None
        for input_tensor in inputs
    )

    # predict
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)

    # already in the correct grame, so just cumsum is needed to get the
    # positions relative to the last p_in position, which is known
    prediction = prediction.reshape(30, 2)
    prediction = prediction.numpy()
    prediction = np.cumsum(prediction, axis=0)

    # correct the prediction
    return prediction


# index = 200 # left turn
# index = 100 # lane change
# index = 20  # straight
index = 40000  # great left turn
# index = 5 # impressive overtake
# index = 1000 # slow down to avoid crash
# index = 500 # odd scene with lots of entities

visualization_scene = visualization_dataset[index]
predictions = get_prediction(model_config, data_config, index)

animate(visualization_scene, predictions, filename="animation.gif")
