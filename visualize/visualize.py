"""
Module for visualizing the predictions of the model.
"""

import sys
import os

import yaml
import tqdm
import torch
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.animation as animation

sys.path.append("../")
import data_loader.data_loaders as data

from models.base import BaseModel

import transformations.agent_center as AgentCenter
from transformations.model_preprocessing.pre_simple_rnn import preSimpleRNN


# open the config file
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


def update_plot(timestep, scenes, axs, preds=None):
    """
    Plots a scene to an axis at a given timestamp.
    """

    for scene, ax, preds in zip(scenes, axs, preds):
        ax.clear()  # Clear the current plot

        timestamp = timestep

        scene_timestamp = timestamp
        agent_index = np.where(scene["track_id"] == scene["agent_id"])[0][0]

        num_agents = len(scene["track_id"])

        if timestamp < len(scene["p_in"][0]):
            positions = np.array(scene["p_in"])
            prior_offset = np.zeros(2)
        else:
            positions = np.array(scene["p_out_transformed"])
            timestamp -= len(scene["p_in"][0])
            prior_offset = np.sum(scene["p_in"][agent_index], axis=0)

        agent_positions = positions[agent_index]
        positions = positions[:num_agents, timestamp, :]

        offset = np.cumsum(agent_positions, axis=0)
        total_offset = offset[timestamp] + prior_offset

        lane_positions = scene["lane"] - total_offset
        lane_norms = scene["lane_norm"]

        # get the lane norm angles
        lane_angles = np.arctan2(lane_norms[:, 1], lane_norms[:, 0])

        # get the indices of the lane angles greater than 0
        lane_indices = np.where(lane_angles > 0)[0]

        # filter out the lanes that are not in the correct direction
        lane_positions = lane_positions[lane_indices]
        lane_norms = lane_norms[lane_indices]

        positive_lanes = np.where(lane_positions[:, 1] > -5)[0]
        lane_positions = lane_positions[positive_lanes]
        lane_norms = lane_norms[positive_lanes]

        # order by distance to 0, 0
        lane_distances = np.linalg.norm(lane_positions, axis=1)
        lane_indices = np.argsort(lane_distances)[:20]
        lane_positions = lane_positions[lane_indices]
        lane_norms = lane_norms[lane_indices]

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
        lim = 20
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # set the aspect ratio of the plot to be equal
        ax.set_aspect("equal")


def animate(scenes, preds=None, filename="animation.gif"):
    """
    Animates a scene with optional predictions.
    """

    # num cols is 3
    ncols = 3

    nrows = len(scenes) // ncols
    nrows += 1 if len(scenes) - nrows * ncols > 0 else 0

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(3 * len(scenes), 10)
    )

    # flatten the axes
    axs = axs.flatten()

    num_timestamps = len(scenes[0]["p_in"][0]) + len(scenes[0]["p_out"][0])

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=tqdm.tqdm(range(num_timestamps)),
        fargs=(scenes, axs, preds),
        interval=20,  # Time in milliseconds between frames
        repeat=True,
    )

    # Save the animation as a GIF
    ani.save(filename, writer="pillow", fps=10)


model_config = config["model"]
data_config = config["data"]
model_input_loader, _ = data.create_data_loader(
    model_config, data_config, train=True
)


def transform(x):
    """
    transforms x, TODO needs to be 'modernized'
    """
    x["inverse"] = None
    x = AgentCenter.apply(x)
    x = preSimpleRNN.apply(x, data_config)
    return x


computer_name = os.uname()[1]
train_path = data_config[computer_name]["train_path"]
model_input_dataset = data.ArgoverseDataset(
    train_path, transform
)

# ground truth dataset for visualization
visualization_dataset = data.ArgoverseDataset(
    train_path, AgentCenter.apply
)


def move_inputs_to_device(inputs, device):
    """
    Move the inputs to the device.

    Args:
        inputs (list): List of inputs to be moved to the device.
        device (torch.device): The device to move the inputs to.

    Returns:
        list: List of inputs moved to the device.
    """

    # . FIXME there's gotta be a better way to do this
    # inputs on device
    input_tensors = []
    for input_tensor in inputs:
        if isinstance(input_tensor, tuple):
            input_tensor = list(input_tensor)

        # if its the lanes, it will appear as list[tensor]
        if isinstance(input_tensor, list):
            lanes = input_tensor[0].to(device)
            final_lanes = [ip.to(device) for ip in input_tensor[1]]
            input_tensor = (lanes, final_lanes)
        elif input_tensor is not None:
            input_tensor = input_tensor.to(device)

        input_tensors.append(input_tensor)
    return input_tensors


def get_prediction(model_cfg, data_cfg, idx):
    """
    Get the prediction for a given index.
    """
    model_input = model_input_dataset[idx]

    # collate into batch, TODO fix prediction_correction
    inputs, _, _, _ = data.collate([model_input])

    model = BaseModel(model_cfg, data_cfg)

    # model = SimpleRNN(model_cfg, data_cfg)
    path = f"../models/saved_weights/{model_cfg['name']}.pth"
    model.load_state_dict(torch.load(path, weights_only=True))
    model.to("cpu")

    model.teacher_forcing_freq = 0

    inputs = move_inputs_to_device(inputs, "cpu")

    # predict
    model.eval()
    with torch.no_grad():
        prediction, _ = model(inputs)

    # already in the correct grame, so just cumsum is needed to get the
    # positions relative to the last p_in position, which is known
    prediction = prediction.reshape(30, 2)
    prediction = prediction.numpy()
    prediction = np.cumsum(prediction, axis=0)

    # correct the prediction
    return prediction


# INDEX = 200 # left turn
# INDEX = 100 # lane change
# INDEX = 20  # straight
# INDEX = 40000  # great left turn
# INDEX = 5 # impressive overtake
# INDEX = 1000 # slow down to avoid crash
# INDEX = 500 # odd scene with lots of entities

# indices = [100, 40000, 5, 500, 200, 1000, 20, 300, 500]
indices = [100, 40000, 5, 500, 200, 1000]
# indices = [201,202,203,204,205,206]
# indices = [301,302,303,304,305,306]
# indices = [401,402,403,404,405,406]
# indices = [501,502,503,504,505,506]
# indices = [601,602,603,604,605,606]
# indices = [701,702,703,704,705,706]
# indices = [801,802,803,804,805,806]

viz_scenes = [visualization_dataset[i] for i in indices]
predictions = [get_prediction(model_config, data_config, i) for i in indices]

animate(viz_scenes, predictions, filename="animation.gif")
