"""
Saves a gif of a scene from the traffic prediction model.
"""

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml

import data_loader.data_loaders as data

colors = ["blue", "red", "green", "orange", "purple", "pink", "olive", "cyan"]


def update_plot(timestamp, scene, ax):
    """
    Update the plot with the current scene.
    """

    ax.clear()  # Clear the current plot

    scene_timestamp = timestamp

    num_agents = len(scene["track_id"])

    if timestamp < len(scene["p_in"][0]):
        predicting = "Input Data"
        positions = np.array(scene["p_in"])
        velocities = np.array(scene["v_in"])
    else:
        predicting = "Output Data"
        positions = np.array(scene["p_out"])
        velocities = np.array(scene["v_out"])
        timestamp -= len(scene["p_in"][0])

    positions = positions[:num_agents, timestamp, :]

    velocities = velocities[:num_agents, timestamp, :]

    if "offsets" in scene:
        offset = np.cumsum(scene["offsets"], axis=0)
        offset = offset[scene_timestamp]
        print(f"Found offset {offset}")
    else:
        offset = np.zeros(2)
    lane_positions = scene["lane"] - offset
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
    for i, position in enumerate(positions):
        color = colors[i % len(colors)]
        ax.scatter(position[0], position[1], color=color, s=10)

    # plot the velocities
    for i, velocity in enumerate(velocities):
        color = colors[i % len(colors)]
        ax.arrow(
            positions[i][0],
            positions[i][1],
            velocity[0],
            velocity[1],
            width=0.05,
            color=color,
        )

    # prepend space if in single digits
    scene_timestamp = str(scene_timestamp).rjust(2, "0")

    ax.set_title(f"{predicting}. Timestamp: {scene_timestamp}")

    # Set the x and y axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Set the axis limits
    lim = 30
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # set the aspect ratio of the plot to be equal
    ax.set_aspect("equal")


def animate(scene, filename="visualize/images/translated_to_agent.gif"):
    """
    Animate the scene.
    """
    fig, ax = plt.subplots()
    num_timestamps = len(scene["p_in"][0]) + len(scene["p_out"][0])
    # num_timestamps = len(scene["p_in"][0])
    # num_timestamps = 10
    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=range(num_timestamps),
        fargs=(scene, ax),
        interval=100,  # Time in milliseconds between frames
        repeat=True,
    )

    # Save the animation as a GIF
    ani.save(filename, writer="pillow", fps=10)

    # # Display the saved GIF
    # return HTML(ani.to_jshtml())


def main(config):
    """
    Visualize a scene for the traffic prediction model.
    """

    model_config = config["model"]
    data_config = config["data"]

    train_loader, _ = data.create_data_loader(
        model_config, data_config, train=True, examine=False
    )

    # Plot a scene: 10, 100, 3000
    scene = train_loader.dataset[100]
    print(scene["offsets"])
    animate(scene)
    # display(animation)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Traffic Prediction")
    args.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        type=str,
        help="config file path (default: config.yaml)",
    )
    args = args.parse_args()

    # get the configuration file
    config_path = args.config

    # open the config file
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    main(config)
