"""
Module used to display test information of matrices
"""
import sys
sys.path.append("../")


import yaml
import data_loader.data_loaders as data




# convert from n x n x 2 to n x n and plot with matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# open the config file
with open("../config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# get the configs
data_config = config["data"]
data_config["xps15"]["shuffle"] = False
# data_config["xps15"]["batch_size"] = 1
data_config["xps15"]["num_workers"] = 1
model_config = config["model"]

# get the data
train_loader, val_loader = data.create_data_loader(
    model_config, data_config, train=True
)

inputs, _, _, _ = next(iter(train_loader))

x, lanes, neighbors, teacher_forcing = inputs

lanes, lanes_final = lanes

# lanes = generate_numpy(lanes, size=20, granularity=0.5)

# # single ts
def plot_ts(ts, lanes, ax):
    """
    Plots a single timestep on ax
    """
    ax.clear()

    # map for batch 0 at ts
    map_b_t = lanes[0, ts]

    map_matrix = map_b_t[0, :, :] + map_b_t[1, :, :]

    ax.imshow(map_matrix, origin='lower', vmin=0, vmax=1)

fig, ax = plt.subplots(figsize=(7, 7))

ani = animation.FuncAnimation(
    fig, plot_ts, frames=range(0, 19), fargs=(lanes, ax), interval=100
)

ani.save("lane_matrices.gif", writer="pillow", fps=10)
