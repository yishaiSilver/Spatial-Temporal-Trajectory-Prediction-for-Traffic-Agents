import matplotlib.pyplot as plt


def plot_timestamp(scene, timestamp):
    positions = scene["p_in"][timestamp]

    # remove 0, 0 positions
    positions = positions[positions[:, 0] != 0]

    lane_positions = scene["lane_positions"]
    lane_norms = scene["lane_norms"]

    # plot the lanes
    for lane_position, lane_norm in zip(lane_positions, lane_norms):
        directional_position = lane_position + lane_norm
        plt.arrow(
            lane_position[0], lane_position[1], 
            directional_position[0], directional_position[1], 
            head_width=0.5)

    # plot the agents
    for position in positions:
        plt.scatter(position[0], position[1])
    
    plt.show()

