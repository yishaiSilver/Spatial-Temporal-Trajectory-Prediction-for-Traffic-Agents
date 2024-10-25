"""
Train Spatiotemporal Trajectory Prediction Model for Traffic Agents.

Args:
    config (str): Path to the configuration file.

Returns:
    None

Raises:
    FileNotFoundError: If the configuration file is not found.
"""

import csv

import argparse
import torch
import yaml
import tqdm

import data_loader.data_loaders as data
from models.base import BaseModel

# from utils.logger_config import logger


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
        # if its the lanes, it will appear as list[tensor]
        if isinstance(input_tensor, list):
            lanes = input_tensor[0].to(device)
            final_lanes = [ip.to(device) for ip in input_tensor[1]]
            input_tensor = (lanes, final_lanes)
        elif input_tensor is not None:
            input_tensor = input_tensor.to(device)

        input_tensors.append(input_tensor)
    return input_tensors


def main(main_config):
    """
    Train the spatiotemporal trajectory prediction model for traffic agents.

    Args:
        config (dict): Configuration dictionary containing model, optimizer,
            loss, data, and num_epochs.

    Returns:
        None
    """

    # get the configs
    data_config = main_config["data"]
    model_config = main_config["model"]

    print(model_config)

    # get the data
    test_loader, _ = data.create_data_loader(
        model_config, data_config, train=False
    )

    model = BaseModel(model_config, data_config, testing=True)

    model_path = f"models/saved_weights/{model_config['name']}.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))

    header = ["ID"] + [f"v{i}" for i in range(1, 61)]

    with open("output.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        with torch.no_grad():
            for datum in tqdm.tqdm(test_loader):
                inputs, _, correction, metadata = datum

                input_tensors = move_inputs_to_device(inputs, model.device)

                outputs, _ = model(input_tensors)

                print(outputs[2, :5])

                # outputs = outputs[0].unsqueeze(0)
                # metadata = [metadata[0]]

                outputs = correction(outputs, metadata)

                # Assuming metadata contains the IDs and outputs is a tensor
                for meta, output in zip(metadata, outputs):
                    scene_id = meta["scene_id"]
                    # print(output)
                    output = output.flatten()
                    output = output.tolist()
                    row = [scene_id] + output
                    writer.writerow(row)


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
