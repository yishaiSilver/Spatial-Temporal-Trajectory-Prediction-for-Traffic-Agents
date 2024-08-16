"""
Train Spatiotemporal Trajectory Prediction Model for Traffic Agents.

Args:
    config (str): Path to the configuration file.

Returns:
    None

Raises:
    FileNotFoundError: If the configuration file is not found.
"""

import argparse
import torch
from torch import nn
import numpy as np
import data_loader.data_loaders as data
# import model.loss as module_loss
# import model.metric as module_metric

import model.model as model_arch  # tod: don't like name

# from trainer import Trainer
# from utils import prepare_device
import yaml
import tqdm

COUNT_MOVING_AVERAGE = 10

def train_epoch(model, optimizer, loss_fn, data_loader):
    """
    Trains the model for one epoch using the given optimizer, loss function, and data loader.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        loss_fn (callable): The loss function used to compute the loss between model outputs and labels.
        data_loader (torch.utils.data.DataLoader): The data loader providing the training data.

    Returns:
        None
    """
    model.train(True)

    losses = []
    iterator = tqdm.tqdm(data_loader, total=int(len(data_loader)))

    for batch_data in iterator:
        inputs, labels, prediction_correction = batch_data

        optimizer.zero_grad()

        predictions = model(inputs)

        predictions, labels = prediction_correction(predictions, labels)

        loss = loss_fn(predictions, labels)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())
        if len(losses) > COUNT_MOVING_AVERAGE:
            losses.pop(0)

        iterator.set_postfix_str(
            f"avg. loss={sum(losses)/len(losses)}"
        )  # tod easy optimize


def validate_epoch(model, loss_fn, data_loader):
    """
    Validates the model for one epoch using the given loss function and data loader.

    Args:
        model (torch.nn.Module): The model to be validated.
        loss_fn (torch.nn.Module): The loss function to calculate the validation loss.
        data_loader (torch.utils.data.DataLoader): The data loader containing the validation data.

    Returns:
        float: The total validation loss for the epoch.
    """

    model.eval()

    val_loss = 0.0
    iterator = tqdm.tqdm(data_loader, total=int(len(data_loader)))
    with torch.no_grad():
        for batch_data in iterator:
            inputs, labels = batch_data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

    return val_loss


def main(main_config):
    """
    Train the spatiotemporal trajectory prediction model for traffic agents.

    Args:
        config (dict): Configuration dictionary containing model, optimizer, loss, data, and num_epochs.

    Returns:
        None
    """

    # Rest of the code...
    # get the model
    # model_config = main_config['model']
    model = model_arch.RNN()  # tod: magic line (sort of)

    # get the optimizer
    # optimizer_config = main_config['optimizer']
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9
    )  # tod: magic line

    # get the loss
    # loss_config = main_config['loss']
    loss_fn = nn.MSELoss()  # tod: magic line

    # get the data
    data_config = main_config["data"]
    train_loader = data.create_data_loader(data_config, train=True)
    val_loader = data.create_data_loader(data_config, train=False)

    # get the number of epochs:
    # num_epochs = main_config['num_epochs']
    num_epochs = 3

    best_val_loss = np.inf
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch}:")

        train_epoch(model, optimizer, loss_fn, train_loader)
        validation_loss = validate_epoch(model, loss_fn, val_loader)

        # save the model if it's the best
        if validation_loss < best_val_loss:
            print(f"Best Validation loss: {validation_loss}")
            best_val_loss = validation_loss
            model_path = "model"
            torch.save(model.state_dict(), model_path)


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
