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
import torch.multiprocessing as tmp
import numpy as np
import data_loader.data_loaders as data

import logging
import coloredlogs

# import model.loss as module_loss
# import model.metric as module_metric


from models.a_simple_mlp import SimpleMLP
from models.b_simple_rnn import SimpleRNN

from utils.logger_config import logger

# from trainer import Trainer
# from utils import prepare_device
import yaml
import tqdm

COUNT_MOVING_AVERAGE = 250


def train_epoch(epoch, model, optimizer, loss_fn, data_loader, model_config):
    """
    Trains the model for one epoch using the given optimizer, loss function,
    and data loader.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer model's parameters.
        loss_fn (callable):  loss function
        data_loader (torch.utils.data.DataLoader): Data loader providing data.

    Returns:
        None
    """

    # load the best model if we're not on the first epoch
    if epoch != 0:
        model_path = f"models/saved_weights/{model_config['name']}.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True))

    # model.train(True)

    device = model.device

    moving_avg_sum = 0
    moving_avg_losses = []

    iterator = tqdm.tqdm(data_loader, total=int(len(data_loader)))

    for batch_data in iterator:
        inputs, labels, prediction_correction, metadata = batch_data

        # inputs on device
        inputs = tuple(
            input_tensor.to(device) if input_tensor is not None else None 
            for input_tensor in inputs
        )
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        predictions = prediction_correction(predictions, metadata)

        loss = loss_fn(predictions, labels)

        loss.backward()

        optimizer.step()

        moving_avg_losses.append(loss.item())
        moving_avg_sum += loss.item()
        if len(moving_avg_losses) > COUNT_MOVING_AVERAGE:
            first_loss = moving_avg_losses.pop(0)
            moving_avg_sum -= first_loss

        moving_avg_mse = moving_avg_sum / len(moving_avg_losses)
        moving_avg_rmse = np.sqrt(moving_avg_mse)

        iterator.set_postfix_str(
            f"avg. loss={moving_avg_rmse:.5f}"
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

    # model.eval()

    device = model.device

    val_loss = 0.0
    iterator = tqdm.tqdm(data_loader, total=int(len(data_loader)))

    moving_avg_sum = 0
    moving_avg_losses = []

    with torch.no_grad():
        for batch_data in iterator:
            inputs, labels, prediction_correction, metadata = batch_data

            # move to the device
            inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            labels = labels.to(device)

            outputs = model(inputs)

            # postprocess the outputs
            outputs = prediction_correction(outputs, metadata)

            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

            moving_avg_losses.append(loss.item())
            moving_avg_sum += loss.item()
            if len(moving_avg_losses) > COUNT_MOVING_AVERAGE:
                first_loss = moving_avg_losses.pop(0)
                moving_avg_sum -= first_loss

            moving_avg_mse = moving_avg_sum / len(moving_avg_losses)
            moving_avg_rmse = np.sqrt(moving_avg_mse)

            iterator.set_postfix_str(
                f"avg. loss={moving_avg_rmse:.5f}"
            )  # tod easy optimize

    return val_loss / len(data_loader)


def main(main_config):
    """
    Train the spatiotemporal trajectory prediction model for traffic agents.

    Args:
        config (dict): Configuration dictionary containing model, optimizer,
            loss, data, and num_epochs.

    Returns:
        None
    """

    # initialize logging
    logging.basicConfig()

    # initialize cuda multiprocessing to avoid error
    # tmp.set_start_method('spawn', force=True)

    # get the configs
    data_config = main_config["data"]
    model_config = main_config["model"]

    # get the data
    train_loader, val_loader = data.create_data_loader(
        model_config, data_config, train=True
    )
    # val_loader = data.create_data_loader(model_config, data_config, train=False)

    # Rest of the code...
    # get the model
    # model = SimpleMLP(model_config, data_config)  # TODO: magic line (sort of)
    model = SimpleRNN(model_config, data_config)
    # switch to config file spec.

    # get the optimizer
    # optimizer_config = main_config['optimizer']
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.0001, momentum=0.1, weight_decay=0.001
    )  # tod: magic line

    # get the loss
    # loss_config = main_config['loss']
    loss_fn = nn.MSELoss()  # tod: magic line

    # get the number of epochs:
    # num_epochs = main_config['num_epochs']
    num_epochs = main_config["num_epochs"]

    best_val_loss = np.inf
    for epoch in range(num_epochs):
        logger.info("EPOCH %d", epoch)

        train_epoch(
            epoch, model, optimizer, loss_fn, train_loader, model_config
        )
        validation_loss = validate_epoch(model, loss_fn, val_loader)

        # MSE -> RMSE
        validation_loss = np.sqrt(validation_loss)

        # save the model if it's the best
        if validation_loss < best_val_loss:
            logger.info("\033[92mSaving. Val. loss: %f\033[0m", validation_loss)
            best_val_loss = validation_loss
            model_path = f"models/saved_weights/{model_config['name']}.pth"
            torch.save(model.state_dict(), model_path)
        else:
            logger.info(
                "\033[91mNot saving. Val. loss: %f\033[0m", validation_loss
            )


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
