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
import yaml
import tqdm
from torchnet.meter import MovingAverageValueMeter

import data_loader.data_loaders as data

from models.base import BaseModel
from utils.optimizer import get_optimizer

from utils.logger_config import logger


COUNT_MOVING_AVERAGE = 250


def move_inputs_to_device(inputs, device):
    """
    Move the inputs to the device.

    Args:
        inputs (list): List of inputs to be moved to the device.
        device (torch.device): The device to move the inputs to.

    Returns:
        list: List of inputs moved to the device.
    """

    #. FIXME there's gotta be a better way to do this
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


def fde_loss(predictions, labels):
    """
    Calculates the FDE loss between the predictions and the labels.
    """

    prediction_final_ts = predictions[:, -1, :]
    labels_final_ts = labels[:, -1, :]

    fde = torch.norm(prediction_final_ts - labels_final_ts, p=2, dim=-1)

    return fde.mean()


def ade_loss(predictions, labels):
    """
    Calculates the ADE loss between the predictions and the labels.

    1/T * sum_t(||p_t - l_t||)

    """
    displacement_error = torch.norm(predictions - labels, p=2, dim=-1)
    ade = displacement_error.mean()

    return ade

def l1_regularization(model, lambda_l1=0.0001):
    """
    L1 regularization for the model.
    """

    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l1_norm *= lambda_l1

    return l1_norm

def l2_regularization(model, lambda_l2=0.0001):
    """
    L2 regularization for the model.
    """

    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    l2_norm *= lambda_l2

    return l2_norm

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

    rmse_meter = MovingAverageValueMeter(COUNT_MOVING_AVERAGE)
    ade_meter = MovingAverageValueMeter(COUNT_MOVING_AVERAGE)

    # load the best model if we're not on the first epoch
    if epoch != 0:
        model_path = f"models/saved_weights/{model_config['name']}.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True))

    # model.train(True)

    device = model.device

    iterator = tqdm.tqdm(
        data_loader,
        total=int(len(data_loader)),
        bar_format="{l_bar}{bar:50}{r_bar}{bar:-50b}",  # broken laptop screen :/
    )

    for i, batch_data in enumerate(iterator):
        # get rid of unused warning
        _ = i

        inputs, labels, prediction_correction, metadata = batch_data

        input_tensors = move_inputs_to_device(inputs, device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # if i > 20:
        #     with profile(activities=[
        #                 ProfilerActivity.CPU,
        #                 # ProfilerActivity.CUDA
        #                 ], record_shapes=True) as prof:
        #             with record_function("model_inference"):
        #                 outputs = model(input_tensors)

        #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

        #     exit()
        # i += 1

        predictions, ortho_loss = model(input_tensors)

        predictions = prediction_correction(predictions, metadata)

        # normal_loss = loss_fn(predictions, labels)
        ade = ade_loss(predictions, labels)
        fde = fde_loss(predictions, labels)
        # l1 = l1_regularization(model)
        l2 = l2_regularization(model)

        # want to optimize for both the normal loss and the orthogonality loss
        loss = ade + fde + ortho_loss + l2

        loss.backward()

        optimizer.step()

        # now just business as usual
        rmse = np.sqrt(loss_fn(predictions, labels).item())


        rmse_meter.add(rmse)
        ade_meter.add(ade.item())

        rmse = rmse_meter.value()[0]
        ade = ade_meter.value()[0]

        iterator.set_postfix_str(
            f"avg. RMSE={rmse:.5f}, avg. ADE={ade:.5f}"
        )

        if torch.isnan(rmse):
            logger.error("\033[91mNaN loss. Exiting.\033[0m")
            exit(1)


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

    rmse_meter = MovingAverageValueMeter(COUNT_MOVING_AVERAGE)

    device = model.device

    val_loss = 0.0
    iterator = tqdm.tqdm(
        data_loader,
        total=int(len(data_loader)),
        bar_format="{l_bar}{bar:50}{r_bar}{bar:-50b}",
    )

    with torch.no_grad():
        for batch_data in iterator:
            inputs, labels, correction, metadata = batch_data

            input_tensors = move_inputs_to_device(inputs, device)
            labels = labels.to(device)

            # with profile(activities=[
            #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function("model_inference"):
            #         outputs = model(inputs)

            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            # exit()

            outputs, _ = model(input_tensors)

            # postprocess the outputs
            outputs = correction(outputs, metadata)

            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

            rmse = np.sqrt(loss.item())

            rmse_meter.add(rmse)

            moving_avg_rmse = rmse_meter.value()[0]

            iterator.set_postfix_str(
                f"avg. RMSE={moving_avg_rmse:.5f}"
            )  # tod easy optimize20

            if torch.isnan(moving_avg_rmse):
                logger.error("\033[91mNaN loss. Exiting.\033[0m")
                exit(1)

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

    # get the configs
    data_config = main_config["data"]
    model_config = main_config["model"]

    # get the data
    train_loader, val_loader = data.create_data_loader(
        model_config, data_config, train=True
    )

    model = BaseModel(model_config, data_config)

    optimizer = get_optimizer(model, main_config["optimizer"])

    # get the loss
    # loss_config = main_config['loss']
    loss_fn = nn.MSELoss()

    num_epochs = main_config["num_epochs"]

    best_val_loss = np.inf
    # save the model if it's the best
    model_path = f"models/saved_weights/{model_config['name']}.pth"
    for epoch in range(num_epochs):
        # load the best model
        # model.load_state_dict(torch.load(model_path, weights_only=True))

        logger.info("EPOCH %d", epoch)

        train_epoch(
            epoch, model, optimizer, loss_fn, train_loader, model_config
        )
        validation_loss = validate_epoch(model, loss_fn, val_loader)

        # MSE -> RMSE
        validation_loss = np.sqrt(validation_loss)

        if validation_loss < best_val_loss:
            logger.info("\033[92mSaving. Val. loss: %f\033[0m", validation_loss)
            best_val_loss = validation_loss
            torch.save(model.state_dict(), model_path)

            # also save a text file with the validation loss
            with open(f"{model_path}.txt", "w", encoding="utf-8") as text_file:
                text_file.write(f"{validation_loss}")
        else:
            logger.info(
                "\033[91mNot saving. Val. loss: %f\033[0m", validation_loss
            )

            # load the best model
            model.load_state_dict(torch.load(model_path, weights_only=True))


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
