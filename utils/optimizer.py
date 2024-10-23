"""
Module used to construct optimizers via config specifications.
"""

import torch.optim as optim
from utils.logger_config import logger

def get_optimizer(model, optimizer_config):
    """
    Get the optimizer for the model based on the optimizer configuration.

    Args:
        model (nn.Module): The model for which the optimizer is being created.
        optimizer_config (dict): The configuration for the optimizer.

    Returns:
        torch.optim.Optimizer: The optimizer for the model.
    """

    optimizer_name = optimizer_config["name"]
    optimizer_params = optimizer_config["params"]

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), **optimizer_params)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

    optimizer_message = f"\nCreated \033[93m{optimizer_name}\033[0m:"
    for key, value in optimizer_params.items():
        optimizer_message += f"\n\t\033[92m{key}\033[0m: {value}"

    logger.debug(optimizer_message)

    return optimizer
