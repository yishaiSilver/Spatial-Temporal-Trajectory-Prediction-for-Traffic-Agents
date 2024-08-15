import argparse
import collections
import torch
import torch.nn as nn
import numpy as np
import data_loader.data_loaders as data
# import model.loss as module_loss
# import model.metric as module_metric

import model.model as model_arch # TODO: don't like name

# from trainer import Trainer
# from utils import prepare_device
import yaml
import tqdm

def train_epoch(model, optimizer, loss_fn, data_loader):
    model.train(True)

    losses = []
    iterator = tqdm.tqdm(data_loader, total=int(len(data_loader)))

    print(len(iterator))
    for data in iterator:
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())
        if len(losses) > 10: # TODO magic number, ew
            losses.pop(0)

        iterator.set_postfix_str(f"avg. loss={sum(losses)/len(losses)}") # TODO easy optimize

def validate_epoch(model, loss_fn, data_loader):
    model.eval()

    val_loss = 0.0
    iterator = tqdm.tqdm(data_loader, total=int(len(data_loader)))
    with torch.no_grad():
        for data in iterator:
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

    return val_loss


def main(config):
    # get the model
    # model_config = config['model']
    model = model_arch.RNN() # TODO: magic line (sort of)

    # get the optimizer
    # optimizer_config = config['optimizer']
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # TODO: magic line

    # get the loss
    # loss_config = config['loss']
    loss_fn = nn.MSELoss() # TODO: magic line

    # get the data
    data_config = config['data']
    train_loader = data.create_data_loader(data_config, train=True)
    val_loader = data.create_data_loader(data_config, train=False)

    # get the number of epochs:
    # num_epochs = config['num_epochs']
    num_epochs = 3

    best_val_loss=np.inf
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch}:")

        train_epoch(model, optimizer, loss_fn, train_loader)
        validation_loss = validate_epoch(model, loss_fn, val_loader)

        # save the model if it's the best
        if validation_loss < best_val_loss:
            print(f"Best Validation loss: {validation_loss}")
            best_val_loss = validation_loss
            model_path = 'model'
            torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Traffic Prediction')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)')
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    main(config)