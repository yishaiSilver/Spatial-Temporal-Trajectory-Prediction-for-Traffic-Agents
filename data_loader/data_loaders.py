import torch
from torch.utils.data import Dataset, DataLoader
import os, os.path 
import numpy 
import pickle
from glob import glob

from .collate import collate_fn

# number of sequences in each dataset
# train:205942  val:3200 test: 36272 
# sequences sampled at 10HZ rate

class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""
    def __init__(self, data_path: str, transform=None):
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform

        self.pkl_list = glob(os.path.join(self.data_path, '*'))
        self.pkl_list.sort()
        
    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):

        pkl_path = self.pkl_list[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        if self.transform:
            data = self.transform(data)

        return data

def create_data_loader(config, train=True):
        #   data_path: str, transforms=None, batch_size=4, shuffle=False, val_split=0.0, num_workers=1
        if train:
            data_path = config['train_path']
        else:
            data_path = config['val_path']

        batch_size = config['batch_size']
        shuffle = config['shuffle']
        num_workers = config['num_workers']

        transforms = None

        transform_fn = lambda x: x if transforms is None else [transform(x) for transform in transforms]
        dataset = ArgoverseDataset(data_path, transform=transform_fn)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
