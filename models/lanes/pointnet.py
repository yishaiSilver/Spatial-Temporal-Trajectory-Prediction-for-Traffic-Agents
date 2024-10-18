import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils.logger_config import logger


class TNet(nn.Module):
    """
    Code gotten from fxia22's pointnet.pytorch repository
    """

    def __init__(self, k=2):
        super().__init__()

        self.conv1 = nn.Conv1d(k, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 256, 1)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)  # get the final tranformation matrix

        self.k = k

    def forward(self, x):
        """
        creates a k x k transformation matrix
        """
        batchsize = x.shape[0]

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))


        num_points = x.size(2)
        x = nn.MaxPool1d(kernel_size=num_points)(x).view(batchsize, -1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        np_identity = np.identity(self.k).astype(np.float32)
        identity = Variable(torch.from_numpy(np_identity))
        identity = identity.view(1, self.k * self.k).repeat(batchsize, 1)

        if x.is_cuda:
            identity = identity.cuda()

        x = x + identity
        x = x.view(-1, self.k, self.k)

        return x


class PointNet(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        initialization of pointnet.
        """
        super().__init__(*args, **kwargs)

        # transformation network
        self.tnet1 = TNet(k=2)
        self.tnet2 = TNet(k=32)

        # shared mlp 1
        self.conv1 = nn.Conv1d(2, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1)

        # shared mlp 2
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1)
        self.conv4 = nn.Conv1d(32, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=1) # 256 global feature size

    def forward(self, x):
        """
        feed data through pointnet
        """
        batchsize = x.shape[0]

        # apply first transformation
        first_matrix = self.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), first_matrix).transpose(2, 1)

        # go through first mlp
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))

        # apply second transformation
        second_matrix = self.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), second_matrix).transpose(2, 1)
        
        # go through second mlp
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        # max pool
        num_points = x.size(2)
        x = nn.MaxPool1d(kernel_size=num_points)(x).view(batchsize, -1)

        return x


