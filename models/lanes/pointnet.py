"""
This file contains the implementation of the PointNet backbone used to encode
the lanes.
"""

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

    def __init__(self, k):
        super().__init__()

        self.conv1 = nn.Conv1d(k, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 256, 1)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)  # get the final tranformation matrix

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def get_orthogonality_loss(self, matrix):
        """
        Takes in a matrix and calculates the orthogonality loss.
        """

        k = matrix.shape[1]
        identity = torch.eye(k).to(matrix.device)  # Identity matrix
        product = torch.bmm(matrix, matrix.transpose(2, 1))  # T * T^T

        # Compute the difference from the identity matrix
        loss = F.mse_loss(product, identity.expand_as(product))

        return loss

    def forward(self, x):
        """
        creates a k x k transformation matrix
        """
        batchsize = x.shape[0]

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        num_points = x.size(2)
        x = nn.MaxPool1d(kernel_size=num_points)(x).view(batchsize, -1)

        x = F.leaky_relu(self.bn4(self.fc1(x)))
        x = F.leaky_relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        np_identity = np.identity(self.k).astype(np.float32)
        identity = Variable(torch.from_numpy(np_identity))
        identity = identity.view(1, self.k * self.k).repeat(batchsize, 1)

        if x.is_cuda:
            identity = identity.cuda()

        x = x + identity
        x = x.view(-1, self.k, self.k)

        loss = self.get_orthogonality_loss(x)

        return x, loss


class PointNet(nn.Module):
    """
    The PointNet backbone being used to encode the lanes.
    """

    def __init__(self, num_points, input_dims=4, output_dims=64):
        """
        initialization of pointnet.
        """
        super().__init__()

        logger.debug(f"num_points: {num_points}")

        global_feature_dims = 128
        combination_size = (
            global_feature_dims * num_points + global_feature_dims
        )

        # transformation network
        self.tnet1 = TNet(k=input_dims)
        self.tnet2 = TNet(k=32)

        # shared mlp 1
        self.conv1 = nn.Conv1d(input_dims, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1)

        # shared mlp 2
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1)
        self.conv4 = nn.Conv1d(32, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, global_feature_dims, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(global_feature_dims)

        # mlp for combining local and global features
        self.fc1 = nn.Linear(combination_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dims)

        torch.set_float32_matmul_precision("high")

    def forward(self, x):
        """
        feed data through pointnet
        """
        batchsize = x.shape[0]

        # apply first transformation
        tm_1, ortho_loss_1 = self.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), tm_1).transpose(2, 1)

        # go through first mlp
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))

        # apply second transformation
        tm_2, ortho_loss_2 = self.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), tm_2).transpose(2, 1)

        # go through second mlp
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))

        # pull out the local features
        local_feats = x.view(batchsize, -1)

        # max pool to get the global features
        num_points = x.size(2)
        global_feats = nn.MaxPool1d(kernel_size=num_points)(x).view(
            batchsize, -1
        )

        # combine the local and global features
        x = torch.cat((local_feats, global_feats), dim=1)

        # go through the final mlp to get the embeddings
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        # orthogonality losses
        ortho_loss = ortho_loss_1 + ortho_loss_2

        return x, ortho_loss
