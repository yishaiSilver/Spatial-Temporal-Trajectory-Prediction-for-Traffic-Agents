"""
This file contains the implementation of the PointNet backbone used to encode
the lanes.
"""

import torch
import torch.nn as nn


from utils.logger_config import logger

class ResidualBlock(nn.Module):
    """
    A block that implements a residual convolutional connection.
    """

    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_size, 
                 stride, 
                 downsample=False
                 ):
        """
        Initialize Module
        """

        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        if self.downsample:
            self.downsample_fn = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass of ResNet
        """
        identity = x
        if self.downsample:
            identity = self.downsample_fn(x)

        # logger.info(self.downsample)
        # logger.info(f"identity shape: {identity.shape}")

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # logger.info(f"out shape: {out.shape}")

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    A ResNet Image encoder for lanes and neighbors
    """

    def __init__(self, embedding_size=256):
        """
        initialization of pointnet.
        """
        super().__init__()

        self.in_channels = 2

        self.layer1 = self._make_layer(ResidualBlock, 4, 3, kernel_size=5, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 8, 3, kernel_size=5, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 32, 2, kernel_size=5, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, embedding_size)

    def _make_layer(self, block, out_channels, blocks, kernel_size=3, stride=1):
        layers = []

        downsample = self.in_channels != out_channels or stride != 1
        layers.append(block(self.in_channels, out_channels, kernel_size, stride, downsample))

        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size, stride, downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        feed data through pointnet
        """

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
