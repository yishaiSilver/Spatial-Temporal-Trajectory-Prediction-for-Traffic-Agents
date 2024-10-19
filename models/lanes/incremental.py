"""
Simple module to slide the 
"""
import torch
import torch.nn as nn

class incremental(nn.Module):
    """
    Simply moves the lanes as the agent moves through space.
    """
    def __init__(self):
        """
        Initializes module
        """
        super().__init__(self)

    def forward(self, x, lanes):
        """
        Forward function of module
        """
        lanes = lanes - x
        lanes = lanes.view(lanes.size(0), -1).unsqueeze(1)
        pass
