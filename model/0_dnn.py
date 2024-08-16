import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class RNN(nn.Module):
    def __init__(
        self, input_size=38, num_layers=2, hidden_size=256, device="cpu"
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tracked = 4

        self.fc1 = nn.Linear(self.s3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, input_size)

        self.fc4 = nn.Linear(self.s3, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 2)

        self.device = device

    def forward(self, x, future=0):
        x = x.float()
        outputs = []
        n_samples = x.size(0)

        ret = F.relu(self.fc1(ht3))
        ret = F.relu(self.fc2(ret))
        ret = self.fc3(ret)

        out = F.relu(self.fc4(ht3))
        out = F.relu(self.fc5(out))
        out = self.fc6(out)
        outputs.append(out)

        print(outputs)

        outputs = torch.cat(outputs, dim=1)
        return outputs
