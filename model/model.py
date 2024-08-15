import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class RNN(nn.Module):
    def __init__(self, input_size=8, num_layers=2, hidden_size=256, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tracked = 4
        
        smallest = 128
        
        self.s1 = smallest
        self.s2 = smallest * 2
        self.s3 = smallest * 4
        
        self.lstm1 = nn.LSTMCell(input_size, self.s1)
        self.lstm2 = nn.LSTMCell(self.s1, self.s2)
        self.lstm3 = nn.LSTMCell(self.s2, self.s3)
        
        self.lstm4 = nn.LSTMCell(input_size, self.s1)
        self.lstm5 = nn.LSTMCell(self.s1, self.s2)
        self.lstm6 = nn.LSTMCell(self.s2, self.s3)
        
        self.fc1 = nn.Linear(self.s3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, input_size)
        
        self.fc4 = nn.Linear(self.s3, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 2)
        
        
        
    def forward(self, x, future=0):
        x = x.float()
        outputs = []
        n_samples = x.size(0)
        
        ht = torch.zeros(n_samples, self.s1, dtype=torch.float32).to(self.device)
        ct = torch.zeros(n_samples, self.s1, dtype=torch.float32).to(self.device)
        ht2 = torch.zeros(n_samples, self.s2, dtype=torch.float32).to(self.device)
        ct2 = torch.zeros(n_samples, self.s2, dtype=torch.float32).to(self.device)
        ht3 = torch.zeros(n_samples, self.s3, dtype=torch.float32).to(self.device)
        ct3 = torch.zeros(n_samples, self.s3, dtype=torch.float32).to(self.device)
        
        for input_t in x.split(1, dim=1):
            input_t = input_t.reshape((n_samples, self.input_size))
            ht, ct = self.lstm1(input_t, (ht, ct))
            ht2, ct2 = self.lstm2(ht, (ht2, ct2))
            ht3, ct3 = self.lstm3(ht2, (ht3, ct3))
            
        ret = input_t
        for i in range(future):
            ht, ct = self.lstm4(ret, (ht, ct))
            ht2, ct2 = self.lstm5(ht, (ht2, ct2))
            ht3, ct3 = self.lstm6(ht2, (ht3, ct3))
                                 
            ret = F.relu(self.fc1(ht3))
            ret = F.relu(self.fc2(ret))
            ret = self.fc3(ret)
            
            out = F.relu(self.fc4(ht3))
            out = F.relu(self.fc5(out))
            out = self.fc6(out)
            outputs.append(out)
            
        outputs = torch.cat(outputs, dim=1)
        return outputs