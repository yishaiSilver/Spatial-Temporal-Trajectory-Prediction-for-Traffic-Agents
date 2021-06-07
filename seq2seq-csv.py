import torch
from torch.utils.data import Dataset, DataLoader
import os, os.path 
import numpy as np
import pickle
from glob import glob
import numpy as np
import csv
from tqdm import tqdm


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

def conv_pos_to_disp_for_collate(x, last_known=None, use_known = False):
    arr = np.zeros(x.shape)
    if not use_known:
        for j in range(1, arr.shape[2]):
            arr[:, j, :] = x[:, j, :] - x[:, j - 1, :]
    else:
        arr[:, 0, :] = x[:, 0, :] - last_known
        for j in range(1, arr.shape[2]):
            arr[:, j, :] = x[:, j, :] - x[:, j - 1, :]
    return arr

def get_inp(pin, agent_id, track_id, car_mask):
    pin = np.array(pin)
    car_mask = np.count_nonzero(np.array(car_mask))
    num_tracked = 4

    din = conv_pos_to_disp_for_collate(pin)
    
    inp = np.zeros((num_tracked, 19, 2))
    
    agent_index = np.where(track_id == agent_id)[0][0]
    
    dx = pin[:, 18, 0] - pin[agent_index, 18, 0]
    dy = pin[:, 18, 1] - pin[agent_index, 18, 1]
    ds = (dx**2 + dy**2) ** 0.5
    
    closest = np.argsort(ds)
    
    rin = pin[:, :, :] - pin[agent_index, :, :]
#             rout = pout[i, :, :, :] - pout[i, agent_index, :, :]
    
    inp[0] = din[agent_index]
    for j in range(1, num_tracked):
#                 inp[i, j] = din[i, agent_index]
        if closest[j] >= car_mask:
            break
        inp[j] = rin[closest[j]]


    inp = inp.reshape((19, 2*num_tracked))

    return inp

def get_lk(pin, agent_id, track_id):
    pin = np.array(pin)
    
    agent_index = np.where(track_id == agent_id)[0][0]
    
    return pin[agent_index, 18]

def my_collate_for_csv(batch):
    inp = [get_inp(scene['p_in'], scene['agent_id'], scene['track_id'], scene['car_mask']) for scene in batch]
    lk = [get_lk(scene['p_in'], scene['agent_id'], scene['track_id']) for scene in batch]
    scene_idx = [scene['scene_idx'] for scene in batch]
    return [inp, lk, scene_idx]

def conv_pos_to_disp(x, last_known=None, use_known = False):
    arr = np.zeros(x.shape)
    for i in range(arr.shape[0]):
        if not use_known:
            for j in range(1, arr.shape[2]):
                arr[i, :, j, :] = x[i, :, j, :] - x[i, :, j - 1, :]
        else:
            arr[i, :, 0, :] = x[i, :, 0, :] - last_known[i]
            for j in range(1, arr.shape[2]):
                arr[i, :, j, :] = x[i, :, j, :] - x[i, :, j - 1, :]
    return arr

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RNN(nn.Module):
    def __init__(self, input_size=8, num_layers=2, hidden_size=256):
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
        
        self.fc1 = nn.Linear(self.s3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_size)
        
        self.fc4 = nn.Linear(self.s3, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 2)
        
    def forward(self, x, future=0):
        x = x.float()
        outputs = []
        n_samples = x.size(0)
        
        ht = torch.zeros(n_samples, self.s1, dtype=torch.float32).to(device)
        ct = torch.zeros(n_samples, self.s1, dtype=torch.float32).to(device)
        ht2 = torch.zeros(n_samples, self.s2, dtype=torch.float32).to(device)
        ct2 = torch.zeros(n_samples, self.s2, dtype=torch.float32).to(device)
        ht3 = torch.zeros(n_samples, self.s3, dtype=torch.float32).to(device)
        ct3 = torch.zeros(n_samples, self.s3, dtype=torch.float32).to(device)
        
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

def conv_disp_to_pos(x, last_known):
    # print("cdtp: {}".format(x.shape))
    x[0, :] += last_known
    for i in range(1, 30):
        x[i, :] += x[i - 1, :]
    return x

def create_csv_for_vals(model, device, loader):#     model = RNN().to(device)
    with open('csv_submission.csv', 'w') as csv_sub_wrap:
        csv_sub = csv.writer(csv_sub_wrap)
        
        first_row = ["ID",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",  "v8",  "v9",  "v10",  "v11",  "v12",  "v13",  "v14",  "v15",  "v16",  "v17",  "v18",  "v19",  "v20",  "v21",  "v22",  "v23",  "v24",  "v25",  "v26",  "v27",  "v28",  "v29",  "v30",  "v31",  "v32",  "v33",  "v34",  "v35",  "v36",  "v37",  "v38",  "v39",  "v40",  "v41",  "v42",  "v43",  "v44",  "v45",  "v46",  "v47",  "v48",  "v49",  "v50",  "v51",  "v52",  "v53",  "v54",  "v55",  "v56",  "v57",  "v58",  "v59",  "v60"]
        csv_sub.writerow(first_row)
        
        model.eval()
        with torch.no_grad():
            iterator = tqdm(loader, total=int(len(loader)))
            for batch_idx, batch in enumerate(iterator):
                inp, lk, scene_idx = batch
                inp = np.array(inp)

                data = torch.from_numpy(inp)

                data = data.to(device)
                output = model(data, 30)

                output = output.reshape((inp.shape[0], 30, 2)).cpu().numpy()                

                for i in range(inp.shape[0]):
                    pout = output[i]
                    last_known = lk[i]
                    pout = conv_disp_to_pos(pout, last_known)
                    pout = pout.reshape((60))
                    out = []
                    for j in range(61):
                        out.append(1)
                    out[0] = scene_idx[i]
                    out[1:] = pout
                    csv_sub.writerow(out)
#                     csv_sub.writerow([scene_idx[i], pout])
#                     csv_sub.write(scene_idx)
#                     csv_sub.writerow(pout)

print("Beginning.")
device = "cuda"
model = RNN().to(device) #using cpu here
batch_sz = 100

print(sum(p.numel() for p in model.parameters()))


new_path = "./new_val_in/new_val_in/"
val_dataset  = ArgoverseDataset(data_path=new_path)
loader = DataLoader(val_dataset,batch_size=batch_sz, shuffle = True, collate_fn=my_collate_for_csv, num_workers=1)


# do_both(model, device, train_loader, test_loader, optimizer, epoch)
PATH = "seq2seq-neighbors.pth"
model.load_state_dict(torch.load(PATH))


# visualize(model, device, train_loader)

create_csv_for_vals(model, device, loader)