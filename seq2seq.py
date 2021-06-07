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

def get_inp(pin, vin, agent_id, track_id, car_mask):
    pin = np.array(pin)
    car_mask = np.count_nonzero(np.array(car_mask))
    num_tracked = 4

    din = conv_pos_to_disp_for_collate(pin)
    
    inp = np.zeros((num_tracked * 2, 19, 2))
    
    agent_index = np.where(track_id == agent_id)[0][0]
    
    dx = pin[:, 18, 0] - pin[agent_index, 18, 0]
    dy = pin[:, 18, 1] - pin[agent_index, 18, 1]
    ds = (dx**2 + dy**2) ** 0.5
    
    closest = np.argsort(ds)
    
    rin = pin[:, :, :] - pin[agent_index, :, :]
#             rout = pout[i, :, :, :] - pout[i, agent_index, :, :]
    
    inp[0] = din[agent_index]
    inp[1] = vin[agent_index]
    inp_index = 1
    for j in range(1, 4):
#                 inp[i, j] = din[i, agent_index]
        if closest[j] >= car_mask:
            break
        inp[inp_index] = rin[closest[j]]
        inp_index += 1
        inp[inp_index] = vin[closest[j]]
        inp_index += 1


    inp = inp.reshape((19, 2*2*num_tracked))

    return inp

          
def get_outp(pin, pout, agent_id, track_id):
    pin = np.array(pin)
    pout = np.array(pout)
    
    last_known = pin[:, 18, :]
    
    dout = conv_pos_to_disp_for_collate(pout, last_known, True)
    
    agent_index = np.where(track_id == agent_id)[0][0]
    
    outp = dout[agent_index]
    
    outp = outp.reshape((60*1))

    return outp
                

def my_collate(batch):
    """ collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature] """
    # city = [scene['city'] for scene in batch]
    # scene_idx = [scene['scene_idx'] for scene in batch]
    # agent_id = [scene['agent_id'] for scene in batch]
    # car_mask = [scene['car_mask'] for scene in batch]
    # track_id = [scene['track_id'] for scene in batch]
    # pin = [scene['p_in'] for scene in batch]
    # vin = [scene['v_in'] for scene in batch]
    # pout = [scene['p_out'] for scene in batch]
    # vout = [scene['v_out'] for scene in batch]
    # lane = [scene['lane'] for scene in batch]
    # lane_norm = [scene['lane_norm'] for scene in batch]
    # inp = [get_inp(scene['p_in'], scene['agent_id'], scene['track_id'], scene['car_mask']) for scene in batch]
    inp = [get_inp(scene['p_in'], scene['v_in'], scene['agent_id'], scene['track_id'], scene['car_mask']) for scene in batch]
    outp = [get_outp(scene['p_in'], scene['p_out'], scene['agent_id'], scene['track_id']) for scene in batch]

    # get_inp_outp(pin, pout, agent_id, track_id, car_mask)

    # return [city, scene_idx, agent_id, car_mask, track_id, pin, vin, pout, vout, lane, lane_norm]
    return [inp, outp]


def my_collate_for_csv(batch):
    """ collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature] """
    # city = [scene['city'] for scene in batch]
    # scene_idx = [scene['scene_idx'] for scene in batch]
    # agent_id = [scene['agent_id'] for scene in batch]
    # car_mask = [scene['car_mask'] for scene in batch]
    # track_id = [scene['track_id'] for scene in batch]
    # pin = [scene['p_in'] for scene in batch]
    # vin = [scene['v_in'] for scene in batch]
    # lane = [scene['lane'] for scene in batch]
    # lane_norm = [scene['lane_norm'] for scene in batch]
    
    
    # return [city, scene_idx, agent_id, car_mask, track_id, pin, vin, lane, lane_norm]

    
    inp = [get_inp(scene['p_in'], scene['agent_id'], scene['track_id'], scene['car_mask']) for scene in batch]
    return [inp]

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
    def __init__(self, input_size=16, num_layers=2, hidden_size=256):
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

RMSE = []
#### from tqdm import tqdm_notebook as tqdm
# def train(model, device, train_loader, optimizer, epoch, log_interval=10000):
# #     model = RNN().to(device)
# #     model.load_state_dict(torch.load("MODEL"))
#     model.train()
    
#     num_tracked = model.num_tracked
#     iterator = tqdm(train_loader, total=int(len(train_loader)))
    
#     total = 0
#     count = 0
#     for batch_idx, batch in enumerate(iterator):
#         city, scene_idx, agent_id, car_mask, track_id, pin, vin, pout, vout, lane, lane_norm = batch
#         pin = np.array(pin)
#         pout = np.array(pout)
#         vin = np.array(vin)
#         vout = np.array(vout)
#         car_mask = np.count_nonzero(np.array(car_mask))
        
            
#         last_known = pin[:, :, 18, :]
    
#         din = conv_pos_to_disp(pin)
#         dout = conv_pos_to_disp(pout, last_known, True)
         
#         inp = np.zeros((len(agent_id), num_tracked, 19, 2))
#         outp = np.zeros((len(agent_id), 30, 2))
#         for i in range(len(agent_id)):
#             agent_index = np.where(track_id[i] == agent_id[i])[0][0]
            
#             dx = pin[i, :, 18, 0] - pin[i, agent_index, 18, 0]
#             dy = pin[i, :, 18, 1] - pin[i, agent_index, 18, 1]
#             ds = (dx**2 + dy**2) ** 0.5
            
#             closest = np.argsort(ds)
            
#             rin = pin[i, :, :, :] - pin[i, agent_index, :, :]
# #             rout = pout[i, :, :, :] - pout[i, agent_index, :, :]
            
#             inp[i, 0] = din[i, agent_index]
#             outp[i] = dout[i, agent_index]
#             for j in range(1, num_tracked):
# #                 inp[i, j] = din[i, agent_index]
#                 if closest[j] >= car_mask:
#                     break
#                 inp[i, j] = rin[closest[j]]
                
# #                 outp[i, j] = rout[closest[j]]
        
#         inp = inp.reshape((len(agent_id), 19, 2*num_tracked))
#         outp = outp.reshape((len(agent_id), 60*1))
    
#         data = torch.from_numpy(inp)
#         target = torch.from_numpy(outp)
        
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data, 30)

#         target = target.float()
        
#         loss = nn.MSELoss()(output, target)
        
#         eps = 1e-6
#         rmse = torch.sqrt(loss + eps)
#         RMSE.append(rmse.item())
        
#         loss.backward()
        
#         nn.utils.clip_grad_norm_(model.parameters(), 3)
        
#         optimizer.step()
    
#         total += loss.item()
#         count += 1
#         iterator.set_postfix_str("loss={}, avg.={}".format(loss.item(), total/count))

def train(model, device, train_loader, optimizer, epoch, log_interval=10000):
#     model = RNN().to(device)
#     model.load_state_dict(torch.load("MODEL"))
    model.train()
    
    num_tracked = model.num_tracked
    iterator = tqdm(train_loader, total=int(len(train_loader)))
    
    total = 0
    count = 0
    for batch_idx, batch in enumerate(iterator):
        inp, outp = batch
        inp = np.array(inp)
        outp = np.array(outp)

        data = torch.from_numpy(inp)
        target = torch.from_numpy(outp)
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, 30)

        target = target.float()
        
        loss = nn.MSELoss()(output, target)
        
        eps = 1e-6
        rmse = torch.sqrt(loss + eps)
        RMSE.append(rmse.item())
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        
        optimizer.step()
    
        total += loss.item()
        count += 1
        iterator.set_postfix_str("loss={}, avg.={}".format(loss.item(), total/count))


def test(model, device, test_loader):
#     model = RNN().to(device)
#     model.load_state_dict(torch.load("MODEL"))
    model.eval()
    test_loss = 0
    total_dist = 0
    num_tested = 0
    with torch.no_grad():
        iterator = tqdm(test_loader, total=int(100))
        for batch_idx, batch in enumerate(iterator):
            
            if num_tested >= 100:
                break;
            
            city, scene_idx, agent_id, car_mask, track_id, pin, vin, pout, vout, lane, lane_norm = batch
            pin = np.array(pin)
            pout = np.array(pout)
            vin = np.array(vin)
            vout = np.array(vout)
            car_mask = np.array(car_mask)
        
        
            last_known = pin[:, :, 18, :]
            pin = conv_pos_to_disp(pin)
            pout = conv_pos_to_disp(pout, last_known, True)

            pin = pin.reshape((len(agent_id), 19, 120))
            pout = pout.reshape((len(agent_id), 3600)) # 30x120

            data = torch.from_numpy(pin)
            target = torch.from_numpy(pout)
            
            data, target = data.to(device), target.to(device)
            output = model(data, 30)
            
            target = target.float()
            
            num_tested += 1
            test_loss += nn.MSELoss()(output, target).item() # sum up batch loss
            
    test_loss /= num_tested
    print("Test loss: {}".format(test_loss))


print("Beginning.")

learning_rate = 0.001
momentum = 0.5
device = "cuda"
model = RNN().to(device) #using cpu here
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
batch_sz = 100
num_epoch = 10

print("# Params: {}".format(sum(p.numel() for p in model.parameters())))


new_path = "./new_train/new_train/"
val_dataset  = ArgoverseDataset(data_path=new_path)
dataset_len = len(val_dataset)
indices = np.arange(0, len(val_dataset))
np.random.shuffle(indices)

# train_loader = DataLoader(val_dataset, batch_size=batch_sz, collate_fn=my_collate, num_workers=1,
#                          sampler=torch.utils.data.SubsetRandomSampler(indices[:int(dataset_len*0.8)]))
# test_loader = DataLoader(val_dataset, batch_size=batch_sz, collate_fn=my_collate, num_workers=1,
#                         sampler=torch.utils.data.SubsetRandomSampler(indices[int(dataset_len*0.8):]))


train_loader = DataLoader(val_dataset,batch_size=batch_sz, shuffle = True, collate_fn=my_collate, num_workers=4)
test_loader = DataLoader(val_dataset,batch_size=batch_sz, shuffle=True, collate_fn=my_collate, num_workers=4)

print(len(train_loader))

# do_both(model, device, train_loader, test_loader, optimizer, epoch)

PATH = "seq2seq-neighbors.pth"
# model.load_state_dict(torch.load(PATH))

# validation_err = 10000
# num_valids_wrong = 0

for epoch in range(1, num_epoch + 1):
    print("EPOCH: {} -----------------------------------".format(epoch))
    train(model, device, train_loader, optimizer, epoch)
#     test(model, device, test_loader)
    torch.save(model.state_dict(), PATH)