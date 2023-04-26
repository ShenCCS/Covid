from feedback import *
from image_transformer import ImageTransformer, LogScaler
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm_notebook
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from matplotlib import pyplot as plt

import torchvision.transforms as transforms
import glob
import torch
import numpy as np
import random
import pandas as pd
import copy
import torch.nn.functional as F


# sequence numpy array data
seq_npy_data_path = './np_image_totalunit/BA-107694-tsne-binary-perplexity=50-pixel=100/total_seq_array.npy'

print("Loading Data")

load_data = np.load(seq_npy_data_path)

print("Finish Loading Data")
print("Total Sequence data shape: {}".format(load_data.shape))

# Sequence diff label

print("Loading label") 
seq_diff_data_path = './np_image_totalunit/BA-107694-tsne-binary-perplexity=50-pixel=100/label.npy'
load_diff_lab = np.load(seq_diff_data_path, allow_pickle=True)

print("Finish Loading Label")
print("Total Sequence diff lenght: {}".format(len(load_diff_lab)))

ln = LogScaler()
load_data_norm  = ln.fit_transform(load_data)
# binary process
load_diff_lab_norm = [0 if diff=='N' else 1 for  diff in load_diff_lab]

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=42)

train_indx, test_indx = next(sss.split(load_data_norm, load_diff_lab_norm))
train_ids = [load_data_norm[ind] for ind in train_indx]
train_labels = [load_diff_lab_norm[ind] for ind in train_indx]

print(len(train_ids), len(train_labels)) 

test_ids = [load_data_norm[ind] for ind in test_indx]
test_labels = [load_diff_lab_norm[ind] for ind in test_indx]

print(len(test_ids), len(test_labels))

np.random.seed(2020)
random.seed(2020)
torch.manual_seed(2020)

class TransferDataset(Dataset):
    def __init__(self, ids, labels, transform):
        self.transform = transform
        self.ids = ids
        self.labels = labels
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        singel_image_ = self.ids[idx].astype(np.float32)
        singel_image_ = singel_image_.flatten()
        seed = np.random.randint(1e9)       
        random.seed(seed)
        np.random.seed(seed)
        singel_image_ = torch.unsqueeze(torch.FloatTensor(singel_image_), 0)
        # singel_image_ = torch.unsqueeze(self.transform(singel_image_)[0], axis=0)
        label = int(self.labels[idx])

        return singel_image_, label

transformer = transforms.Compose([transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            ])     

train_ds = TransferDataset(ids= train_ids, labels= train_labels, transform= transformer)
test_ds = TransferDataset(ids= test_ids, labels= test_labels, transform= transformer)

print(len(train_ds), len(test_ds))

batch_size = 32
train_dl = DataLoader(train_ds, batch_size= batch_size, 
                        shuffle=True)
test_dl = DataLoader(test_ds, batch_size= 2*batch_size, 
                        shuffle=False)  


for xb,yb in train_dl:
    # print(xb.shape, yb.shape)
    # print(xb[0][0][0][0])
    print(xb[0])
    print(yb[0])
    break

for xb,yb in test_dl:
    print(xb.shape, yb.shape)
    break



class RNN(nn.Module):
    def __init__(self, class_n, Input_Size):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=Input_Size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(128, class_n)
    def forward(self, x):
        r_out, (h_c, h_h) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
model = RNN(2, 29409).to(device)
model.initialize_weights()
# model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
# model



# loss_func = nn.CrossEntropyLoss(reduction="sum", weight=class_weights)
loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(model.parameters(), lr=0.003)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1)
os.makedirs("./models", exist_ok=True)
path2weights = "./models/BA_20000_weights_LSTM_diffclass.pt"
params_train={
    "num_epochs": 50,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": test_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": path2weights,
    }


def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    '''
    for xb, yb in tqdm_notebook(dataset_dl):
    '''
    for xb, yb in (dataset_dl):
      xb=xb.to(device)
      yb=yb.to(device)
      # print(type(xb), type(yb.shape))
      output=model(xb)
      loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
      running_loss+=loss_b
      
      if metric_b is not None:
          running_metric+=metric_b
      if sanity_check is True:
          break
    
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def train_val(model, params):
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    
    loss_history={
        "train": [],
        "val": [],
    }
    
    metric_history={
        "train": [],
        "val": [],
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        

        #print("train loss: %.6f, dev loss: %.6f,  train accuracy: %.2f,valid accuracy: %.2f" %(train_loss,val_loss, 100*train_metric,100*val_metric))
        print("train loss: %.6f, dev loss: %.6f,  train accuracy: %.6f,valid accuracy: %.6f" %(train_loss,val_loss, train_metric,val_metric))
        print("-"*10) 
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

model,loss_hist,metric_hist = train_val(model,params_train)


def plot_loss(loss_hist,metric_hist):
    num_epochs= len(loss_hist["train"])
    # exper_name = 'RandomAffine'
    #exper_name = 'LR3e-3'
    #plt.title(f"Train-Val Loss {exper_name}")
    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), metric_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1), metric_hist["val"],label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig("acc-50.png")
    
    
    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig("loss-50.png")



plot_loss(loss_hist,metric_hist)
print("model performance pic save")

