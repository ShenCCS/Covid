#epoch 111 251 263

from feedback import *
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.models import alexnet
from torch.nn import Module
from torch import nn
from tqdm import tqdm_notebook
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from time import time

import copy
import datetime
import numpy as np
import torchvision.transforms as transforms
import glob
import torch
import numpy as np
import random


begin = time()
# npy_path = './np_image_totalunit/multiclass_totalunit/'
npy_path = './np_image_totalunit/multiclass_nactg/'

npy_data_list = [os.path.join(npy_path,'image_npy',i ) for i in sorted(os.listdir(os.path.join(npy_path,'image_npy')))]
label_ = np.load(os.path.join(npy_path,'label.npy'))


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

train_indx, test_indx = next(sss.split(npy_data_list, label_))
train_ids = [npy_data_list[ind] for ind in train_indx]
train_labels = [label_[ind] for ind in train_indx]

test_ids = [npy_data_list[ind] for ind in test_indx]
test_labels = [label_[ind] for ind in test_indx]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

train_indx, val_indx = next(sss.split(train_indx, train_labels))
train_ids = [npy_data_list[ind] for ind in train_indx]
train_labels = [label_[ind] for ind in train_indx]

val_ids = [npy_data_list[ind] for ind in val_indx]
val_labels = [label_[ind] for ind in val_indx]


print("Train : ",len(train_ids), len(train_labels))
print("val : ",len(val_ids), len(val_labels))
print("Test : ",len(test_ids), len(test_labels))


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
        # print(self.ids[idx])
        singel_image_ = np.load(self.ids[idx]).astype(np.float32)
        seed = np.random.randint(1e9)       
        random.seed(seed)
        np.random.seed(seed)
        singel_image_ = self.transform(singel_image_)
        # singel_image_ = torch.unsqueeze(self.transform(singel_image_)[0], axis=0)
        label = int(self.labels[idx])
        # print(label)

        return singel_image_, label


transformer = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            ])     

train_ds = TransferDataset(ids= train_ids, labels= train_labels, transform= transformer)
val_ds = TransferDataset(ids= val_ids, labels= val_labels, transform= transformer)
print("Train & val :",len(train_ds), len(val_ds))


model = alexnet(pretrained=False, num_classes=max(label_)+1)
# model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# model



# loss_func = nn.CrossEntropyLoss(reduction="sum", weight=class_weights)
loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1)
os.makedirs("./models", exist_ok=True)
# path2weights = "./models/weights_Multiclass_Covid19(Non-kmer3)_IndexRemark.2022.03.22.pt"
path2weights = "./models/weights_Multiclass_Covid19(Non-kmer3)_IndexRemark.2022.03.24[NATCG]/weights_Multiclass_Covid19(Non-kmer3)[NACGT].2023.04.20.pt"
torch.save(model.state_dict(), path2weights)
params_train={
    "num_epochs": 10, #100 (x)
    "optimizer": opt, 
    "loss_func": loss_func,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": path2weights,
    }



def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

class logs_realtime_reply:
    def __init__(self):
        self.avg_loss=np.inf
        self.running_metic = {"Loss":0, "Accuracy":0}
        self.end_epoch_metric = None
    def metric_stack(self, inputs, targets, loss):

        classes = torch.argmax(inputs, dim=1)
        self.running_metic['Loss'] +=loss
        classes = torch.argmax(inputs, dim=1)
        acc = torch.mean((classes == targets).float())
        # print(acc)
        self.running_metic['Accuracy'] += np.round(acc.cpu().numpy(), 5)*100
    def mini_batch_reply(self, current_step, epoch, iter_len):
        # avg_reply_metric = {"Loss":None, "TP":None, "FP":None, "FN": None, "Spec": None, "Sens": None}
        avg_reply_metric = {"Loss":None, "Accuracy": None}
        for j in avg_reply_metric:
            avg_reply_metric[j] = round(self.running_metic[j]/int(current_step),5)
        
        if current_step ==iter_len:
            self.end_epoch_metric = avg_reply_metric
        return avg_reply_metric
    def epoch_reply(self):
        return self.end_epoch_metric
        
def train(train_loader, model, criterion, optimizer, epoch):
    get_logs_reply = logs_realtime_reply()
    model.train()
    stream = tqdm(train_loader)
   
    for i, (image, label) in enumerate(stream, start=1):
        image=image.to(device)
        label=label.to(device)
        output=model(image)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, 0.5)
        optimizer.step()
        
        get_logs_reply.metric_stack(output, label, loss = round(loss.item(), 5))
        avg_reply_metric = get_logs_reply.mini_batch_reply(i, epoch, len(stream))
        avg_reply_metric['lr'] = optimizer.param_groups[0]['lr']
        stream.set_description(f"Epoch: {epoch}. Train. {str(avg_reply_metric)}")
    return avg_reply_metric['Loss'], avg_reply_metric['Accuracy']
# model validate
def validate(valid_loader, model, criterion, epoch):
    global best_vloss, best_vacc
    get_logs_reply2 = logs_realtime_reply()
    model.eval()
    stream_v = tqdm(valid_loader)
    with torch.no_grad():
        for i, (image, label) in enumerate(stream_v, start=1):
            image=image.to(device)
            label=label.to(device)
            output=model(image)
            loss = criterion(output, label)
            get_logs_reply2.metric_stack(output, label, loss = round(loss.item(), 5))
            avg_reply_metric = get_logs_reply2.mini_batch_reply(i, epoch, len(stream_v))
            stream_v.set_description(f"Epoch: {epoch}. Valid. {str(avg_reply_metric)}")
        avg_reply_metric = get_logs_reply2.epoch_reply()

    for x in avg_reply_metric:
        if x=='Loss' and avg_reply_metric[x]<best_vloss:
            best_vloss = avg_reply_metric[x]
            current_loss = avg_reply_metric['Loss']
            best_ck_name = path2weights
            torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 
                    'loss':  current_loss,}, best_ck_name)
            print('save...', best_ck_name)
    return avg_reply_metric['Loss'], avg_reply_metric['Accuracy']



def  train_valid_process_main(model):
    global best_vloss, best_vacc
    best_vloss = np.inf
    best_vacc = 0.00
    loss_history={
        "train": [],
        "val": [],
    }
    
    metric_history={
        "train": [],
        "val": [],
    }
    # Subject Dataloader Building
    batch_size = 32
    train_dl = DataLoader(train_ds, batch_size= batch_size, 
                            shuffle=True)
    val_dl = DataLoader(val_ds, batch_size= 2*batch_size, 
                            shuffle=False)  

    for epoch in range(1, params_train["num_epochs"] + 1):
        train_loss, train_metric = train(train_dl, model, loss_func, opt, epoch)
        val_loss, val_metric = validate(val_dl, model, loss_func, epoch)
        print("Epoch: ",epoch, f" Train/Valid Loss: {train_loss}|{val_loss}  ", f" Train/Valid Accuracy: {train_metric}|{val_metric}")
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        lr_scheduler.step(val_loss)
    return loss_history, metric_history


loss_history, metric_history  = train_valid_process_main(model)


num_epochs= len(loss_history["train"])
# exper_name = 'RandomAffine'
exper_name = 'LR3e-3'
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_history["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_history["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()

plt.savefig("model performance_loss_10epoch.png")
plt.clf()
print("loss saved")


num_epochs= len(metric_history["train"])
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1), metric_history["train"],label="train")
plt.plot(range(1,num_epochs+1), metric_history["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.savefig("model performance_acc_10epoch.png")
print("acc saved")

end = time()
print(end-begin)