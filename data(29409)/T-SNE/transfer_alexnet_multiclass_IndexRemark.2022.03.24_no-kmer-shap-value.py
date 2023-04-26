from feedback import *
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from torchvision.models import alexnet
from torch.nn import Module
from torch import nn
import shap
import numpy
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import glob
from PIL import Image
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import random
from sklearn.metrics import accuracy_score,auc,roc_curve


save_weight_path ='./models/weights_Multiclass_Covid19(Non-kmer3)_IndexRemark.2022.03.24[NATCG]/'
# save_weight_path = './models/weights_Multiclass_Covid19(Non-kmer3)_IndexRemark.2022.03.24[NACGTRYKMSWBDHV]/'

weights_name = "weights_Multiclass_Covid19(Non-kmer3)[NACGT].2023.04.20.pt"
# weights_name = "weights_Multiclass_Covid19(Non-kmer3)[NACGTRYKMSWBDHV].2022.03.24.pt"

path2weights = os.path.join(save_weight_path,weights_name)

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
        singel_image_ = np.load(self.ids[idx]).astype(np.float32)
        seed = np.random.randint(1e9)       
        random.seed(seed)
        np.random.seed(seed)
        singel_image_ = self.transform(singel_image_)
        label = int(self.labels[idx])

        return singel_image_, label


transformer = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            ])     

train_ds = TransferDataset(ids= train_ids, labels= train_labels, transform= transformer)
test_ds = TransferDataset(ids= test_ids, labels= test_labels, transform= transformer)
print(len(train_ds), len(test_ds))

imgs, label = train_ds[10]
batch_size = 32
train_dl = DataLoader(train_ds, batch_size= batch_size, 
                        shuffle=True)
test_dl = DataLoader(test_ds, batch_size= 2*batch_size, 
                        shuffle=False)  

# eval
models = alexnet(pretrained=False, num_classes=max(label_)+1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(path2weights, map_location=torch.device('cuda:0'))
# checkpoint = torch.load(path2weights, map_location=device)

# checkpoint = torch.load('./models/weights_Multiclass_Covid19(Non-kmer3)_IndexRemark.2022.03.24[NACGTRYKMSWBDHV]/weights_Multiclass_Covid19(Non-kmer3)[NACGTRYKMSWBDHV].2022.03.24.pt', map_location=torch.device('cpu'))
models.load_state_dict(checkpoint['model_state_dict'])
models.to(device)
test_dl = DataLoader(test_ds, batch_size= 351, shuffle=False)  
models.eval()

with torch.no_grad():
    for xb, yb in tqdm(test_dl):
        xb=xb.to(device)
        yb=yb.to(device)
    predict_result = np.argmax(models(xb).cpu().numpy(), axis=1)
    ground_truth = yb.cpu().numpy()
    # predict_result = np.argmax(models(xb),dim=1).cpu().numpy()


print('ACC:',accuracy_score(ground_truth, predict_result))

fpr,tpr,thresholds = roc_curve(ground_truth,predict_result,pos_label = 1)
print('AUC:',auc(fpr,tpr))


for xb,yb in train_dl:
    print(yb[0])
    break

for xb,yb in test_dl:
    print(xb.shape, yb.shape)
    break

train_index = np.random.choice(len(train_ds), 100, replace=False)
test_index = np.random.choice(len(test_ds), 5, replace=False)
test_data = [np.load(test_ids[i]).astype(np.float32) for i in test_index]

for image, label in train_dl:
    image = image.to(device)
    label = label.to(device)
    print(image.shape)
    print(type(image))
    break

for image2, label2 in test_dl:
    print(image2[0:5].shape)
    print(type(image2))
    print(type(image2))
    break


test_images = image2[0:5]
batch = next(iter(train_dl))
images,_ = batch


    
e = shap.DeepExplainer(models, image)

shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s,1,-1),1,2)for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(),-1,1),1,2)

# plot the feature attributions
shap.image_plot(shap_numpy, test_numpy)
plt.savefig("shap.png")
print("shap pic save")