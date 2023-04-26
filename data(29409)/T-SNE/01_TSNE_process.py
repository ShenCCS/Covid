import pandas as pd
import numpy as np
import pickle as pk, os
import sys
import matplotlib.pyplot as plt

from Bio import SeqIO
from tqdm.notebook import tqdm
from itertools import product
from sklearn.manifold import TSNE
from image_transformer import ImageTransformer, LogScaler
from time import time
from sklearn.model_selection import train_test_split

begin = time()

print("Import Finish")

# loading  onehot covid sequnece csv data.
print("Loading csv data")

source_covid_csv_data = pd.read_csv('./DatasetCGRD/20221004-covid-data/covid data 107694 20230411.csv')

print("Loading Finish")

source_covid_csv_data_col = source_covid_csv_data.columns
source_covid_csv_data_diff = source_covid_csv_data[source_covid_csv_data_col[0]]
source_covid_csv_data_sequnce = source_covid_csv_data[source_covid_csv_data_col[1::]].values



# # only tcga rna unit
def clean(x):
	x = x.upper() 
	
	if x == 'T' or x == 'A' or x == 'G' or x == 'C' or x == '-' or x == 'N':
		return x

	if x == 'U' or x == 'Y':
		return 'T'
	
	if x == 'K' or x == 'S':
		return 'G'

	if x == 'M' or x == 'R' or x == 'W' or x == 'H' or x=='V' or x=='D':
		return 'A'

	if x== 'B':
		return 'C'

dict_search = {}

for idx, i in enumerate('TCAGN-'):
    dict_search[i] = idx

print(dict_search)

num_new_sequences =[]

for k in tqdm(source_covid_csv_data_sequnce):
	temp_store=[]
	for j in k:
		temp_store.append(dict_search[clean(j)])
	num_new_sequences.append(temp_store)
total_sequence_array = np.array(num_new_sequences)

del num_new_sequences, source_covid_csv_data, source_covid_csv_data_sequnce
print(total_sequence_array.shape)

train_sequence_array,test_sequence_array = train_test_split(total_sequence_array,test_size=0.25,random_state=42)

print("Test .shape: {}".format(test_sequence_array.shape))
print("Train .shape: {}".format(train_sequence_array.shape))

ln = LogScaler()
X_train_norm = ln.fit_transform(train_sequence_array)
tsne = TSNE(n_components=2, perplexity=50, metric='cosine',
            random_state=1701, n_jobs=-1)

it = ImageTransformer(feature_extractor=tsne, pixels=100)
X_train_img = it.fit_transform(X_train_norm)

# Save TSNE model
# Scickit Learn tsne model dosen't had save function, only can save model value format

save_model_path = './deepinsight_location_npy/'
save_name = 'BA-107694-tsne-binary-perplexity=50-pixel=100.pkl'
pk.dump(it, open(os.path.join(save_model_path, save_name),"wb"))

print("model saved")
print(it.feature_density_matrix().shape)
print(it.coords().shape)
np.sum(np.array(it.feature_density_matrix())>0)

fig, ax = plt.subplots(1, 4, figsize=(25, 7))

for i in range(0,4):
    ax[i].imshow(X_train_img[i])
    ax[i].title.set_text("Train[{}] ".format(i))
plt.tight_layout()
plt.savefig("TSNE_Training PIC.png")
print("TSNE PIC SAVE")

# transform total sequnce to image
total_sequence_tsne_image_ = it.transform(total_sequence_array)
total_sequence_tsne_image_.shape

# multiclass_nactg multiclass_totalunit
# save_path = './np_image_totalunit/multiclass_nactg_200px/'

save_path = './np_image_totalunit/BA-107694-tsne-binary-perplexity=50-pixel=100/'

if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(os.path.join(save_path,'image_npy')):
    os.mkdir(os.path.join(save_path,'image_npy'))


np.save(f'{save_path}/label.npy', source_covid_csv_data_diff.values)
for idx, image in enumerate(total_sequence_tsne_image_):
    if (idx)<10:
        np.save(f"{save_path}/image_npy/0000{idx}.npy", image)
    elif (idx)<100:
        np.save(f"{save_path}/image_npy/000{idx}.npy", image)
    elif (idx)<1000:
        np.save(f"{save_path}/image_npy/00{idx}.npy", image)
    elif (idx)<10000:
        np.save(f"{save_path}/image_npy/0{idx}.npy", image)
    else:
        np.save(f"{save_path}/image_npy/{idx}.npy", image)

#save total sequnce data to numpy
np.save(f"{save_path}/total_seq_array", total_sequence_array)
end = time()
print(end - begin)