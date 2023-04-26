import pandas as pd
import numpy as np

from time import time
from itertools import product
from tqdm.notebook import tqdm
from feedback import *
from image_transformer import ImageTransformer, LogScaler
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from itertools import product


begin = time()

nas_path = "./seq_data/"

print("Loading Data")
lineage_label = pd.read_csv('./seq_data/covid ba23 and ba237 lineage data ver1 20230317.csv')[['outcome_difference']]
lineage_label = np.array(lineage_label.fillna("None"))
print("Finish Loading Data")

label_ = []
new_lineage_label = []


for idx, rna in enumerate(SeqIO.parse('./seq_data/sequence data ba23 and ba237 ver1 20230317.fasta',"fasta")):
    label_.append(lineage_label[idx][0].split(' ')[0])

    new_lineage_label.append(str(rna.seq).replace('-','N'))

print('sample:', len(new_lineage_label))


class_,_ ,_,_= np.unique(label_,return_counts=True,return_index=True,return_inverse=True)

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

for k in new_lineage_label:
	temp_store=[]
	for j in k:
		temp_store.append(dict_search[clean(j)])
	num_new_sequences.append(temp_store)

total_sequence_array = np.array(num_new_sequences)

del num_new_sequences

print(total_sequence_array.shape)


class_dict_ = {}

for idx, i in enumerate(class_):
    class_dict_[i] = idx


print(class_dict_)

multi_label = []

for i in label_:
    multi_label.append(class_dict_[i])

train_sequence_array,test_sequence_array = train_test_split(total_sequence_array,test_size=0.25,random_state=42)

print("Test .shape: {}".format(test_sequence_array.shape))
print("Train .shape: {}".format(train_sequence_array.shape))

ln = LogScaler()
X_train_norm = ln.fit_transform(total_sequence_array)
tsne = TSNE(n_components=2, metric='euclidean',random_state=1701)
            
it =  ImageTransformer(feature_extractor=tsne, pixels=100)
X_train_img = it.fit_transform(X_train_norm)

#Draw N and Y and Diff

print("starting N and Y and Diff pic")

N = []
Y = []

for i in range(0,107963):
    if multi_label[i] == 0:
        N.append(X_train_img[i])
    if multi_label[i] == 1:
        Y.append(X_train_img[i])

print("N :",len(N))
print("Y :",len(Y))

fig, ax = plt.subplots()

ax.imshow(N[0])
ax.title.set_text("N")
plt.tight_layout()
plt.savefig('N.png')

fig, ax = plt.subplots()

ax.imshow(Y[0])
ax.title.set_text("Y")
plt.tight_layout()
plt.savefig('Y.png')


img1 = cv2.imread("./N.png")
img2 = cv2.imread("./Y.png")

print("Nand Y saved")


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


diff = cv2.absdiff(gray1, gray2)
diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]


contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)


cv2.imwrite("diff.png",N)

print("Diff saved")


fdm = it.feature_density_matrix()
fdm[fdm == 0] = np.nan

plt.figure(figsize=(10, 7))

ax = sns.heatmap(fdm, cmap="viridis", linewidths=0.01, linecolor="lightgrey", square=True)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

for _, spine in ax.spines.items():
    spine.set_visible(True)
_ = plt.title("Genes per pixel")
plt.savefig("pic.png")
print("fig save")

fig, ax = plt.subplots(1, 4, figsize=(25, 7))

for i in range(0,4):
    ax[i].imshow(X_train_img[i])
    ax[i].title.set_text("Train[{}] ".format(i))

plt.tight_layout()
plt.savefig("train.png")
print("fig save")

print(it.feature_density_matrix().shape)
print(it.coords().shape)


np.save("deepinsight_location_npy/feature_density_matrix_[NACGT]-multiclass=107964.npy",it.feature_density_matrix())
np.save("deepinsight_location_npy/coords_[NACGT]-multiclass=107964.npy",it.coords())

# multiclass_nactg multiclass_totalunit
np.save('./np_image_totalunit/multiclass_nactg/label.npy',multi_label)
for idx, image in enumerate(X_train_img):
    if (idx)<10:
        np.save(f"./np_image_totalunit/multiclass_nactg/image_npy/000{idx}.npy", image)
    elif (idx)<100:
        np.save(f"./np_image_totalunit/multiclass_nactg/image_npy/00{idx}.npy", image)
    elif (idx)<1000:
        np.save(f"./np_image_totalunit/multiclass_nactg/image_npy/0{idx}.npy", image)
    else:
        np.save(f"./np_image_totalunit/multiclass_nactg/image_npy/{idx}.npy", image)

end = time()
print(end-begin)