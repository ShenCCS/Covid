import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import sys

from Bio import SeqIO
from tqdm.notebook import tqdm
from tensorflow.keras import models,layers,Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from time import time
begin = time()

print("Loading Data")
source_covid_csv_data = pd.read_csv('./DatasetCGRD/20221004-covid-data/covid data 107694 20230411.csv')


source_covid_csv_data_col = source_covid_csv_data.columns
source_covid_csv_data_diff = source_covid_csv_data[source_covid_csv_data_col[0]]
source_covid_csv_data_sequnce = source_covid_csv_data[source_covid_csv_data_col[1::]].values

print("Finish Loading Data")

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
for k in source_covid_csv_data_sequnce:
	temp_store=[]
	for j in k:
		temp_store.append(dict_search[clean(j)])
	num_new_sequences.append(temp_store)
total_sequence_array = np.array(num_new_sequences)
del num_new_sequences, source_covid_csv_data, source_covid_csv_data_sequnce
print("total_sequence shape",total_sequence_array.shape)

type(total_sequence_array)
sequencedata = pd.DataFrame(total_sequence_array)

from sklearn.model_selection import train_test_split

train_sequence_array,test_sequence_array,train_label,test_label= train_test_split(sequencedata,source_covid_csv_data_diff,test_size=0.1,random_state=42)
train_sequence_array,validation_sequence_array,train_label,val_label= train_test_split(train_sequence_array,train_label,test_size=0.28,random_state=42)

print("Train .shape: {}".format(train_sequence_array.shape))
print("validation .shape: {}".format(validation_sequence_array.shape))
print("Test .shape: {}".format(test_sequence_array.shape))


lb  = LabelEncoder()
train_label = to_categorical(lb.fit_transform(train_label))
val_label = to_categorical(lb.fit_transform(val_label))

train_sequence_array_values = train_sequence_array.values
validation_sequence_array_values = validation_sequence_array.values

train_sequence_array_values = train_sequence_array_values.reshape(train_sequence_array_values.shape[0],29409,1)
validation_sequence_array_values = validation_sequence_array_values.reshape(validation_sequence_array_values.shape[0],29409,1)

model = Sequential()
model.add(LSTM(units=256,input_shape = (29409,1)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=2,activation='sigmoid'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
train_history = model.fit(train_sequence_array_values,train_label,batch_size=64,epochs=5,validation_data=(validation_sequence_array_values,val_label),verbose=2)


history_dict = train_history.history
history_dict.keys()

history_dict = train_history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1,len(acc) + 1)

plt.plot(epochs,loss_values,label = 'Training Loss')
plt.plot(epochs,val_loss,label = 'validation Loss')
plt.title('Training and validatiob loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("LSTM-loss.png")
plt.clf()
print("loss saved")


plt.plot(epochs,acc,label = 'Training acc')
plt.plot(epochs,val_acc,label = 'validation acc')
plt.title('Training and validatiob acc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("LSTM-acc.png")
print("ACC saved")

