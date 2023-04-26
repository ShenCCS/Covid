# Data shape = (107964,29409)
## 資料太大無法上傳，程式碼中的路徑都已經寫好，所有的資料都在黑色硬碟中的 \panshi\Run\code_sequence_data
## Code : 01_TSNE_process.py

* Loading data : covid data 107694 20230411.csv

* Program Output : TSNE pic

Note: (檔案在黑色硬碟的 panshi\Run\code\DatasetCGRD\20221004-covid-data\covid data 107694 20230411.csv)
1. Data route : ./DatasetCGRD/20221004-covid-data/covid data 107694 20230411.csv
2. image_transformer.py 要跟程式放在同一個資料夾底下

## Code : 02_LSTM_process.py

* Loading data : ./np_image_totalunit/BA-107694-tsne-binary-perplexity=50-pixel=100/total_seq_array.npy

* Program Output : LSTM model performance

Note: (檔案在黑色硬碟的 panshi\Run\code\np_image_totalunit\BA-107694-tsne-binary-perplexity=50-pixel=100\total_seq_array.npy)
1. Data route : ./np_image_totalunit/BA-107694-tsne-binary-perplexity=50-pixel=100/total_seq_array.npy
2. image_transformer.py 要跟程式放在同一個資料夾底下
