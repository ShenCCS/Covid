# Data shape = (107964,29903)
## 資料太大無法上傳，程式碼中的路徑都已經寫好，所有的資料都在黑色硬碟中的 \panshi\Run\code_sequence_data

## First Step (T-SNE)
* Code : deepinsight_generate_image.py

* Loading data(1) : covid ba23 and ba237 lineage data ver1 20230317.csv 
* Loading data(2) : sequence data ba23 and ba237 ver1 20230317.fasta

* Program Output : T-SNE Trainning pictures & N picture & Y picture & diff(N & Y) picture

Note: 
1. Data route : ./seq_data/
2. Output pictures route: same as code
3. Output ndarray route : ./np_image_totalunit/multiclass_nactg/image_npy/

> deepinsight_generate_image.py

## Second Step (AlexNet)
* Code : transfer_alexnet_multiclass_IndexRemark.2022.03.24_no-kmer_recode.py

* Loading data : ndarray from deepinsight_generate_image.py

* Program Output : model performance

Note: 
1. Data route : ./np_image_totalunit/multiclass_nactg/
2. Output model performance route: same as code
3. Line 111 (setting epoch)
4. Line 251 263 (setting model performance file name)

> transfer_alexnet_multiclass_IndexRemark.2022.03.24_no-kmer_recode.py
> 
## Third Step (Shap & AUC & ACC)
* Code : transfer_alexnet_multiclass_IndexRemark.2022.03.24_no-kmer_recode.py

* Program Output : shap & AUC & ACC

Note: 
1. save weight path : ./models/weights_Multiclass_Covid19(Non-kmer3)_IndexRemark.2022.03.24[NATCG]/
2. weight name : "weights_Multiclass_Covid19(Non-kmer3)[NACGT].2023.04.20.pt"
3. Shap Error

> transfer_alexnet_multiclass_IndexRemark.2022.03.24_no-kmer_recode.py
