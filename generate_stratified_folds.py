import os
import numpy as np
import csv
import cv2
import pandas as pd

img_list = os.listdir('/scratch/shreyu/ptx-textseg-dataset/candid_ptx_dataset/dicom_files')
neg, small, medium, large = [], [], [], []

for img_path in img_list:
    mask = cv2.imread(os.path.join('/scratch/shreyu/ptx-textseg-dataset/candid_ptx_dataset/','masks',img_path+'.jpg'),0)
    mask = cv2.resize(mask, (224, 224))
    mask = np.array(mask/255, dtype='uint8')
    sum_val = np.sum(mask)

    if(sum_val == 0):
        neg.append(img_path)
    elif(sum_val <= 375):
        small.append(img_path)
    elif(sum_val > 375 and sum_val <= 1250):
        medium.append(img_path)
    elif(sum_val > 1250):
        large.append(img_path)

neg = np.array_split(neg, 5)
small = np.array_split(small, 5)
medium = np.array_split(medium, 5)
large = np.array_split(large, 5)

train, val, test = [], [], []
for i in range(5):
    tmp = np.concatenate((neg[i],small[i],medium[i],large[i])) 
    np.random.shuffle(tmp)
    test.append(tmp)
    tmp = np.concatenate((neg[(i+1)%5],small[(i+1)%5],medium[(i+1)%5],large[(i+1)%5]))
    np.random.shuffle(tmp)
    val.append(tmp)
    tmp = np.concatenate((neg[(i+2)%5],small[(i+2)%5],medium[(i+2)%5],large[(i+2)%5],
                                                  neg[(i+3)%5],small[(i+3)%5],medium[(i+3)%5],large[(i+3)%5],
                                                  neg[(i+4)%5],small[(i+4)%5],medium[(i+4)%5],large[(i+4)%5]))
    np.random.shuffle(tmp)
    train.append(tmp)

#print(len(test), len(train), len(val))
folds = {}
max_length = 0
for i in range(5):
    folds[f"fold_{i}_train"] = list(train[i])
    folds[f"fold_{i}_val"] = list(val[i])
    folds[f"fold_{i}_test"] = list(test[i])
    max_length = max(max_length, train[i].shape[0], val[i].shape[0], test[i].shape[0])

for key in folds.keys():
    folds[key] += (max_length - len(folds[key])) * [-1]

df_folds = pd.DataFrame(folds)
df_folds.head()

df_folds.to_csv("folds_updated.csv")
