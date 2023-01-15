import os
import random

import torch
import numpy as np
import math

#data_dir = r"../output_npz_new/Sleep-edf20"  #for sleepedf
#data_dir = r"../output_npz_new2/shhs1"#for shhs1
data_dir = r"../output_npz_new2/shhs2"  #for shhs2

output_dir = r"../output_npz_new7"
files = os.listdir(data_dir)
files = np.array([os.path.join(data_dir, i) for i in files])
files.sort()


#edf20_permutation = np.array([23,0,1,29,30,31,32,18,  14,19,15,16,7,17,6,   27 ,22 ,21 ,38 ,24 ,8 , 3 ,2 ,37 ,28 ,34,9 ,25 ,36 ,20 ,35 ,26 ,33,12 ,5 ,4,10,11 ,13])
#edf20_permutation = np.array([22,23,0,1,29,30,31,32,   18,19,16,17,14,15,6,7,  27,28,10,11,2,3,33,34,24,25,20,21,35,36,8,9,12,13,37,38,4,5,26])
#shhs1_permutation = np.array([3, 1, 34, 2, 25, 39, 11, 14, 41, 38, 17, 40, 0, 15, 23, 6, 32, 9, 35, 7, 33, 8, 26, 13, 31, 19, 20, 12, 37, 4, 24,    10, 27, 16, 18, 22, 28, 21, 36, 5, 30, 29])
shhs2_permutation = np.array([0,51,35,37,23,27,12,19,21,   39,14,53,5,18,40,2,34,32,   48,24,47,15,44,38,8,30,33,29,20,13,1,6,31,25,16,10,4,45,43,28,7,11,36,46,17,9,49,52,26,22,42,3,50,41,])

#files = files[edf20_permutation]
#files = files[shhs1_permutation]
files = files[shhs2_permutation]

#divide data among different subjects
#EDF20
'''
len_test = math.ceil(len(files) * 0.2)
len_valid = int(len(files) * 0.2)
print("len_test",len_test)
print("len_valid",len_valid)
'''

#SHHS1
'''
len_test = int(len(files) * 0.2)
len_valid = math.ceil(len(files) * 0.2)
print("len_test",len_test)
print("len_valid",len_valid)
'''

#SHHS2

len_test = int(len(files) * 0.18)
len_valid = int(len(files) * 0.18)
print("len_test",len_test)
print("len_valid",len_valid)

######## Test files ##########
test_files = files[0:len_test]
print("test files:",test_files)
# load files
X_test = np.load(test_files[0])["x"]
y_test = np.load(test_files[0])["y"]

for np_file in test_files[1:]:
    X_test = np.vstack((X_test, np.load(np_file)["x"]))
    y_test = np.append(y_test, np.load(np_file)["y"])

data_save = dict()
data_save["samples"] = torch.from_numpy(X_test.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_test)
#torch.save(data_save, os.path.join(output_dir, "test_a.pt"))
#torch.save(data_save, os.path.join(output_dir, "test_b.pt"))
torch.save(data_save, os.path.join(output_dir, "test_c.pt"))

######## Validation ##########
validation_files = files[len_test:(len_test + len_valid)]
print("validation_files:",validation_files)
# load files
X_val = np.load(validation_files[0])["x"]
y_val = np.load(validation_files[0])["y"]

for np_file in validation_files[1:]:
    X_val = np.vstack((X_val, np.load(np_file)["x"]))
    y_val = np.append(y_val, np.load(np_file)["y"])

data_save = dict()
data_save["samples"] = torch.from_numpy(X_val.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_val)
#torch.save(data_save, os.path.join(output_dir, "val_a.pt"))
#torch.save(data_save, os.path.join(output_dir, "val_b.pt"))
torch.save(data_save, os.path.join(output_dir, "val_c.pt"))

######## TesT ##########
train_files = files[(len_test+len_valid):]
print("train_files:",train_files)
# load files
X_train = np.load(train_files[0])["x"]
y_train = np.load(train_files[0])["y"]

for np_file in train_files[1:]:
    X_train = np.vstack((X_train, np.load(np_file)["x"]))
    y_train = np.append(y_train, np.load(np_file)["y"])

#shuffle training data
data_index = np.array([i for i in range(len(y_train))])
print("data_index before shuffle", data_index)
random.shuffle(data_index)
X_train = X_train[data_index]
y_train = y_train[data_index]
print("data_index shuffled=", data_index)

data_save = dict()
data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_train)
#torch.save(data_save, os.path.join(output_dir, "train_a.pt"))
#torch.save(data_save, os.path.join(output_dir, "train_b.pt"))
torch.save(data_save, os.path.join(output_dir, "train_c.pt"))

