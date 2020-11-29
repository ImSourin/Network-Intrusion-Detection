import os
import glob
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils

def get_data():
    os.chdir('C:/Users/user/Desktop/prj' +'/data')
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    col_names = ["Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets","Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s","Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance","FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count","CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size","Fwd Header Length","Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate","Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes","Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min","Label"]
    data = pd.concat([pd.read_csv(f,header=0,names=col_names) for f in all_filenames ])
    data.Label = preprocessing.LabelEncoder().fit_transform(data["Label"])
    X_attr = ["Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets","Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s","Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance","FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count","CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size","Fwd Header Length","Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate","Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes","Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"]
    y_attr = ["Label"]
    X = data[X_attr].as_matrix()
    y = data[y_attr].as_matrix()
    print(len(X_attr))
    for j in range(len(X)):
        # if str(X[j][14]) == 'NaN':
        #     X[j][14] = -1
        #     print(X[j][14])
        # elif str(X[j][14]) == 'Infinity':
        #     X[j][14] = 10000000000000.0
        #     print(X[j][14])
        # elif "E+" in str(X[j][14]):
        #     temp = int(X[j][14][:-4])
        #     p = int(X[j][14][len(X[j][14])-1])
        #     X[j][14] = temp*(10**p)
        #     print(X[j][14])
        X[j][14] = float(X[j][14])
        X[j][15] = float(X[j][15])
        if math.isinf(X[j][14]):
            X[j][14] = 10**15
            # print("inf")
        elif math.isnan(X[j][14]):
            X[j][14] = -1
            # print("nan")
        if math.isinf(X[j][15]):
            X[j][15] = 10**15
        elif math.isnan(X[j][15]):
            X[j][15] = -1
        y[j] = float(y[j])
    # for i in range(len(X[0])):
    #     maxi = -1
    #     for j in range(len(X)):
    #         maxi = max(maxi,X[j][i])
    #     print(maxi)
    X = np.array(X)
    # print(X[0])
    X_input = np.zeros((X.shape[0],X.shape[1]+3))
    X_input[:,:-3] = X
    # print(X_input[0])
    # y = np.array(y,dtype=int)
    # X1 = X
    # X1 = (X1 - X1.mean())/(X1.std()+1e-8)
    X = np.reshape(X_input,(X.shape[0],9,9))
    for i in range(len(X)):
        X[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-8)
    print(X.shape)
    # print(X[0])
    nc = max(y)+1
    # print(nc[0])
    y1 = np_utils.to_categorical(y,nc[0])
    os.chdir('..')
    return (X,y1,y,nc[0])

if __name__=="__main__":
    get_data()