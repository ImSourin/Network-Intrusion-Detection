import os
import glob
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils

def get_data():
    os.chdir('C:/Users/user/Desktop/prj' +'/livedata')
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    col_names = ["Flow ID","Src IP","Src Port","Dst IP","Dst Port","Protocol","Timestamp","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Byts/s","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Len","Bwd Header Len","Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt","URG Flag Cnt","CWE Flag Count","ECE Flag Cnt","Down/Up Ratio","Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg","Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Init Fwd Win Byts","Init Bwd Win Byts","Fwd Act Data Pkts","Fwd Seg Size Min","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min","Label"]
    # print(len(col_names))
    data = pd.concat([pd.read_csv(f,header=0,names=col_names) for f in all_filenames ])
    # data.Label = preprocessing.LabelEncoder().fit_transform(data["Label"])
    X_attr = ["Dst Port","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Byts/s","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Len","Bwd Header Len","Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt","URG Flag Cnt","CWE Flag Count","ECE Flag Cnt","Down/Up Ratio","Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg","Fwd Header Len","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg","Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Init Fwd Win Byts","Init Bwd Win Byts","Fwd Act Data Pkts","Fwd Seg Size Min","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"]
    X = data[X_attr].as_matrix()
    # print(len(X[0]))
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
    # for i in range(len(X[0])):
    #     maxi = -1
    #     for j in range(len(X)):
    #         maxi = max(maxi,X[j][i])
    #     print(maxi)
    X = np.array(X)
    # print(X[0])
    X_input = np.zeros((X.shape[0],X.shape[1]+3))
    # print(X[0])
    X_input[:,:-3] = X
    # print(X_input[0])
    # y = np.array(y,dtype=int)
    # X1 = X
    # X1 = (X1 - X1.mean())/(X1.std()+1e-8)
    X = np.reshape(X_input,(X.shape[0],9,9))
    for i in range(len(X)):
        X[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-8)
    # print(X.shape)
    # print(X[0])
    # nc = max(y)+1
    # print(nc[0])
    # y1 = np_utils.to_categorical(y,nc[0])
    os.chdir('..')
    return X

if __name__=="__main__":
    get_data()