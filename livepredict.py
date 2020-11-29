from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat
import pickle
import argparse
import keras
import numpy as np
import tensorflow
import readlive
import re
import base64
import os
import cv2
from keras.models import load_model

def predict():
    os.chdir('C:/Users/user/Desktop/prj/models')
    model = load_model('model.h5')
    os.chdir('..')
    input = readlive.get_data()
    # input = [80,4421382,4,0,24,0,6,6,6,0,0,0,0,0,5.42816703,0.904694505,1473794,2552042.631,4420639,340,4421382,1473794,2552042.631,4420639,340,0,0,0,0,0,0,0,0,0,80,0,0.904694505,0,6,6,6,0,0,0,0,0,0,1,0,0,0,0,7.5,6,0,80,0,0,0,0,0,0,4,24,0,0,256,-1,3,20,0,0,0,0,0,0,0,0,0,0,0]
    # input = np.array(input)
    # input = np.reshape(input,(1,9,9))
    # input = (input - input.mean())/(input.std()+1e-8)
    for i in range(len(input)):
        ip = np.reshape(input[i],(1,9,9,1))
        op = model.predict(ip)
        op = np.reshape(op,len(op[0]))
        if np.where(op == np.amax(op))[0][0] != 0:
            print("Attack Detected!!")
        print(np.where(op == np.amax(op))[0][0])

if __name__=="__main__":
    predict()