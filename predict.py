from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat
import pickle
import argparse
import keras
import numpy as np
import tensorflow
import data
import re
import base64
import os
import cv2
from keras.models import load_model

os.chdir('C:/Users/user/Desktop/prj/models')
model = load_model('model.h5')
os.chdir('..')
# input,_,output,_ = data.get_data()
#input = [80,4421382,4,0,24,0,6,6,6,0,0,0,0,0,5.42816703,0.904694505,1473794,2552042.631,4420639,340,4421382,1473794,2552042.631,4420639,340,0,0,0,0,0,0,0,0,0,80,0,0.904694505,0,6,6,6,0,0,0,0,0,0,1,0,0,0,0,7.5,6,0,80,0,0,0,0,0,0,4,24,0,0,256,-1,3,20,0,0,0,0,0,0,0,0,0,0,0]
# exp_op = 0
input = [80,1083538,3,6,26,11601,20,0,8.666666667,10.26320288,4380,0,1933.5,1757.789948,10730.58813,8.306123089,135442.25,377725.315,1070206,41,12982,6491,8165.669109,12265,717,1083407,216681.4,477167.0977,1070206,41,0,0,0,0,72,132,2.768707696,5.537415393,0,4380,1162.7,1645.241762,2706820.456,0,0,0,1,0,0,0,0,2,1291.888889,8.666666667,1933.5,72,0,0,0,0,0,0,3,26,6,11601,8192,229,2,20,0,0,0,0,0,0,0,0,0,0,0]
# exp_op = 2
input = np.array(input)
input = np.reshape(input,(1,9,9,1))
input = (input - input.mean())/(input.std()+1e-8)
# id=-1
# cor = 0
# tot = 0
# for i in range(len(output)):
# 	if output[i]!=0:
# 		ip = np.reshape(input[i],(1,9,9,1))
# 		op = model.predict(ip)
# 		op = np.reshape(op,len(op[0]))
# 		ans = np.where(op==np.amax(op))[0][0]
# 		if ans == 0:
# 			cor = cor+1
# 		tot=tot+1
# 		print(ans)
# print(input)
# print(output)
# print(len(op[0]))
op = model.predict(input)
op = np.reshape(op,len(op[0]))
print(np.where(op==np.amax(op))[0])
# print(cor)
# print(tot)