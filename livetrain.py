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
from scapy.all import sniff
from scapy.all import wrpcap
from scapy.arch.windows import show_interfaces

def getlive():
    show_interfaces()
    # pkts = sniff(timeout=15,iface="Npcap Loopback Adapter")
    pkts = sniff(timeout=15)
    os.chdir(os.getcwd()+'/livepcap')
    wrpcap('data.pcap',pkts)
    os.chdir('..')
    os.chdir('C:/Users/user/Desktop/prj/cicflowmeter-4/CICFlowMeter-4.0/bin')
    os.system("cfm.bat \"C:/Users/user/Desktop/prj/livepcap\" \"C:/Users/user/Desktop/prj/livedata\"")
    os.chdir('C:/Users/user/Desktop/prj')

def livetrain():
    os.chdir('C:/Users/user/Desktop/prj/models')
    model = load_model('model.h5')
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    os.chdir('..')
    model_input = readlive.get_data()
    opt = int(input("Give type of attack : "))
    output = []
    for _ in range(len(model_input)):
        output.append(opt)
    model_input = np.reshape(model_input,(model_input.shape[0],9,9,1))
    output = np_utils.to_categorical(output,15)
    model.fit(model_input,output,batch_size=500,epochs=10,validation_split=0.3, shuffle=True)
    os.chdir('C:/Users/user/Desktop/prj/models')
    model.save('model.h5')
    os.chdir('..')

if __name__=="__main__":
    getlive()
    livetrain()