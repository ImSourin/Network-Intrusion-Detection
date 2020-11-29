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
import os

X_train,y_train,_,num_classes = data.get_data()
# X_train = np.reshape(X_train,(X_train.shape[0],9,9,1))
# input_shape = (9,9,1)

model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train,batch_size=500,epochs=10,validation_split=0.3, shuffle=True)
os.chdir('C:/Users/user/Desktop/prj/models')
model.save('model2.h5')
os.chdir('..')