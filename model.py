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
X_train = np.reshape(X_train,(X_train.shape[0],9,9,1))
input_shape = (9,9,1)
model = Sequential()
model.add(Convolution2D(20,
                        (2,2),
                        padding='valid',
                        input_shape=input_shape,
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(10,
                        (3,3),
                        activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
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
model.save('model.h5')
os.chdir('..')