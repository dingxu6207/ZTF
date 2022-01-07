#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:05:40 2020

@author: dingxu
"""

#tensorboard --logdir=./log

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation,Dropout
import os
import shutil
import tensorflow as tf
import imageio
from keras.callbacks import EarlyStopping
#from keras.optimizers import adam, rmsprop, adadelta

from random import shuffle
from keras.callbacks import TensorBoard

path = ''
file = 'magtemprature.txt'
data1 = np.loadtxt(path+file)

#data1[:,2] = data1[:,2]/10000
data = np.copy(data1)
print(len(data))



for j in range(100):
    np.random.shuffle(data)


P = 0.8
duan = int(len(data)*P)

dataX = data[:duan,0:2]
#dataX = data[:duan,0:50]
dataY = data[:duan,2]
#dataY[:,0] = dataY[:,0]/90

testX = data[duan:,0:2]
#testX = data[duan:,0:50]
testY = data[duan:,2]
#testY[:,0] = testY[:,0]/90aaa

models = Sequential()
models.add(Dense(30,activation='relu' ,input_dim=2))
Dropout(0.2)
models.add(Dense(10, activation='relu'))
Dropout(0.2)
models.add(Dense(5, activation='relu'))
models.add(Dense(1))

models.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

def displayimg(testY, predictY):
    cancha = testY- predictY[:,0]
    plt.figure(0)
    plt.clf()
    plt.subplot(211)
    plt.plot(testY, predictY,'.')
    plt.plot(testY, testY, '-r')
    
    plt.subplot(212)
    plt.plot(testY, cancha,'.')
    plt.pause(0.1)

    
    

class PredictionCallback(tf.keras.callbacks.Callback):    
    def on_epoch_end(self, epoch, logs={}):
        #print(self.validation_data[0])
        if (epoch%1 == 0): 
            y_pred = self.model.predict(testX)
            displayimg(testY, y_pred)

    
# checkpoint
MODEL = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\ZTFMCMC\\modelT\\'
filepath = MODEL+'weights-improvement-{epoch:05d}-{val_loss:.4f}.hdf5'
# 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,mode='min')
early_stopping = EarlyStopping(patience=200)
callback_lists = [early_stopping, checkpoint, PredictionCallback()]
history = models.fit(dataX, dataY, batch_size=10, epochs=20000, validation_data=(testX, testY),shuffle=True,callbacks=callback_lists)

predictY = models.predict(testX)
predictdatay = models.predict(dataX)

score = models.evaluate(dataX, dataY, batch_size=10)

models.save('phmodsample2.h5')
print(score)



plt.figure(6)
history_dict=history.history
loss_value=history_dict['loss']
val_loss_value=history_dict['val_loss']
epochs=range(1,len(loss_value)+1)
plt.plot(epochs,loss_value,'r',label='Training loss')
plt.plot(epochs,val_loss_value,'b',label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()



