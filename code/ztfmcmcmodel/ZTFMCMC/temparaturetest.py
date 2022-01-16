# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:56:42 2022

@author: dingxu
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\ZTFMCMC\\modelT\\'


modelT = load_model(path+'weights-improvement-13313-22578.7060.hdf5')

magtemprature = np.loadtxt('magtemprature.txt')

datax = magtemprature[0:3648, 0:2]
datay = magtemprature[0:3648, 2]

predictdatay = modelT.predict(datax)

resiual = datay-predictdatay[:,0]

lin45 = np.arange(3500,10000,0.1)
plt.figure(0)
plt.plot(predictdatay, datay, '.')
plt.plot(lin45, lin45)
plt.plot(predictdatay, resiual+2000, '.')
plt.axhline(y=2000,ls=":",c="black")#添加水平直线

plt.xlabel('Predict-T',fontsize=18)
plt.ylabel('Gaia-T',fontsize=18)

plt.text(5500, 3074, 'resiual=148.16', fontsize=18, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签