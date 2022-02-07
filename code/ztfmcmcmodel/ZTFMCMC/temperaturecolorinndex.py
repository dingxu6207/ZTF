# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 23:25:47 2022

@author: dingxu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tensorflow.keras.models import load_model

path = ''
file = 'magtemprature.txt'
data = np.loadtxt(path+file)

pathm = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\ZTFMCMC\\modelT\\'
modelT = load_model(pathm+'weights-improvement-13313-22578.7060.hdf5')


colorindex = data[:,0] - data[:,1]
T = data[:,2]

datax = data[0:data.shape[0], 0:2]
predictdatay = modelT(datax)

plt.figure(0)
plt.plot(colorindex, T, '.', c='b', label='origin data')
plt.plot(colorindex, predictdatay, '.', c='r', label='fitting by model')
plt.legend()

plt.xlabel('clorindex',fontsize=18)
plt.ylabel('temperature',fontsize=18)