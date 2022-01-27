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

predictdatay = modelT.predict(colorindex)

plt.figure(0)
plt.plot(colorindex, T, '.')

#f = c/(ci+b)
def func(x, a, b):
    return a/(x+b)


popt, pcov = curve_fit(func, colorindex, T)
print(popt)
a = popt[0] 
b = popt[1]

yvals = func(colorindex,a,b) #拟合y值
print('popt:', popt)
print('系数a:', a)
print('系数b:', b)
plt.plot(colorindex, predictdatay, '.', c='r')