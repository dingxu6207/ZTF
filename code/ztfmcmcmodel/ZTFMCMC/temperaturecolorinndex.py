# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 23:25:47 2022

@author: dingxu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tensorflow.keras.models import load_model
from scipy.optimize import curve_fit

'''
B−V=a1+a2/T
T = a2/(B-V+a1)

'''

def func(x, a2, a1, c):
    return a2/(x+a1)+c


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
#plt.plot(colorindex, predictdatay, '.', c='r', label='fitting by model')

popt, pcov = curve_fit(func, colorindex, T)
a = popt[0]
b = popt[1]
c = popt[2]
ALLY = a/(colorindex+b) + c
plt.plot(colorindex, ALLY, '.', c='r', label='fitting data')


cancha = ALLY - T 

plt.plot(colorindex, cancha, '.', c='g', label='resiual data')
plt.axhline(y=0,ls=":",c="black")#添加水平直线

plt.text(0.5, 1991, r'$\sigma$'+'='+str(np.round(np.std(cancha),3)), fontsize=18, color = "r")

plt.legend()
plt.xlabel('colorindex',fontsize=18)
plt.ylabel('temperature',fontsize=18)
