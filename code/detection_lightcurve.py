# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:18:45 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate
from kneed import KneeLocator
from scipy.spatial import cKDTree

def computedistortion(lightdata):
    datay = lightdata[:,1]
    lendata = len(lightdata)
    len1 = int(lendata/8)
    len2 = int(3*lendata/8)

    len3 = int(5*lendata/8)
    len4 = int(7*lendata/8)
    fendata = np.median(datay[len1:len2])/np.median(datay[len3:len4])
    return fendata
    
path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 3221207.txt'
data = np.loadtxt(path+file)
#fileone = 'mag(0-1).txt'
#data = np.loadtxt(fileone)
phase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])
flux = datay
sx1 = np.linspace(0,1,100)
s = np.diff(datay,2).std()/np.sqrt(6)
num = len(datay)
func1 = interpolate.UnivariateSpline(data[:,0], datay,s=s*s*num)#强制通过所有点
sy1 = func1(sx1)
diffy = np.diff(sy1,1)
# 计算各种参数组合下的拐点
fendata = computedistortion(data)
print('fendata=', fendata)

plt.figure(0)
plt.plot(phase, datay, '.')
plt.plot(sx1, sy1, '.')
plt.plot(sx1[1:], diffy, '.')
plt.plot
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向 
