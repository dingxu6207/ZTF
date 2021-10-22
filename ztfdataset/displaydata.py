# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 21:34:00 2021

@author: dingxu
"""
import numpy as np
import matplotlib.pyplot as plt

path = 'I:\\ZTFDATA\\YUANDATA\\RR\\'
file = '36.txt'
data = np.loadtxt(path+file)

phase = data[:,0]
mag = data[:,1]

plt.figure(2)
plt.plot(phase, mag, '.')
ax1 = plt.gca()
ax1.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax1.invert_yaxis() #y轴反向