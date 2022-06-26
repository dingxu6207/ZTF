# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 22:21:18 2022

@author: dingxu
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
from scipy.spatial import cKDTree
import time

mpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\model\\'
dpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\'
#file = 'savedata01050l3.txt'

model10mcmc = load_model(mpath+'model10mcnew.hdf5')
model10l3mcmc = load_model(mpath+'model10l3mc.hdf5')

#data = np.loadtxt(dpath+file)
#np.random.shuffle(data)
#hang,lie = data.shape
#
#for i in range(0, hang):
#    data[i,0:100] = -2.5*np.log10(data[i,0:100])
#    data[i,0:100] = data[i,0:100] - np.mean(data[i,0:100])
#
#np.savetxt('magparametersl3.txt', data)


datanol3 = np.loadtxt('magparametersnol3.txt')
datal3 = np.loadtxt('magparametersl3.txt')

targetpath = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
fileone = 'KIC 11618883.txt'
targetdata = np.loadtxt(targetpath+fileone)

phrase = targetdata[:,0]
datay = targetdata[:,1]-np.mean(targetdata[:,1])
x = np.linspace(0,1,100) #x轴
noisy = np.interp(x,phrase,datay) #y轴

t1=time.time()  
kdt = cKDTree(datanol3[:,0:100])
distnol3, indicesnol3 = kdt.query(noisy)
print('time=',time.time()-t1) 

t1=time.time()  
kdt = cKDTree(datal3[:,0:100])
distl3, indicesl3 = kdt.query(noisy)
print('time=',time.time()-t1) 

plt.figure(1)
plt.plot(x, noisy, '.')
plt.plot(x, datanol3[indicesnol3,0:100], '.')
plt.plot(x, datal3[indicesl3,0:100], '.')

print(datanol3[indicesnol3,100:])
print(datal3[indicesl3,100:])
