# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 20:58:49 2022

@author: dingxu
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt

#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\ZTFMCMC\\data\\'
#file = 'savedata01050Tl3.txt'

#model1 = load_model(path+'model1.hdf5')
#model10 = load_model(path+'model10.hdf5')
#l3model1 = load_model(path+'model1l3.hdf5')
#l3model10 = load_model(path+'model10l3.hdf5')

mpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\model\\'
dpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\'
file = 'savedata01050l3.txt'

model10mcmc = load_model(mpath+'model10mc.hdf5')
model10l3mcmc = load_model(mpath+'model10l3mc.hdf5')


data = np.loadtxt(dpath+file)
np.random.shuffle(data)
data = data[0:60000,:]
#np.savetxt('savedata01050TN.txt', data)

hang,lie = data.shape

datax = data[:,0:101]
for i in range(0, hang):
    datax[i,0:100] = -2.5*np.log10(datax[i,0:100])
    datax[i,0:100] = datax[i,0:100] - np.mean(datax[i,0:100])

data[:,102] = data[:,102]/90
datay = data[:,[100, 102, 103, 104, 105, 106]]

predict1 = model10l3mcmc(datay)

plt.figure(0)
ax = plt.gca()
ax.plot(datax[0,0:100])
ax.plot(predict1[0],'.')
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

canchajuzhen = datax[:,0:100] - predict1

temp = []
for i in range(0, hang):
    residata = np.std(canchajuzhen[i,:])
    residata = np.round(residata, 5)
    temp.append(residata)

nptemp = np.array(temp)
plt.figure(1)
plt.hist(temp,bins=2000)
plt.xlim(-0.002,0.004)
plt.xlabel('residual',fontsize=18)
plt.ylabel('frequency',fontsize=18)
