# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 20:58:49 2022

@author: dingxu
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt


def quantile(x, q, weights=None): 

    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()
    

mpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\model\\'
dpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\'
file = 'savedata01050l3.txt'

model10mcmc = load_model(mpath+'model10mc.hdf5')
model10l3mcmc = load_model(mpath+'model10l3mc.hdf5')


#data = np.loadtxt(dpath+file)
#np.random.shuffle(data)
#data = data[0:15000,:]
#np.savetxt('testdata01050l3.txt', data)
data = np.loadtxt('testdata01050l3.txt')
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
plt.xlabel('phase', fontsize=18)
plt.ylabel('mag', fontsize=18)
print(data[0,100:108])
print(np.std(data[0,0:100]-np.array(predict1[0])))

canchajuzhen = datax[:,0:100] - predict1

temp = []
for i in range(0, hang):
    residata = np.std(canchajuzhen[i,:])
    residata = np.round(residata, 5)
    temp.append(residata)

nptemp = np.array(temp)
plt.figure(1)
plt.hist(temp,bins=500)
plt.xlim(-0.001,0.002)
plt.xlabel('mag',fontsize=18)
plt.ylabel('number',fontsize=18)
plt.savefig('resiualmagl3.png')

q_16, q_50, q_84 = quantile(nptemp, [0.16, 0.5, 0.84])  
print(q_16, q_50, q_84)
plt.axvline(x = q_16, ls = "-", c = "green", linewidth=1)
plt.axvline(x = q_50, ls = "-", c = "green", linewidth=1)
plt.axvline(x = q_84, ls = "-", c = "green", linewidth=1)
plt.title(r'$\sigma=0.000199^{+0.000339}_{-0.000130}$')