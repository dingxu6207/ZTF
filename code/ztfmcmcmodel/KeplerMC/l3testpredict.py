# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:19:36 2022

@author: dingxu
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt

mpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\model\\'
dpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\'
file = 'savedata01050l3.txt'

#model1 = load_model(mpath+'model1.hdf5')
model10 = load_model(mpath+'model10.hdf5')
#l3model1 = load_model(mpath+'model1l3.hdf5')
l3model10 = load_model(mpath+'model10l3.hdf5')


data = np.loadtxt(dpath+file)
print(len(data))

np.random.shuffle(data)
data = data[0:100000,:]
#np.savetxt('savedata01050TN.txt', data)

hang,lie = data.shape

datax = data[:,0:101]
for i in range(0, hang):
    datax[i,0:100] = -2.5*np.log10(datax[i,0:100])
    datax[i,0:100] = datax[i,0:100] - np.mean(datax[i,0:100])

datay = data[:,102:107]

predict1 = l3model10.predict(datax)

plt.figure(0)
plt.plot(datay[:,0], predict1[:,0], '.', markersize=0.1)
plt.plot(np.arange(50,90,0.1), np.arange(50,90,0.1), c="orange")
plt.xlabel('incl',fontsize=18)
plt.ylabel('predict-incl',fontsize=18)
cancha = datay[:,0]- predict1[:,0]
plt.plot(datay[:,0], cancha+40, '.', c = 'green', markersize=0.1)
plt.axhline(y=40,ls="--",c="orange",linewidth=2)#添加水平直线
plt.text(55, 40+4, r'$\sigma$'+'='+str(np.round(np.std(cancha),3)), fontsize=18, color = "r")
plt.ylim(30,100)


plt.figure(1)
plt.plot(datay[:,1], predict1[:,1]/10, '.', markersize=0.1)
plt.plot(np.arange(0,10,0.01), np.arange(0,10,0.01))
plt.xlabel('q',fontsize=18)
plt.ylabel('predict-q',fontsize=18)
cancha = datay[:,1]- predict1[:,1]/10
plt.plot(datay[:,1], cancha-1, '.', c = 'green', markersize=0.1)
plt.axhline(y=-1,ls="--",c="orange",linewidth=2)#添加水平直线
plt.text(2, -0.39, r'$\sigma$'+'='+str(np.round(np.std(cancha),3)), fontsize=18, color = "r")
plt.ylim(-3,11)

plt.figure(2)
plt.plot(datay[:,2], predict1[:,2]/100, '.', markersize=0.1)
plt.plot(np.arange(0,1,0.01), np.arange(0,1,0.01))
plt.xlabel('f',fontsize=18)
plt.ylabel('predict-f',fontsize=18)
cancha = datay[:,2]- predict1[:,2]/100
plt.plot(datay[:,2], cancha-0.05, '.', c = 'green', markersize=0.1)
plt.axhline(y=-0.05,ls="--",c="orange",linewidth=2)#添加水平直线
plt.text(0.27, 0, r'$\sigma$'+'='+str(np.round(np.std(cancha),3)), fontsize=18, color = "r")
print(np.std(cancha))
plt.ylim(-0.2,1.1)

plt.figure(3)
plt.plot(datay[:,3], predict1[:,3]/100, '.', markersize=0.1)
plt.plot(np.arange(0.8,1.2,0.01), np.arange(0.8,1.2,0.01))
plt.xlabel('T2/T1',fontsize=18)
plt.ylabel('predict-T2/T1',fontsize=18)
cancha = datay[:,3]- predict1[:,3]/100
plt.plot(datay[:,3], cancha+0.7, '.', c = 'green', markersize=0.1)
plt.axhline(y=0.7,ls="--",c="orange",linewidth=2)#添加水平直线
plt.text(0.9, 0.7+0.06, r'$\sigma$'+'='+str(np.round(np.std(cancha),3)), fontsize=18, color = "r")
plt.ylim(0.6,1.3)

plt.figure(4)
plt.plot(datay[:,4], predict1[:,4]/100, '.', markersize=0.1)
plt.plot(np.arange(0.0,1.0,0.01), np.arange(0.0,1.0,0.01))
plt.xlabel('l3',fontsize=18)
plt.ylabel('predict-l3',fontsize=18)
cancha = datay[:,4]- predict1[:,4]/100
plt.plot(datay[:,4], cancha-0.25, '.', c = 'green', markersize=0.1)
plt.axhline(y=-0.25,ls="--",c="orange",linewidth=2)#添加水平直线
plt.text(0.27, -0.1, r'$\sigma$'+'='+str(np.round(np.std(cancha),3)), fontsize=18, color = "r")
plt.ylim(-0.5, 1.2)