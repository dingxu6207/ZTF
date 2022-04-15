# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:59:30 2021

@author: dingxu
"""

from tensorflow.keras.models import load_model
from scipy import interpolate
#from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing    



path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\model\\'
model = load_model(path+'model10.hdf5')



model.summary()

path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 8265951.txt'



data = np.loadtxt(path+file)
datay = data[:,1]-np.mean(data[:,1])


plt.figure(0)
plt.plot(data[:,0], datay, '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)



sx1 = np.linspace(0,1,100)
func1 = interpolate.UnivariateSpline(data[:,0], datay,s=0.0)#强制通过所有点
sy1 = func1(sx1)
plt.figure(0)
plt.plot(sx1, sy1, '.', c='r')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

T1 = 5983
listsy1 = sy1.tolist()
listsy1.append(T1/5800)

sy1 = np.array(listsy1)
nparraydata = np.reshape(sy1,(1,101))

prenpdata = model(nparraydata)

print(prenpdata)