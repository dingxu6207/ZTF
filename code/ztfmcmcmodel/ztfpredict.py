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
#model = load_model('incl.hdf5')
model = load_model('model1.hdf5')
#model = load_model('alldown.hdf5')
#model = load_model('all.hdf5')
#model = load_model('q.hdf5')
#model = load_model('accall.hdf5')


model.summary()

path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 7766185.txt'

temperature = 5910/5850

data = np.loadtxt(path+file)
datay = data[:,1]-np.mean(data[:,1])


plt.figure(0)
plt.plot(data[:,0], datay, '.')
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)



sx1 = np.linspace(0,1,100)
func1 = interpolate.UnivariateSpline(data[:,0], datay,s=0.0)#强制通过所有点
sy1 = func1(sx1)
plt.figure(0)
plt.plot(sx1, sy1, '.', c='r')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

listsys = sy1.tolist()
listsys.append(temperature)
npsy1 = np.array(listsys)

nparraydata = np.reshape(npsy1,(1,101))

prenpdata = model.predict(nparraydata)

print(prenpdata)