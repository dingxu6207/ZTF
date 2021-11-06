# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:35:26 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy.pyasl import foldAt
from scipy import interpolate

data = np.loadtxt('lta1')
lendata = len(data)
jd = data[:,0][0:15000]
flux = data[:,1][0:15000]
P = 0.733738


phases = foldAt(jd, P)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
resulflux = flux[sortIndi]
resultmag = -2.5*np.log10(resulflux)
resultmag = resultmag - np.mean(resultmag)
 

listmag = resultmag.tolist()
listmag.extend(listmag)
listphrase = phases.tolist()
listphrase.extend(listphrase+np.max(1)) 
nplistmag = np.array(listmag)
nplistphase = np.array(listphrase)

s = np.diff(nplistmag,2).std()/np.sqrt(6)
num = len(nplistmag)
#lvalue = np.max(nplistphase)
sx1 = np.linspace(0,1,2000)
nplistphase = np.sort(nplistphase)
func1 = interpolate.UnivariateSpline(nplistphase, nplistmag,s=s*s*num)#强制通过所有点
sy1 = func1(sx1)
indexmag = np.argmax(sy1)
nplistphase = nplistphase-sx1[indexmag]

phasemag = np.vstack((nplistphase, nplistmag)) #纵向合并矩阵
phasemag = phasemag.T
phasemag = phasemag[phasemag[:,0]>=0]
phasemag = phasemag[phasemag[:,0]<=1]

plt.figure(0)   
plt.plot(jd, flux, '.')

np.savetxt('cxh.txt', phasemag)
plt.figure(1)   
plt.plot(phasemag[:,0], phasemag[:,1], '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
