# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:20:30 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate

CSV_FILE_PATH = '55.csv'
dfdata = pd.read_csv(CSV_FILE_PATH)

dfdata = dfdata[dfdata['band']=='r']

hjd = dfdata['HJD']
mag = dfdata['mag']

hjdmag = dfdata[['HJD', 'mag']] 
nphjdmag = np.array(hjdmag)

nphjd = nphjdmag[:,0]
npmag = nphjdmag[:,1]



P = 0.3702918
phases = foldAt(nphjd, P)
sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
resultmag = npmag[sortIndi]

#plt.plot(phases, resultmag,'.')

listmag = resultmag.tolist()
listmag.extend(listmag)

listphrase = phases.tolist()
listphrase.extend(listphrase+np.max(listphrase)) 

indexmag = listmag.index(max(listmag))

nplistphrase = np.array(listphrase)
nplistphrase = nplistphrase-nplistphrase[indexmag]
nplistmag = np.array(listmag)

#phasemag = np.concatenate([nplistphrase, nplistmag],axis=1)


phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
phasemag = phasemag.T

phasemag = phasemag[phasemag[:,0]>0]
phasemag = phasemag[phasemag[:,0]<1]

#去除异常点
mendata = np.mean(phasemag[:,1])
stddata = np.std(phasemag[:,1])
sigmamax = mendata+2*stddata
sigmamin = mendata-2*stddata

phasemag = phasemag[phasemag[:,1] > sigmamin]
phasemag = phasemag[phasemag[:,1] < sigmamax]


phrase = phasemag[:,0]
flux = phasemag[:,1]
sx1 = np.linspace(0.01,0.99,100)
func1 = interpolate.UnivariateSpline(phrase, flux,s=0.12)#强制通过所有点
sy1 = func1(sx1)


plt.figure(1)
plt.plot(sx1, sy1,'.', c='r')#对原始数据画散点图

plt.figure(0)
plt.plot(phrase, flux,'.')
plt.plot(sx1, sy1,'.', c='r')#对原始数据画散点图
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

interdata = np.vstack((sx1,sy1))
np.savetxt('ztf1.txt', interdata.T)
praflux = np.vstack((phrase, flux))
np.savetxt('ztf2.txt', praflux.T)

