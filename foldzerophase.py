# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:28:30 2021

@author: dingxu
"""

#Period: 0.3702918 ID: ZTFJ000006.67+641227.6 SourceID: 55

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate

P = 0.3933134
CSV_FILE_PATH = '396700.csv'
dfdata = pd.read_csv(CSV_FILE_PATH)
hjd = dfdata['HJD']
mag = dfdata['mag']
    
rg = dfdata['band'].value_counts()
lenr = rg['r']
nphjd = np.array(hjd)
npmag = np.array(mag)
    
hang = rg['g']
nphjd = nphjd[hang:]
npmag = npmag[hang:]-np.mean(npmag[hang:])
    
phases = foldAt(nphjd, P)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
resultmag = npmag[sortIndi]

listmag = resultmag.tolist()
listmag.extend(listmag)
    
listphrase = phases.tolist()
listphrase.extend(listphrase+np.max(1)) 

nplistmag = np.array(listmag)
sortmag = np.sort(nplistmag)

maxindex = np.median(sortmag[-5:])

indexmag = listmag.index(maxindex)


nplistphrase = np.array(listphrase)
nplistphrase = nplistphrase-nplistphrase[indexmag]
nplistmag = np.array(listmag)
    
phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
phasemag = phasemag.T
    
phasemag = phasemag[phasemag[:,0]>=0]
phasemag = phasemag[phasemag[:,0]<=1]
#######################


 
num = len(resultmag)   
sx1 = np.linspace(0,1,100)
s=np.diff(resultmag,2).std()/np.sqrt(6)
func1 = interpolate.UnivariateSpline(phases, resultmag,k=3,s=s*s*num,ext=3)#强制通过所有点0.225
sy1 = func1(sx1)

plt.figure(0)
plt.plot(listphrase, listmag, '.')
plt.plot(sx1, sy1, '.')
    
plt.figure(1)
plt.plot(phasemag[:,0], phasemag[:,1], '.')

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向





