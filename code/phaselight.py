# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 08:49:49 2021

@author: dingxu
"""
import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate
import os,pickle,time
from scipy import interpolate

P = 0.5820740
CSV_FILE_PATH = '152.csv'

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
nplistphase = np.array(listphrase)

plt.figure(0)
plt.plot(nplistphase, nplistmag,'.')
try:
    s = np.diff(nplistmag,2).std()/np.sqrt(6)
    num = len(nplistmag)
    lvalue = np.max(nplistphase)
    sx1 = np.linspace(0,1,1000)
    func1 = interpolate.UnivariateSpline(nplistphase, nplistmag,s=s*s*num)#强制通过所有点
    sy1 = func1(sx1)
    indexmag = np.argmax(sy1)
    nplistphase = nplistphase-sx1[indexmag]
    plt.plot(sx1, sy1,'.')
except:
    sortmag = np.sort(nplistmag)
    maxindex = np.median(sortmag[-9:])
    indexmag = listmag.index(maxindex)
    nplistphase = nplistphase-nplistphase[indexmag]
#sy2 = 



phasemag = np.vstack((nplistphase, nplistmag)) #纵向合并矩阵
phasemag = phasemag.T
    
phasemag = phasemag[phasemag[:,0]>=0]
phasemag = phasemag[phasemag[:,0]<=1]

plt.figure(1)
plt.plot(phasemag[:,0], phasemag[:,1],'.')