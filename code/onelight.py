# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:49:45 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate
import os,pickle,time
#dat=np.genfromtxt('table2data.txt',dtype=str)


def ztf_2(CSV_FILE_PATH,P):
    dfdata = pd.read_csv(CSV_FILE_PATH)
    
    hjd = dfdata['HJD']
    mag = dfdata['mag']
    
    rg = dfdata['band'].value_counts()
    try:
        lenr = rg['r']
    except:
        return 0

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
    try:
        maxindex = np.median(sortmag[-9:])
        indexmag = listmag.index(maxindex)
    except:
        return 0
    
    nplistphrase = np.array(listphrase)
    nplistphrase = nplistphrase-nplistphrase[indexmag]
    nplistmag = np.array(listmag)
    
    phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
    
    return phasemag

P = 0.3347166 
CSV_FILE_PATH = '8560.csv'
phasemag = ztf_2(CSV_FILE_PATH,P)
plt.figure(0)
plt.plot(phasemag[:,0], phasemag[:,1],'.')