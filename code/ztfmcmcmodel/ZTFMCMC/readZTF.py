# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:51:23 2021

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
        return [0,0],0,0

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
    
    dexin = int(1*lenr/2)
    try:
        indexmag = listmag.index(max(listmag[0:dexin]))
    except:
        indexmag = listmag.index(max(listmag))
    
    nplistphrase = np.array(listphrase)
    nplistphrase = nplistphrase-nplistphrase[indexmag]
    nplistmag = np.array(listmag)
    
    phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
    
    #去除异常点
    mendata = np.mean(phasemag[:,1])
    stddata = np.std(phasemag[:,1])
    sigmamax = mendata+4*stddata
    sigmamin = mendata-4*stddata
    
    phasemag = phasemag[phasemag[:,1] > sigmamin]
    phasemag = phasemag[phasemag[:,1] < sigmamax]
    
    dmag1=np.diff(phasemag,2).std()/np.sqrt(6)
    dmag2=np.diff(phasemag,2).std()/np.sqrt(6)
    
    return phasemag,dmag1,dmag2

