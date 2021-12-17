# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:13:28 2021

@author: dingxu
"""

import scipy.signal as ss 
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate
import os,pickle,time
#import tools_32 as sim
#from pyhht.emd import EMD

file = 'Z:/DingXu/ZTF_jkf/alldata/0003.pkl'
dat = pickle.load(open(file,'rb'))
tot = len(dat)
for i in range(0, tot):
   
    idname = dat[i][0]
    name = dat[i][1]
    RA = dat[i][2]
    DEC = dat[i][3]
    P = dat[i][4]
    gmag = dat[i][5]
    rmag = dat[i][6]
    xy = dat[i][7] 
    num = xy.shape[0]
    

    phase,flux = xy[:,0],xy[:,1]
   

    sx1 = np.linspace(0,1,100)

    s = np.diff(flux,2).std()/np.sqrt(6)
    print('it is ok'+str(i))
    
    try:
        phase = np.sort(phase)
        func1 = interpolate.UnivariateSpline(phase, flux,k=3,s=s*s*num,ext=3)#强制通过所有点0.225
        sy1 = func1(sx1)
    
    
        plt.figure(2)
        plt.plot(phase,flux,'.')
        plt.plot(sx1,sy1)
        plt.title(str(i))
        plt.pause(0.1)
        plt.clf()
    except:
        print('it is error!')
        
    
    