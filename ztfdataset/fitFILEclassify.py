# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 16:54:15 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate
import os,pickle,time,shutil
import matplotlib.pyplot as plt

def ztf_2(CSV_FILE_PATH,P):
###############进行拼接##########################
    dfdata = pd.read_csv(CSV_FILE_PATH)
    
    hjd = dfdata['HJD']
    mag = dfdata['mag']
    
    rg = dfdata['band'].value_counts()
    try:
        lenr = rg['r']
    except:
        return np.array([[1.0,2.0],[3.0,4.0]])

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
    
#############以上进行拼接#####################
    
    nplistmag = np.array(listmag)
    nplistphase = np.array(listphrase)
    try:
        s = np.diff(nplistmag,2).std()/np.sqrt(6)
        num = len(nplistmag)
        #lvalue = np.max(nplistphase)
        sx1 = np.linspace(0,1,1000)
        nplistphase = np.sort(nplistphase)
        func1 = interpolate.UnivariateSpline(nplistphase, nplistmag,s=s*s*num)#强制通过所有点
        sy1 = func1(sx1)
        indexmag = np.argmax(sy1)
        nplistphase = nplistphase-sx1[indexmag]
        #nplistphrase = np.array(listphrase)
    except:
        try:
            sortmag = np.sort(nplistmag)
            maxindex = np.median(sortmag[-9:])
            indexmag = listmag.index(maxindex)
            nplistphase = nplistphase-nplistphase[indexmag]
        except:
            return np.array([[1.0,2.0],[3.0,4.0]])
#################以上求最大值对应的位置#########################
    phasemag = np.vstack((nplistphase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
     
    return phasemag




#tot=781602
tot=781602
w=10000
t1w=tot//w
dat=np.genfromtxt('Table2data.txt',dtype=str)

datemp = []
tot=dat.shape[0]
ID=0
for j in range(t1w+1):
    for i in range(w):
        if ID>(tot-1):
            break
        sourceid=dat[ID,1]
        P = float(dat[ID][4])
        gmag=float(dat[ID,8])
        dirnm='Z:/DingXu/ZTF_jkf/alldata/'+str(int(sourceid)//w).zfill(4)
        filename = dirnm+'/'+str(sourceid).zfill(7)+'.csv'
        
        if os.path.getsize(filename)>100:
            print(dat[ID,24].upper())
            print(sourceid)
                
            if (dat[ID,24].upper()=='EA'):
                pm = ztf_2(filename, P)             
                if len(pm[:,0]) > 100:
                    s1 = np.diff(pm[:,1],2).std()/np.sqrt(6)
                    s2 = np.std(pm[:,1])
                    sx1 = np.linspace(0,1,500)
                    sy1 = np.interp(sx1, pm[:,0], pm[:,1])
                    sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                    sx1sy1 = sx1sy1.T
                    if s2/s1>2:
                        np.savetxt('I:\\ZTFDATA\\YUANDATA\\EA\\'+sourceid+'.txt', sx1sy1)
               
            if (dat[ID,24].upper()=='EW' and ID<5000000):
                pm = ztf_2(filename, P)                             
                if len(pm[:,0]) > 100:
                    sx1 = np.linspace(0,1,500)
                    sy1 = np.interp(sx1, pm[:,0], pm[:,1])
                    s1 = np.diff(pm[:,1],2).std()/np.sqrt(6)
                    s2 = np.std(pm[:,1])
                    sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                    sx1sy1 = sx1sy1.T
                    if s2/s1>2.5:
                        np.savetxt('I:\\ZTFDATA\\YUANDATA\\EW\\'+sourceid+'.txt', sx1sy1)
                        
                                             
            if (dat[ID,24].upper()=='BYDRA'):
                pm = ztf_2(filename, P)             
                if len(pm[:,0]) > 100:
                    s1 = np.diff(pm[:,1],2).std()/np.sqrt(6)
                    s2 = np.std(pm[:,1])
                    sx1 = np.linspace(0,1,500)
                    sy1 = np.interp(sx1, pm[:,0], pm[:,1])
                    sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                    sx1sy1 = sx1sy1.T
                    if s2/s1>1.5:
                        np.savetxt('I:\\ZTFDATA\\YUANDATA\\BYDRA\\'+sourceid+'.txt', sx1sy1)
#                        plt.figure(2)
#                        plt.plot(pm[:,0], pm[:,1], '.')
#                        plt.plot(sx1,sy1,'.')
#                        plt.pause(0.1)
#                        plt.clf()
#                        ax1 = plt.gca()
#                        ax1.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
#                        ax1.invert_yaxis() #y轴反向
                            
           
            if (dat[ID,24].upper()=='RR'):
                pm = ztf_2(filename, P)             
                if len(pm[:,0]) > 100:
                    s1 = np.diff(pm[:,1],2).std()/np.sqrt(6)
                    s2 = np.std(pm[:,1])
                    sx1 = np.linspace(0,1,500)
                    sy1 = np.interp(sx1, pm[:,0], pm[:,1])
                    sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                    sx1sy1 = sx1sy1.T
                    if s2/s1>1.5:
                        np.savetxt('I:\\ZTFDATA\\YUANDATA\\RR\\'+sourceid+'.txt', sx1sy1)

            if (dat[ID,24].upper()=='RRC'):
                pm = ztf_2(filename, P)  
                if len(pm[:,0]) > 100:
                    s1 = np.diff(pm[:,1],2).std()/np.sqrt(6)
                    s2 = np.std(pm[:,1])
                    sx1 = np.linspace(0,1,500)
                    sy1 = np.interp(sx1, pm[:,0], pm[:,1])
                    sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                    sx1sy1 = sx1sy1.T
                    if s2/s1>1.5:
                        np.savetxt('I:\\ZTFDATA\\YUANDATA\\RRC\\'+sourceid+'.txt', sx1sy1)
              
                
            if (dat[ID,24].upper()=='RSCVN'):
                pm = ztf_2(filename, P)             
                if len(pm[:,0]) > 100:
                    s1 = np.diff(pm[:,1],2).std()/np.sqrt(6)
                    s2 = np.std(pm[:,1])
                    sx1 = np.linspace(0,1,500)
                    sy1 = np.interp(sx1, pm[:,0], pm[:,1])
                    sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                    sx1sy1 = sx1sy1.T
                    if s2/s1>1.5:
                        np.savetxt('I:\\ZTFDATA\\YUANDATA\\RSCVN\\'+sourceid+'.txt', sx1sy1)
                       
                    
            if (dat[ID,24].upper()=='CEP' or dat[ID,24].upper()=='CEPII'):
                pm = ztf_2(filename, P)                         
                if len(pm[:,0]) > 100:
                    s1 = np.diff(pm[:,1],2).std()/np.sqrt(6)
                    s2 = np.std(pm[:,1])
                    sx1 = np.linspace(0,1,500)
                    sy1 = np.interp(sx1, pm[:,0], pm[:,1])
                    sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                    sx1sy1 = sx1sy1.T
                    if s2/s1>1.5:
                        np.savetxt('I:\\ZTFDATA\\YUANDATA\\CEP\\'+sourceid+'.txt', sx1sy1)
        
            if (dat[ID,24].upper()=='DSCT'):
                pm = ztf_2(filename, P)                         
                if len(pm[:,0]) > 100:
                    s1 = np.diff(pm[:,1],2).std()/np.sqrt(6)
                    s2 = np.std(pm[:,1])
                    sx1 = np.linspace(0,1,500)
                    sy1 = np.interp(sx1, pm[:,0], pm[:,1])
                    sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                    sx1sy1 = sx1sy1.T
                    if s2/s1>1.5:
                        np.savetxt('I:\\ZTFDATA\\YUANDATA\\DSCT\\'+sourceid+'.txt', sx1sy1)
        ID+=1
            