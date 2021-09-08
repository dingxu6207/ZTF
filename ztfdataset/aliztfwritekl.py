# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:38:05 2021

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
###############进行拼接##########################
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
            return 0
#################以上求最大值对应的位置#########################
    phasemag = np.vstack((nplistphase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
     
    return phasemag


tot=781602
#tot=100
w=10000
t1w=tot//w
dat=np.genfromtxt('Table2data.txt',dtype=str)

tot=dat.shape[0]
ID=0
for j in range(t1w+1):

    dirnm0='alldata/'+str(j).zfill(4)
    
    t1=time.time()
    data=[]
    for i in range(w):
        if ID>(tot-1):
            break
        sourceid=dat[ID,1]
        dirnm='Z:/DingXu/ZTF_jkf/alldata/'+str(int(sourceid)//w).zfill(4)
        tmp=[]
        if (dat[ID,24].upper()=='EW'):
            name=dat[ID,0]
            gmag=float(dat[ID,8])
            HANG=int(dat[ID,12])
            P = float(dat[ID][4])
            
            filename = dirnm+'/'+str(sourceid).zfill(7)+'.csv' #175
            
            if os.path.getsize(filename)>100:
    
                tmp.extend([int(sourceid)])
                tmp.append(name)
                tmp.append(P)
                tmp.append(gmag)
                pm=ztf_2(filename,P) 
                tmp.append(pm)
                try:
                    plt.clf()
                    plt.figure(2)          
                    plt.plot(pm[:,0],pm[:,1],'.')
                    plt.title(sourceid)
                    plt.pause(1)
                except:
                    print('plot is error')
                data.append(tmp) 
                print(ID,sourceid,filename)

        ID+=1
        
    pickle.dump(data,open(dirnm0+'.pkl','wb'))
    print(time.time()-t1)
