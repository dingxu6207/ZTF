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
    dfdata = dfdata[dfdata['band']=='r']
    if len(dfdata)<30:
        return [0,0]
        
    #提取r波段数据
    hjdmag = dfdata[['HJD', 'mag']] 
    nphjdmag = np.array(hjdmag)
    nphjd = nphjdmag[:,0]
    npmag = nphjdmag[:,1]
    #相位折叠和拼接
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
            return [0,0]
#################以上求最大值对应的位置#########################
    phasemag = np.vstack((nplistphase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
     
    return phasemag


tot = 781602
#tot=100
w = 10000
t1w = tot//w
dat = np.genfromtxt('Table2data.txt',dtype=str)

tot = dat.shape[0]
ID = 0
for j in range(t1w+1):

    dirnm0='Z:/DingXu/ZTF_jkf/alldata/'+str(j).zfill(4)
    
    t1 = time.time()
    data = []
    for i in range(w):
        if ID>(tot-1):
            break
        sourceid = dat[ID,1]
        dirnm='Z:/DingXu/ZTF_jkf/alldata/'+str(int(sourceid)//w).zfill(4)
        tmp = []
        if (dat[ID,24].upper()=='EW'):
            name=dat[ID,0]
            RA = float(dat[ID,2])
            DEC = float(dat[ID,3])
            P = float(dat[ID][4])
            gmag = float(dat[ID,8])
            rmag = float(dat[ID,9])
            HANG=int(dat[ID,12])
            
            filename = dirnm+'/'+str(sourceid).zfill(7)+'.csv' #175
            
            if os.path.getsize(filename)>100:
    
                tmp.append([int(sourceid)])
                tmp.append(name)
                tmp.append(RA)
                tmp.append(DEC)
                tmp.append(P)
                tmp.append(gmag)
                tmp.append(rmag)
                pm = ztf_2(filename,P) 
                if len(pm)>30:
                    tmp.append(pm)
                    data.append(tmp) 
                    print(ID,sourceid,filename)
               
#                    plt.clf()
#                    plt.figure(0)          
#                    plt.plot(pm[:,0],pm[:,1],'.')
#                    plt.title(sourceid)
#                    plt.pause(0.1)
                   

        ID+=1
        
    pickle.dump(data,open(dirnm0+'.pkl','wb'))
    print(time.time()-t1)
