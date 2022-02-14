# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 23:24:38 2022

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
        return 0
    
    hjdmag = dfdata[['HJD', 'mag']] 
    nphjdmag = np.array(hjdmag)
    nphjd = nphjdmag[:,0]
    npmag = nphjdmag[:,1]
    
   
    #相位折叠和拼接
    phases = foldAt(nphjd, P)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = npmag[sortIndi]

    
    s = np.diff(resultmag,2).std()/np.sqrt(6)
    #提取r波段数据

    return s

tmp = []
tot = 10
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
        #tmp = []
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
                   
                pm = ztf_2(filename,P) 
                
                if pm != 0:
                    tmp.append(pm)
                    print('std_noise = ', pm)
                                  
        ID+=1
        
np.savetxt('std_nose.txt', np.array(tmp))
import numpy as np 
import matplotlib.pylab as plt
tmp = np.loadtxt('std_nose.txt')
plt.hist(np.array(tmp), bins=100)
plt.title(r"$\mu$"+'='+str(np.round(np.mean(tmp),4)))
plt.xlabel('Photometric error',fontsize=18)
plt.ylabel('frequency',fontsize=18)