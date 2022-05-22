# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:30:30 2022

@author: dingxu
"""

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate
import os,pickle,time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ztfquery import lightcurve
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
from astropy.timeseries import LombScargle
import os
#dat=np.genfromtxt('table2data.txt',dtype=str)

def ztf_2(CSV_FILE_PATH, P):
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
        sx1 = np.linspace(0,1,10000)
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

def zerophasemag(phases, resultmag):
    
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
        sx1 = np.linspace(0,1,10000)
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
    
def zerophasemagN(phases, resultmag):
    listmag = resultmag.tolist()
    listmag.extend(listmag)  
    listphrase = phases.tolist()
    listphrase.extend(listphrase+np.max(phases)) 
    #############以上进行拼接#####################
    
    nplistmag = np.array(listmag)
    nplistphase = np.array(listphrase)

    s = np.diff(nplistmag,2).std()/np.sqrt(6)
    num = len(nplistmag)
    sx1 = np.linspace(0,1,10000)
    nplistphase = np.sort(nplistphase)
    func1 = interpolate.UnivariateSpline(nplistphase, nplistmag,s=s*s*num)#强制通过所有点
    sy1 = func1(sx1)
    indexmag = np.argmax(sy1)
    nplistphase = nplistphase-sx1[indexmag]
    #nplistphrase = np.array(listphrase)

    #################以上求最大值对应的位置#########################
    phasemag = np.vstack((nplistphase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
    
    return phasemag    

def computeperiod(npjdmag):
    JDtime = npjdmag[:,0]
    targetflux = npjdmag[:,1]
    ls = LombScargle(JDtime, targetflux, normalization='model')
    frequency, power = ls.autopower(minimum_frequency=0.025,maximum_frequency=20)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower, frequency, power


def pholdata(npjdmag, P):
    phases = foldAt(npjdmag[:,0], P)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = npjdmag[:,1][sortIndi]
    
    return phases, resultmag

def computePDM(npjdmag,P):
    timedata = npjdmag[:,0]
    magdata = npjdmag[:,1]
    f0 =1/(2*P) 
    S = pyPDM.Scanner(minVal=f0-0.01, maxVal=f0+0.01, dVal=0.001, mode="frequency")
    P = pyPDM.PyPDM(timedata, magdata)
    bindata = int(len(magdata)/4)
    #bindata = 10
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    return pdmp, delta

def showfig(phases, resultmag, num):
    plt.figure(num)
    plt.plot(phases, resultmag, '.')
    plt.xlabel('phase',fontsize=14)
    plt.ylabel('mag',fontsize=14)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向
    #plt.pause(1)
    #plt.clf()
    
    
def showmjdmag(mjdmag, num):
    plt.figure(num)
    #plt.clf()
    mjd = mjdmag[:,0]
    mag = mjdmag[:,1]
    plt.plot(mjd, mag, '.')
    plt.xlabel('MJD',fontsize=14)
    plt.ylabel('mag',fontsize=14)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向
    #plt.pause(1)
 
def stddata(npjdmag, P):
    yuandata = np.copy(npjdmag[:,1])
    phases, resultmag = pholdata(npmjdmag, P)
    datamag = np.copy(resultmag)
    datanoise = np.diff(datamag,2).std()/np.sqrt(6)
    stdndata = np.std(yuandata)
    return stdndata/datanoise

RAp = 290.10406494140625
DECp = 37.82181167602539
lcq = lightcurve.LCQuery.from_position(RAp, DECp, 1)   
dfdata = lcq.data
dfdata = dfdata[dfdata["filtercode"]=='zr']

dfdata = dfdata[dfdata["catflags"]!= 32768]
mjdmag = dfdata[['mjd', 'mag']]
npmjdmag = np.array(mjdmag)
period, wrongP, maxpower, frequency, power = computeperiod(npmjdmag)

P1 = period
P2 = period*2
stddata1 = stddata(npmjdmag, P1)
stddata2 = stddata(npmjdmag, P2)
#showmjdmag(npmjdmag)

phases, resultmag = pholdata(npmjdmag, P2)
showfig(phases, resultmag, 3)

plt.figure(0)
ax = plt.gca()
ax.plot(npmjdmag[:,0], npmjdmag[:,1], '.')
plt.xlabel('mjd',fontsize=18)
plt.ylabel('mag',fontsize=18)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.figure(1)
plt.plot(frequency, power)
plt.xlabel('frequency', fontsize=18)
plt.ylabel('power', fontsize=18)

