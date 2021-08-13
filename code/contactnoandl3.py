# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 22:34:44 2021

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
from tensorflow.keras.models import load_model
#import tools_32 as sim
#from pyhht.emd import EMD
import pandas as pd
import fitfunction

inclmodel = load_model('incl.hdf5')
model1 = load_model('model1.hdf5')
model10 = load_model('model10.hdf5')
l3model1 = load_model('model1l3.hdf5')
l3model10 = load_model('model10l3.hdf5')

def calculater(ydata, caldata):
    res_ydata  = np.array(ydata) - np.array(caldata)
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((ydata - np.mean(ydata))**2)
    r_squared  = 1 - (ss_res / ss_tot)
    return r_squared

def model1R2(data, phrase):
    predata = data[0].tolist()
    predata[1] = predata[1]/100
    predata[2] = predata[2]/100
    predata[3] = predata[3]/100
    try:
        times,resultflux= fitfunction.plotphoebenol3(predata,phrase)
        r_squared = calculater(resultflux,flux)
        return r_squared,resultflux
    except:
        return 0,phrase
    
def model10R2(data, phrase):
    predata = data[0].tolist()
    predata[1] = predata[1]/10
    predata[2] = predata[2]/100
    predata[3] = predata[3]/100
    try:
        times,resultflux= fitfunction.plotphoebenol3(predata,phrase)
        r_squared = calculater(resultflux,flux)
        return r_squared,resultflux
    except:
        return 0,phrase 
    
def l3model1R2(data, phrase):
    predata = data[0].tolist()
    predata[1] = predata[1]/100
    predata[2] = predata[2]/100
    predata[3] = predata[3]/100
    predata[4] = predata[4]/100
    try:
        times,resultflux= fitfunction.plotphoebel3(predata,phrase)
        r_squared = calculater(resultflux,flux)
        return r_squared,resultflux
    except:
        return 0,phrase
 
def l3model10R2(data, phrase):
    predata = data[0].tolist()
    predata[1] = predata[1]/10
    predata[2] = predata[2]/100
    predata[3] = predata[3]/100
    predata[3] = predata[3]/100
    try:
        times,resultflux= fitfunction.plotphoebel3(predata,phrase)
        r_squared = calculater(resultflux,flux)
        return r_squared,resultflux
    except:
        return 0,phrase


path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 11336707.txt'
data = np.loadtxt(path+file)
phrase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])
flux = datay
sx1 = np.linspace(0,1,100)
s = np.diff(datay,2).std()/np.sqrt(6)
num = len(datay)
func1 = interpolate.UnivariateSpline(data[:,0], datay,s=s*s*num)#强制通过所有点
sy1 = func1(sx1)
nparraydata = np.reshape(sy1,(1,100))
incldata = inclmodel.predict(nparraydata)
inclcom = incldata[0][0]

if inclcom>50:
    predict1 = model1.predict(nparraydata)
    predict10 = model10.predict(nparraydata)
    l3predict1 = l3model1.predict(nparraydata)
    l3predict10 = l3model10.predict(nparraydata)
            
    r1,mag1 = model1R2(predict1, phrase)
    print('*******************1******************')
    r2,mag2 = model10R2(predict10, phrase)
    print('*******************2******************')
    r3,mag3 = model1R2(l3predict1, phrase)
    print('*******************3******************')
    r4,mag4 = model10R2(l3predict10, phrase)
    print('*******************4******************')
            
    R = [r1,r2,r3,r4]
    index = np.argmax(R)
            
    if R[index]<0.6:
        print('it is bad!')
    
    if index==0:
        resultflux = mag1
        predata1 = predict1[0].tolist()
        print('nol31=', predata1)
    
    if index==1:
        resultflux = mag2
        predata10 = predict10[0].tolist()
        print('nol310=', predata10)
        
    if index==2:
        resultflux = mag3
        l3predata1 = l3predict1[0].tolist()
        print('l31=', l3predata1)

    if index==3:
        resultflux = mag4
        l3predata10 = l3predict10[0].tolist()
        print('l310=', l3predata10)
        
    print('R2=', R[index])
    try:

        plt.figure(1)
        plt.plot(phrase,flux,'.')
        #plt.plot(sx1,sy1,'.')
        #plt.plot(phrase, resultflux,'.')
        plt.scatter(phrase, resultflux, c='none',marker='o',edgecolors='r', s=40)
        ax = plt.gca()
        ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
        ax.invert_yaxis() #y轴反向          

    except:
        print('error')
else:
    print('incl=', inclcom)
        
