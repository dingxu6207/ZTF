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
    stdres = np.std(res_ydata)
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((ydata - np.mean(ydata))**2)
    r_squared  = 1 - (ss_res / ss_tot)
    return stdres,r_squared

def model1R2(data, phrase, T1):
    predata = data[0].tolist()
    predata[1] = predata[1]/100
    predata[2] = predata[2]/100
    predata[3] = predata[3]/100
    print(predata)
    try:
        times,resultflux, pbdic, pr1, sr2 = fitfunction.plotphoebenol3T(predata,phrase,T1)
        stdr,r_squared = calculater(resultflux,flux)
        return stdr,r_squared,resultflux
    except:
        return 0,0,phrase
    
def model10R2(data, phrase, T1):
    predata = data[0].tolist()
    predata[1] = predata[1]/10
    predata[2] = predata[2]/100
    predata[3] = predata[3]/100
    try:
        times,resultflux, pbdic, pr1, sr2 = fitfunction.plotphoebenol3T(predata,phrase, T1)
        stdr,r_squared = calculater(resultflux,flux)
        return stdr,r_squared,resultflux
    except:
        return 0,0,phrase 
    
def l3model1R2(data, phrase, T1):
    predata = data[0].tolist()
    predata[1] = predata[1]/100
    predata[2] = predata[2]/100
    predata[3] = predata[3]/100
    predata[4] = predata[4]/100
    try:
        times,resultflux, pbdic, pr1, sr2 = fitfunction.plotphoebel3T(predata,phrase, T1)
        stdr,r_squared = calculater(resultflux,flux)
        return stdr,r_squared,resultflux
    except:
        return 0,0,phrase
 
def l3model10R2(data, phrase, T1):
    predata = data[0].tolist()
    predata[1] = predata[1]/10
    predata[2] = predata[2]/100
    predata[3] = predata[3]/100
    predata[4] = predata[4]/100
    #print(predata)
    try:
        times, resultflux, pbdic, pr1, sr2 = fitfunction.plotphoebel3T(predata,phrase,T1)
        stdr,r_squared = calculater(resultflux,flux)
        return stdr,r_squared,resultflux
    except:
        return 0,0,phrase


path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 8029708.txt'
data = np.loadtxt(path+file)
#fileone = 'phasemag.txt'
#data = np.loadtxt(fileone)
phrase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])
flux = datay
sx1 = np.linspace(0,1,100)
s = np.diff(datay,2).std()/np.sqrt(6)
num = len(datay)

func1 = interpolate.UnivariateSpline(data[:,0], datay,s=s*s*num)#?????????????????????
sy1 = func1(sx1)
nparraydata = np.reshape(sy1,(1,100))
incldata = inclmodel.predict(nparraydata)
inclcom = incldata[0][0]

temperature = 5226/5850
listsys = sy1.tolist()
listsys.append(temperature)
npsy1 = np.array(listsys)
nparraydata = np.reshape(npsy1,(1,101))

if inclcom>40:
    predict1 = model1.predict(nparraydata)
    predict10 = model10.predict(nparraydata)
    l3predict1 = l3model1.predict(nparraydata)
    l3predict10 = l3model10.predict(nparraydata)
            
    stdre1,r1,mag1 = model1R2(predict1, phrase, temperature)
    print('*******************1******************')
    stdre2,r2,mag2 = model10R2(predict10, phrase, temperature)
    print('*******************2******************')
    stdre3,r3,mag3 = l3model1R2(l3predict1, phrase, temperature)
    print('*******************3******************')
    stdre4,r4,mag4 = l3model10R2(l3predict10, phrase, temperature)
    print('*******************4******************')
            
    R = [r1,r2,r3,r4]
    index = np.argmax(R)
       
    if R[index]<0.6:
        print('it is bad!')
    
    if index==0:
        resultflux = mag1
        predata1 = predict1[0].tolist()
        print(stdre1/s)
        print('nol31=', predata1)
    
    if index==1:
        resultflux = mag2
        predata10 = predict10[0].tolist()
        print(stdre2/s)
        print('nol310=', predata10)
        
    if index==2:
        resultflux = mag3
        l3predata1 = l3predict1[0].tolist()
        print(stdre3/s)
        print('l31=', l3predata1)

    if index==3:
        resultflux = mag4
        l3predata10 = l3predict10[0].tolist()
        print(stdre4/s)
        print('l310=', l3predata10)
        
    print('R2=', R[index])
    try:

        plt.figure(1)
        plt.plot(phrase,flux,'.')
        #plt.plot(sx1,sy1,'.')
        plt.plot(phrase, resultflux,'.')
        #plt.scatter(phrase, resultflux, c='none',marker='o',edgecolors='r', s=40)
        ax = plt.gca()
        ax.yaxis.set_ticks_position('left') #???y???????????????????????????
        ax.invert_yaxis() #y?????????          

    except:
        print('error')
else:
    print('incl=', inclcom)
        
