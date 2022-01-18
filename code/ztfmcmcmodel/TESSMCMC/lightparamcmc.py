# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:24:09 2021

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
import pandas as pd

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\model\\'
inclmodel = load_model(path+'incl.hdf5')
model1 = load_model(path+'model1.hdf5')
model10 = load_model(path+'model10.hdf5')
l3model1 = load_model(path+'model1l3.hdf5')
l3model10 = load_model(path+'model10l3.hdf5')
model10mc = load_model(path+'model10mc.hdf5')
l3model10mc = load_model(path+'model10l3mc.hdf5')

def calculater(ydata, caldata):
    res_ydata  = np.array(ydata) - np.array(caldata)
    stdres = np.std(res_ydata)
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((ydata - np.mean(ydata))**2)
    r_squared  = 1 - (ss_res / ss_tot)
    return stdres,r_squared


def model1R2(data, sy1, T1):
    temp = []
    predata = data[0].tolist()
    temp.append(T1)
    temp.append(predata[0]/90) #incl
    temp.append(predata[1]/100)#q
    temp.append(predata[2]/100)
    temp.append(predata[3]/100)
    
    arraymc = np.array(temp)
    mcinput = np.reshape(arraymc,(1,5))
    lightdata = model10mc.predict(mcinput)
    stdr, r_squared = calculater(sy1, lightdata[0])
  
    return lightdata, stdr, r_squared  


def model10R2(data, sy1, T1):
    temp = []
    predata = data[0].tolist()
    temp.append(T1)
    temp.append(predata[0]/90)#incl
    temp.append(predata[1]/10)#q
    temp.append(predata[2]/100)
    temp.append(predata[3]/100)
    
    arraymc = np.array(temp)
    mcinput = np.reshape(arraymc,(1,5))
    lightdata = model10mc.predict(mcinput)
    stdr, r_squared = calculater(sy1, lightdata[0])
    
    return lightdata, stdr, r_squared  

def model1l3R2(data, sy1, T1):
    temp = []
    predata = data[0].tolist()
    temp.append(T1)
    temp.append(predata[0]/90) #incl
    temp.append(predata[1]/100)#q
    temp.append(predata[2]/100)
    temp.append(predata[3]/100)
    temp.append(predata[4]/100)
    
    arraymc = np.array(temp)
    mcinput = np.reshape(arraymc,(1,6))
    lightdata = l3model10mc.predict(mcinput)
    stdr, r_squared = calculater(sy1, lightdata[0])
    
    return lightdata, stdr, r_squared  

def model10l3R2(data, sy1, T1):
    temp = []
    predata = data[0].tolist()
    temp.append(T1)
    temp.append(predata[0]/90) #incl
    temp.append(predata[1]/10)#q
    temp.append(predata[2]/100)
    temp.append(predata[3]/100)
    temp.append(predata[4]/100)
    
    arraymc = np.array(temp)
    mcinput = np.reshape(arraymc,(1,6))
    lightdata = l3model10mc.predict(mcinput)
    stdr, r_squared = calculater(sy1, lightdata[0])
    
    return lightdata, stdr, r_squared  

def tupledata(data, T1, flag):
    temp = []
    predata = data[0].tolist()
    tup1 = (T1*0.8, 1.2*T1)
    tup2 = ((predata[0]/90)*0.8, (predata[0]/90)*1.2)
    
    if flag==0 or flag==2:
        tup3 = ((predata[1]/100)*0.8, (predata[1]/100)*1.2)
        
    if flag==1 or flag==3:
        tup3 = ((predata[1]/10)*0.8, (predata[1]/10)*1.2)
    
    tup4 = ((predata[2]/100)*0.8, (predata[2]/100)*1.2)
    tup5 = ((predata[3]/100)*0.8, (predata[3]/100)*1.2)
    
    if flag==2 or flag==3:
        tup6 = ((predata[4]/100)*0.8, (predata[4]/100)*1.2)
        temp = [tup1, tup2, tup3, tup4, tup5, tup6]
    else:
        temp = [tup1, tup2, tup3, tup4, tup5]
    return temp

def interone(datax, datay):
    interdata = np.copy(datay)
    interdata = interdata -np.mean(interdata)
    sx1 = np.linspace(0,1,100)
    s = np.diff(interdata,2).std()/np.sqrt(6)
    num = len(datay)

    func1 = interpolate.UnivariateSpline(datax, interdata,s=s*s*num)#强制通过所有点
    sy1 = func1(sx1)
    
    return sx1,sy1

def inclprediction(data):
    inputdata  = np.copy(data)
    nparraydata = np.reshape(inputdata,(1,100))
    incldata = inclmodel.predict(nparraydata)
    inclcom = incldata[0][0]
    return inclcom
    
def dataaddT(data, T1):
    inputdata = np.copy(data)
    listsys = inputdata.tolist()
    listsys.append(T1)
    npsy1 = np.array(listsys)
    nparraydata = np.reshape(npsy1,(1,101))
    return nparraydata

#######################################################################
#######################################################################
#path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
#file = 'KIC 11924311.txt'
#data = np.loadtxt(path+file)
    
path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\EWDATA\\'
file = 'TIC 55896456.txt'
data = np.loadtxt(path+file)

#fileone = 'ZTFtestdata.txt'
#data = np.loadtxt(fileone)
phase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])


sx1,sy1 = interone(phase, datay)
inclcom = inclprediction(sy1)

T1 =  6662.70/5850
nparraydata = dataaddT(sy1, T1)


if inclcom>40:
    predict1 = model1.predict(nparraydata)
    predict10 = model10.predict(nparraydata)
    l3predict1 = l3model1.predict(nparraydata)
    l3predict10 = l3model10.predict(nparraydata)
    
    ligdata1, stdr1, r_squared1 = model1R2(predict1, sy1, T1)
    ligdata10, stdr10, r_squared10 = model10R2(predict10, sy1, T1)
    ligdata1l3, stdr1l3, r_squared1l3 = model1l3R2(l3predict1, sy1, T1)
    ligdata10l3, stdr10l3, r_squared10l3 = model10l3R2(l3predict10, sy1, T1)
    
    R = [r_squared1, r_squared10, r_squared1l3, r_squared10l3]
    index = np.argmax(R)
    print('index= '+str(index)+'  R2='+str(R[index]))
    
    
    plt.figure(1)
    plt.plot(sx1,sy1,'.') 
    if index == 0 :    
        temppre = tupledata(predict1, T1, 0)
        plt.plot(sx1, ligdata1[0],'.')
    if index == 1 :    
        temppre = tupledata(predict10, T1, 1)
        plt.plot(sx1, ligdata10[0],'.')
    if index == 2 : 
        temppre = tupledata(l3predict1, T1, 2)
        plt.plot(sx1, ligdata1l3[0],'.')
    if index == 3 :   
        temppre = tupledata(l3predict10, T1, 3)
        plt.plot(sx1, ligdata10l3[0],'.')
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向  
    print(temppre)
    
    