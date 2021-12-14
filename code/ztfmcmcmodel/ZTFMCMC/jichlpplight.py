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
import emcee

inclmodel = load_model('incl.hdf5')
model1 = load_model('model1.hdf5')
model10 = load_model('model10.hdf5')
l3model1 = load_model('model1l3.hdf5')
l3model10 = load_model('model10l3.hdf5')
model10mc = load_model('model10mc.hdf5')
l3model10mc = load_model('model10l3mc.hdf5')

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
    tup2 = ((predata[0]/90)*0.7, (predata[0]/90)*1.3)
    
    if flag==0 or flag==2:
        tup3 = ((predata[1]/100)*0.5, (predata[1]/100)*1.5)
        
    if flag==1 or flag==3:
        tup3 = ((predata[1]/10)*0.5, (predata[1]/10)*1.5)
    
    tup4 = ((predata[2]/100)*0.7, (predata[2]/100)*1.3)
    tup5 = ((predata[3]/100)*0.7, (predata[3]/100)*1.3)
    
    if flag==2 or flag==3:
        tup6 = ((predata[4]/100)*0.7, (predata[4]/100)*1.3)
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

#########################MCMC##########################################
#########################mcmc##########################################
    
def intertwo(datax, datay):
    inputdata = np.copy(datay)
    phase = np.copy(datax)
    x = np.linspace(0,1,100) #x轴
    noisy = np.interp(x, phase, inputdata) #y轴
    sigma = np.diff(noisy,2).std()/np.sqrt(6)
    return x, noisy, sigma



def predict(allpara):
    
    arraymc = np.array(allpara)
    
    if modelindex == 0:
        mcinput = np.reshape(arraymc,(1,5))
        lightdata = model10mc.predict(mcinput)
        
    if modelindex == 1:
        mcinput = np.reshape(arraymc,(1,6))
        lightdata = l3model10mc.predict(mcinput)
        
    return lightdata[0]


def rpars(init_dist):#在ndim 维度上，在初始的范围里面均匀撒ndim个点
    return [np.random.rand() * (i[1]-i[0]) + i[0] for i in init_dist] 


def lnprior(priors, values):#判断MCMC新的点是否在初始的区域里面
    
    lp = 0.
    for value, prior in zip(values, priors):
        if value >= prior[0] and value <= prior[1]:
            lp+=0
        else:
            lp+=-np.inf 
    return lp


def lnprob(z): #计算后仰概率
    
    lnp = lnprior(priors,z)#判断MCMC新的点是否在初始的区域里面

    if not np.isfinite(lnp):
            return -np.inf


    output = predict(z)

    lnp = -0.5*np.sum(np.log(2 * np.pi * sigma ** 2)+(output-noisy)**2/(sigma**2)) #计算似然函数
      
    return lnp


def run(init_dist, nwalkers, niter,nburn):
    
    ndim = len(init_dist)
    # Generate initial guesses for all parameters for all chains
    p0 = [rpars(init_dist) for i in range(nwalkers)] #均匀撒ndim*nwalkers点
 #   print(p0)

    # Generate the emcee sampler. Here the inputs provided include the 
    # lnprob function. With this setup, the first parameter
    # in the lnprob function is the output from the sampler (the paramter 
    # positions).
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob) #建立MCMC模型
    pos, prob, state = sampler.run_mcmc(p0, niter) # 撒点
    emcee_trace = sampler.chain[:, -nburn:, :].reshape(-1, ndim).T #保留最后nburn 个点做统计

    return emcee_trace 

    

    
path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 6050116.txt'
data = np.loadtxt(path+file)
phase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])


sx1,sy1 = interone(phase, datay)
inclcom = inclprediction(sy1)

T1 = 4569/5850
nparraydata = dataaddT(sy1, T1)

###############MCMC#######################
x, noisy, sigma = intertwo(phase, datay)
nwalkers = 20
niter = 500
nburn=200 #保留最后多少点用于计算
##############MCMC#########################

if inclcom>40:
    predict1 = model1.predict(nparraydata)
    predict10 = model10.predict(nparraydata)
    l3predict1 = l3model1.predict(nparraydata)
    l3predict10 = l3model10.predict(nparraydata)
    
    ligdata1, stdr1, r_squared1 = model1R2(predict1, sy1, T1)
    ligdata10, stdr10, r_squared10 = model10R2(predict10, sy1, T1)
    ligdata1l3, stdr1l3, r_squared1l3 = model1l3R2(l3predict1, sy1, T1)
    ligdata10l3, stdr10l3, r_squared10l3 = model10l3R2(l3predict10, sy1, T1)
    
    R = [r_squared1, r_squared10, r_squared1l3, r_squared1l3]
    index = np.argmax(R)
    print('index= '+str(index)+'  R2='+str(R[index]))
    
    
    plt.figure(1)
    ax = plt.gca()
    ax.plot(x, noisy, '.') 
    
    if index == 0 :    
        temppre = tupledata(predict1, T1, 0)
        modelindex = 0
        ax.plot(sx1, ligdata1[0],'-b')
    if index == 1 :    
        temppre = tupledata(predict10, T1, 1)
        modelindex = 0
        ax.plot(sx1, ligdata10[0],'-b')
    if index == 2 : 
        temppre = tupledata(l3predict1, T1, 2)
        modelindex = 1
        ax.plot(sx1, ligdata1l3[0],'-b')
    if index == 3 :   
        temppre = tupledata(l3predict10, T1, 3)
        modelindex = 1
        ax.plot(sx1, ligdata10l3[0],'-b')
  
    
    print(temppre)
    
###################################################
    #初始范围[T，incl,q,f,t2t1,l3]
    init_dist = temppre.copy()
    priors=init_dist.copy()
    ndim = len(priors) #维度数   
    emcee_trace  = run(priors, nwalkers, niter,nburn) #run mcmc
    mu=(emcee_trace.mean(axis=1)) #参数均值
    sigma=(emcee_trace.std(axis=1)) #参数误差
    print('mu=',mu)
    print('sigma=',sigma)
    
    pre = predict(mu.reshape(1,-1))
    ax.plot(x, pre.flatten(),'-r') #理论数据
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向 
    
    