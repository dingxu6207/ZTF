# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 22:45:57 2022

@author: dingxu
"""

import numpy as np
from tensorflow.keras.models import load_model
import emcee
import matplotlib.pylab as plt
import pandas as pd
import time

#############################################################
def quantile(x, q, weights=None): 
 
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()
#########################################################
        
#####################################################
def tupledata(data):
    
    tup1 = (data[0]*0.999, data[0]*1.001)
    tup2 = ((data[1])*0.8, (data[1])*1.2)
    tup3 = ((data[2])*0.8, (data[2])*1.2)
    tup4 = ((data[3])*0.8, (data[3])*1.2)
    tup5 = ((data[4])*0.8, (data[4])*1.2)
    
    temp = [tup1, tup2, tup3, tup4, tup5]

    return temp

def intertwo(datax, datay):
    inputdata = np.copy(datay)
    phase = np.copy(datax)
    x = np.linspace(0,1,100) #x轴
    noisy = np.interp(x, phase, inputdata) #y轴
    sigma = np.diff(noisy,2).std()/np.sqrt(6)
    return x, noisy, sigma

######################################################
def predict(allpara, modelindex = 0):
    
    arraymc = np.array(allpara)
    
    if modelindex == 0:
        mcinput = np.reshape(arraymc,(1,5))
        lightdata = model10mcmc(mcinput)
        
    if modelindex == 1:
        mcinput = np.reshape(arraymc,(1,6))
        lightdata = model10l3mcmc(mcinput)
        
    return np.float32(lightdata[0])

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
#########################################################
   


mpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\model\\'

data = np.loadtxt('testdata01050.txt')
np.random.shuffle(data)
noise = np.random.normal(0,0.001,100)

model10mcmc = load_model(mpath+'model10mcnew.hdf5')
model10l3mcmc = load_model(mpath+'model10l3mc.hdf5')

alltemp = []
for i in range(0, 1000):
    t1=time.time()
    magdata = -2.5*np.log10(data[i,0:100])
    meanmag = magdata - np.mean(magdata)
    meanmagnoise = meanmag + noise
    maxmin = np.max(meanmagnoise)-np.min(meanmagnoise)
    print('maxminmag=', maxmin)
    print('i=', i)
    
    predictdata = [data[i,100], data[i,102]/90, data[i,103], data[i,104], data[i,105]]
    
    temppre = tupledata(predictdata)
    #print(temppre)
    
    phase = np.linspace(0,1,100)
    datay = np.copy(meanmag)
    x, noisy, sigma = intertwo(phase, datay)
    
    ###########################################
    x, noisy, sigma = intertwo(phase, datay)
    nwalkers = 20
    niter = 500
    nburn = 200 #保留最后多少点用于计算
    init_dist = temppre.copy()
    priors = init_dist.copy()
    ndim = len(priors) #维度数 
    emcee_trace  = run(priors, nwalkers, niter,nburn) #run mcmc
    
    mu = []
    sigma_1 = []
    sigma_2 = []
    sigma = []
    for mi, x1 in enumerate(emcee_trace):
        q_16, q_50, q_84 = quantile(x1, [0.16, 0.5, 0.84])          
        q_m, q_p = q_50 - q_16, q_84 - q_50
  
        mu.append(q_50) #median value
        sigma_1.append(q_m) #high limitation
        sigma_2.append(q_p) #low
        sigma.append((q_m+q_p)/2)
    
    sigma_1 = np.array(sigma_1)
    sigma_2 = np.array(sigma_2) 
    sigma = np.array(sigma)
    
    tempone = []
    tempone = [mu[0]*5850, sigma[0]*5850, mu[1]*90, sigma[1]*90, mu[2], sigma[2], mu[3], sigma[3], mu[4], sigma[4],
               data[i,102], data[i,103], data[i,104], data[i,105], maxmin]
    alltemp.append(tempone)
    pre = predict(mu)
        

    plt.clf()
    plt.figure(0)
    ax = plt.gca()
    ax.plot(phase, meanmagnoise, '.')
    ax.plot(phase, pre, c='r')
#    ax.plot(sx1, sy1, c='r')
#    ax.plot(sx1, modellight, c='b')
    plt.xlabel('phase',fontsize=18)
    plt.ylabel('mag',fontsize=18)
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向
    plt.pause(0.0001)
    
    print('time=',time.time()-t1) #MCMC运行时间

name=['T','Terror','INCL','INCLERROR','Q','QERROR','F','FERROR','T2T1','T2T1ERROR','realincl', 'realq', 'realf', 'realt2t1', 'maxminmag' ]     
test = pd.DataFrame(columns=name,data=alltemp)#数据有三列，列名分别为one,two,three
test.to_csv('mcmcerror20.csv',encoding='gbk')
    
    
    
    