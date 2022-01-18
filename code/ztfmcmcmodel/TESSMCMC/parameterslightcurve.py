# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:51:36 2021

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
#from keras.models import load_model
import pandas as pd
import emcee
import corner

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\model\\'
model10mc = load_model(path+'model10mc.hdf5')
l3model10mc = load_model(path+'model10l3mc.hdf5')

#path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
#file = 'KIC 11924311.txt'
#data = np.loadtxt(path+file)

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\EWDATA\\'
file = 'TIC 55896456.txt'
data = np.loadtxt(path+file)

#fileone = 'ZTFtestdata.txt'
#data = np.loadtxt(fileone)

phrase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])
x = np.linspace(0,1,100) #x轴
noisy = np.interp(x,phrase,datay) #y轴
sigma = np.diff(noisy,2).std()/np.sqrt(6) #估计观测噪声值
#sigma=1

###########MCMC参数
nwalkers = 20
niter = 500
nburn=200 #保留最后多少点用于计算
index = 0

#初始范围[T，incl,q,f,t2t1,l3]
init_dist = [(0.9111384615384616, 1.3667076923076922),
             (0.7113870578342014, 1.067080586751302),
             (0.5539359130859375, 0.8309038696289063),
             (0.5249301147460937, 0.7873951721191407),
             (0.7962756958007813, 1.1944135437011718)]

priors=init_dist.copy()
ndim = len(priors) #维度数

def predict(allpara):
    
    arraymc = np.array(allpara)
    #print(arraymc)
    
    if index == 0:
        mcinput = np.reshape(arraymc,(1,5))
        #lightdata = model10mc.predict(mcinput)
        lightdata = model10mc(mcinput)
        
    if index == 1:
        mcinput = np.reshape(arraymc,(1,6))
        #lightdata = l3model10mc.predict(mcinput)
        lightdata = l3model10mc(mcinput)
        
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
         
    print('it is ok')

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

t1=time.time()
emcee_trace  = run(priors, nwalkers, niter,nburn) #run mcmc
print('time=',time.time()-t1) #MCMC运行时间


mu=(emcee_trace.mean(axis=1)) #参数均值
sigma=(emcee_trace.std(axis=1)) #参数误差
print('mu=',mu)
print('sigma=',sigma)


####################绘图
if index == 1:
    figure = corner.corner(emcee_trace.T,bins=100,labels=[r"$Tem$", r"$incl$", r"$q$", r"$f_0$", r"$t2t1$", r"$l3$"],
                       label_kwargs={"fontsize": 15},show_titles=True, title_kwargs={"fontsize": 15}, color ='blue')

if index == 0:
    figure = corner.corner(emcee_trace.T,bins=100,labels=[r"$Tem$", r"$incl$", r"$q$", r"$f_0$", r"$t2t1$"],
                       label_kwargs={"fontsize": 15},show_titles=True, title_kwargs={"fontsize": 15}, color ='blue')
    
plt.savefig('corner.png')
#------------------------------------------------------------
#用输出值预测理论曲线
pre = predict(mu.reshape(1,-1))
plt.figure()
ax = plt.gca()
#ax.plot(x,noisy,'.') #原始数据
ax.plot(phrase, datay, '.')
#ax.plot(x,pre.flatten(),'-r') #理论数据
ax.plot(x,pre,'-r')
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
