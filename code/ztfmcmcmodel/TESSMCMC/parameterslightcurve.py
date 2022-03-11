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
import phoebe


############################################################################
############################################################################
def plotphoebenol3T(padata, times):
    #logger = phoebe.logger('warning')
    incl = padata[1]*90
    q = padata[2]
    f = padata[3]
    t2t1 = padata[4]
    b = phoebe.default_binary(contact_binary=True)
    times = times
    b.add_dataset('lc', times=times, passband= 'TESS:T')
    
    b['period@binary'] = 1

    b['incl@binary'] = incl #58.528934
    b['q@binary'] =   q
    b['teff@primary'] = padata[0]*5850#6500#6500  #6208 
    b['teff@secondary'] = padata[0]*5850*t2t1#6500*92.307556*0.01#6500*100.08882*0.01 #6087

    b['sma@binary'] = 1#0.05 2.32
    
    b.flip_constraint('pot', solve_for='requiv@primary')
    b.flip_constraint('fillout_factor', solve_for='pot')
    b['fillout_factor'] = f    #0.61845703
    
    try:
        try:
            b.run_compute(irrad_method='none')
        except:
            b.run_compute(ntriangles = 8000)
            
        try:
            lumidata = b.compute_pblums()
            pbdic = np.float64(lumidata['pblum@secondary@lc01']/lumidata['pblum@primary@lc01'])
        except:
            pbdic = 0
            
        print('it is ok')
        
        pr1 = b['value@requiv@primary@component']
        pr2 = b['value@requiv@secondary@component']
        
        fluxmodel = b['value@fluxes@lc01@model']
        resultflux = -2.5*np.log10(fluxmodel)
        resultflux = resultflux - np.mean(resultflux)
        return times,resultflux, pbdic, pr1, pr2
        #return times,resultflux, 0, 0, 0
    except:
        return times, times, 0, 0, 0          
    
def plotphoebel3T(padata,times):
    #logger = phoebe.logger('warning')  
    incl = padata[1]*90
    q = padata[2]
    f = padata[3]
    t2t1 = padata[4]
    l3fra = padata[5]
    
    b = phoebe.default_binary(contact_binary=True)
    times = times
    b.add_dataset('lc', times=times, passband= 'TESS:T')
    b.set_value('l3_mode', 'fraction')
    b['period@binary'] = 1

    b['incl@binary'] = incl #58.528934
    b['q@binary'] =   q
    b['teff@primary'] = padata[0]*5850#6500#6500  #6208 
    b['teff@secondary'] = padata[0]*5850*t2t1#6500*92.307556*0.01#6500*100.08882*0.01 #6087
    b.set_value('l3_frac', l3fra)
    b['sma@binary'] = 1#0.05 2.32
    
    b.flip_constraint('pot', solve_for='requiv@primary')
    b.flip_constraint('fillout_factor', solve_for='pot')
    b['fillout_factor'] = f    #0.61845703
    
    try:
        try:
            b.run_compute(irrad_method='none')
        except:
            b.run_compute(ntriangles = 8000)
            
        try:
            lumidata = b.compute_pblums()
            pbdic = np.float64(lumidata['pblum@secondary@lc01']/lumidata['pblum@primary@lc01'])
        except:
            pbdic = 0
            
        print('it is ok')
        pr1 = b['value@requiv@primary@component']
        pr2 = b['value@requiv@secondary@component']

        fluxmodel = b['value@fluxes@lc01@model']
        resultflux = -2.5*np.log10(fluxmodel)
        resultflux = resultflux - np.mean(resultflux)
        return times,resultflux, pbdic, pr1, pr2
    except:
        return times,times, 0, 0, 0
#######################################################3    

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\model\\'
model10mc = load_model(path+'model10mc.hdf5')
l3model10mc = load_model(path+'model10l3mc.hdf5')

#path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
#file = 'KIC 11924311.txt'
#data = np.loadtxt(path+file)

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\EWDATA\\'
file = 'TIC 1981434196.txt'
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
index = 1

#初始范围[T，incl,q,f,t2t1,l3] T/5850 incl/90 
init_dist = [(1.02+0.0001, 1.02+0.0002), 
             (0.74, 0.8),
             (1.75, 2.6), 
             (0.6, 0.80),
             (0.91, 0.93),
             (0.16,0.28)
            ]

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

if index == 0:
    times,resultflux, pbdic, pr1, pr2 = plotphoebenol3T(mu, x)
else:
    times,resultflux, pbdic, pr1, pr2 = plotphoebel3T(mu,x)

####################绘图
if index == 1:
    figure = corner.corner(emcee_trace.T[:,1:],bins=100,labels=[r"$incl$", r"$q$", r"$f_0$", r"$t2t1$", r"$l3$"],
                       label_kwargs={"fontsize": 15}, show_titles=True, title_fmt='.3f', 
                       title_kwargs={"fontsize": 15}, color ='blue')

if index == 0:
    figure = corner.corner(emcee_trace.T[:,1:],bins=100,labels=[r"$incl$", r"$q$", r"$f_0$", r"$t2t1$"],
                       label_kwargs={"fontsize": 15},show_titles=True, title_fmt='.3f',
                       title_kwargs={"fontsize": 15}, color ='blue')
    
plt.savefig('corner.png')
#------------------------------------------------------------
#用输出值预测理论曲线
pre = predict(mu.reshape(1,-1))
plt.figure()
ax = plt.gca()
#ax.plot(x,noisy,'.') #原始数据
ax.plot(phrase, datay, '.')
ax.plot(times, resultflux, '*', c='g') #理论数据
ax.plot(x,pre,'-r')
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.title(file[:-4])