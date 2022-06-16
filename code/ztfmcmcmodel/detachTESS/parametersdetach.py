# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:23:48 2022

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
import corner
import phoebe
from multiprocessing import Pool

#os.environ["OMP_NUM_THREADS"] = "1"

def plotphoebenol3T(padata, times):
    #logger = phoebe.logger('warning')
    times = np.linspace(0,1,201)
    t2t1 = padata[1]
    incl = padata[2]*90
    q = padata[3]
    r1 = padata[4]
    r2 = padata[4]*padata[5]
    ecc = padata[6]
    b = phoebe.default_binary()
    times = times
    b.add_dataset('lc', times=times, passband= 'TESS:T')
    
    b['period@binary'] = 1

    b['incl@binary'] = incl #58.528934
    b['q@binary'] =   q
    b['teff@primary'] = padata[0]*8000#6500#6500  #6208 
    b['teff@secondary'] = padata[0]*8000*t2t1#6500*92.307556*0.01#6500*100.08882*0.01 #6087
    b['requiv@primary'] = r1
    b['requiv@secondary'] = r2
    b['ecc@binary'] = ecc
    
    b['sma@binary'] = 1#0.05 2.32
    
    
    
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


def predict(allpara):
    
    arraymcall = np.array(allpara)
    arraymc = arraymcall[0:7]
    
    if index == 0:
        mcinput = np.reshape(arraymc,(1,7))
        lightdata = model10mc(mcinput)
        
        
    if index == 1:
        mcinput = np.reshape(arraymc,(1,6))
        #lightdata = l3model10mc(mcinput)
         
    return lightdata[0]+arraymcall[8]

def getdata(allpara):
    arraymc = np.array(allpara)
    if index == 0:
        offset = int(arraymc[7])
        dataym = np.hstack((datay[offset:], datay[:offset]))
        
    if index == 1:
        offset = int(arraymc[7])
        dataym = np.hstack((datay[offset:], datay[:offset]))
    
    noisy = np.interp(x,phrase,dataym) #y轴
    
    return noisy


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
    noisy = getdata(z)

    lnp = -0.5*np.sum(np.log(2 * np.pi * sigma ** 2)+(output-noisy)**2/(sigma**2)) #计算似然函数
      
    return lnp


def run(init_dist, nwalkers, niter,nburn):
    
    ndim = len(init_dist)
    # Generate initial guesses for all parameters for all chains
    p0 = [rpars(init_dist) for i in range(nwalkers)] #均匀撒ndim*nwalkers点
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) #建立MCMC模型
    pos, prob, state = sampler.run_mcmc(p0, niter, progress = True) # 撒点
    emcee_trace = sampler.chain[:, -nburn:, :].reshape(-1, ndim).T #保留最后nburn 个点做统计

    return emcee_trace 




mpath = ''
model10mc = load_model(mpath+'model202.hdf5')

path = ''
fileone = 'phasemag.txt'
data = np.loadtxt(path+fileone)

phrase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])
datay = np.hstack((datay[0:], datay[:0]))


x = np.linspace(0, 1, 201) #x轴
noisy = np.interp(x,phrase,datay) #y轴
sigma = np.diff(datay,2).std()/np.sqrt(6) #估计观测噪声值

###########MCMC参数
nwalkers = 60
niter = 1000
nburn = 200 #保留最后多少点用于计算
index = 0

#初始范围[T/8000，T2/T1, INCL/90, Q, R1,R1R2,ECC]
init_dist = [( 0.7679937744137499-0.0001, 0.7679937744137499+0.0001), #T/8000   0.5-1
             (0.5, 1), #T2/T1    0.5-1
             (0.964, 1.0), #INCL/90 0.7-1
             (0.2, 0.6), #q
             (0.0, 0.5),#R1  0.02-0.487
             (0.0, 0.55),#R1R2   0-1.5
             (0, 0.01),#ECC   0-0.1
             (-20, 20),
             (-0.1, 0.1)
             ]

priors = init_dist.copy()
ndim = len(priors) #维度数

emcee_trace  = run(priors, nwalkers, niter,nburn) #run mcmc


mu = []
sigma_1 = []
sigma_2 = []
    
for mi, x1 in enumerate(emcee_trace):
    q_16, q_50, q_84 = quantile(x1, [0.16, 0.5, 0.84])          
    q_m, q_p = q_50 - q_16, q_84 - q_50
  
    mu.append(q_50) #median value
    sigma_1.append(q_m) #high limitation
    sigma_2.append(q_p) #low
 
sigma_1 = np.array(sigma_1)
sigma_2 = np.array(sigma_2)


if index == 0:
    figure = corner.corner(emcee_trace.T[:,:],bins=100,labels=[r"$T1$",  r"$T2T1$", r"$incl$",  r"$q$",  r"$r1$", r"$r1r2$", r"$ecc$" ,r"$offset1$", r"$offset2$"],
                       label_kwargs={"fontsize": 15},title_fmt='.4f',show_titles=True, title_kwargs={"fontsize": 15}, color ='blue')

plt.savefig('corner.png')

times,resultflux, pbdic, pr1, pr2 = plotphoebenol3T(mu, x)
if index == 0:
    offset = int(mu[7])
else :
    offset = int(mu[8])

datay = np.hstack((datay[offset:], datay[:offset])) 


pre=predict(mu)
plt.figure()
ax = plt.gca()
#ax.plot(x,noisy,'.') #原始数据
ax.plot(phrase, datay, '.', c = 'b')
#ax.plot(x,pre.flatten(),'-r') #理论数据
if index == 0:
    ax.plot(times, resultflux+mu[8], marker='x', c='g', markersize = 8) #理论数据
else:
    ax.plot(times, resultflux+mu[9], marker='x', c='g', markersize = 8) #理论数据
    
ax.plot(x, pre,'-r') #理论数据
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)
