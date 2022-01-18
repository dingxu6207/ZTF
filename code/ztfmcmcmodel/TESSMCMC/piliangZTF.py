# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
import phoebe
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
mpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\model\\'
inclmodel = load_model(mpath+'incl.hdf5')
model1 = load_model(mpath+'model1.hdf5')
model10 = load_model(mpath+'model10.hdf5')
l3model1 = load_model(mpath+'model1l3.hdf5')
l3model10 = load_model(mpath+'model10l3.hdf5')
model10mc = load_model(mpath+'model10mc.hdf5')
l3model10mc = load_model(mpath+'model10l3mc.hdf5')
#predicT = load_model('modelT.hdf5')


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
    #predata = data[0]
    temp.append(T1)
    temp.append(predata[0]/90) #incl
    temp.append(predata[1]/100)#q
    temp.append(predata[2]/100)
    temp.append(predata[3]/100)
    
    arraymc = np.array(temp)
    mcinput = np.reshape(arraymc,(1,5))
    lightdata = model10mc.predict(mcinput)
    #lightdata = model10mc(mcinput)
    stdr, r_squared = calculater(sy1, lightdata[0])
  
    return lightdata, stdr, r_squared  


def model10R2(data, sy1, T1):
    temp = []
    predata = data[0].tolist()
    #predata = data[0]
    temp.append(T1)
    temp.append(predata[0]/90)#incl
    temp.append(predata[1]/10)#q
    temp.append(predata[2]/100)
    temp.append(predata[3]/100)
    
    arraymc = np.array(temp)
    mcinput = np.reshape(arraymc,(1,5))
    lightdata = model10mc.predict(mcinput)
    #lightdata = model10mc(mcinput)
    stdr, r_squared = calculater(sy1, lightdata[0])
    
    return lightdata, stdr, r_squared  

def model1l3R2(data, sy1, T1):
    temp = []
    predata = data[0].tolist()
    #predata = data[0]
    temp.append(T1)
    temp.append(predata[0]/90) #incl
    temp.append(predata[1]/100)#q
    temp.append(predata[2]/100)
    temp.append(predata[3]/100)
    temp.append(predata[4]/100)
    
    arraymc = np.array(temp)
    mcinput = np.reshape(arraymc,(1,6))
    lightdata = l3model10mc.predict(mcinput)
    #lightdata = l3model10mc(mcinput)
    stdr, r_squared = calculater(sy1, lightdata[0])
    
    return lightdata, stdr, r_squared  

def model10l3R2(data, sy1, T1):
    temp = []
    predata = data[0].tolist()
    #predata = data[0]
    temp.append(T1)
    temp.append(predata[0]/90) #incl
    temp.append(predata[1]/10)#q
    temp.append(predata[2]/100)
    temp.append(predata[3]/100)
    temp.append(predata[4]/100)
    
    arraymc = np.array(temp)
    mcinput = np.reshape(arraymc,(1,6))
    lightdata = l3model10mc.predict(mcinput)
    #lightdata = l3model10mc(mcinput)
    stdr, r_squared = calculater(sy1, lightdata[0])
    
    return lightdata, stdr, r_squared  

def tupledata(data, T1, flag):
    temp = []
    predata = data[0].tolist()
    #predata = data[0]
    tup1 = (T1*0.9, 1.1*T1)
    tup2 = ((predata[0]/90)*0.9, (predata[0]/90)*1.1)

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
    datax = np.sort(datax)
    func1 = interpolate.UnivariateSpline(datax, interdata,s=s*s*num)#强制通过所有点
    sy1 = func1(sx1)
    
    return sx1,sy1

def inclprediction(data):
    inputdata  = np.copy(data)
    nparraydata = np.reshape(inputdata,(1,100))
    incldata = inclmodel.predict(nparraydata)
    #incldata = inclmodel(nparraydata)
    inclcom = incldata[0][0]
    return inclcom
    
def dataaddT(data, T1):
    inputdata = np.copy(data)
    listsys = inputdata.tolist()
    listsys.append(T1)
    npsy1 = np.array(listsys)
    nparraydata = np.reshape(npsy1,(1,101))
    return nparraydata

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
        #lightdata = model10mc.predict(mcinput)
        lightdata = model10mc(mcinput)
        
    if modelindex == 1:
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
            b.run_compute(ntriangles = 10000)
            
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
            b.run_compute(ntriangles = 10000)
            
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

alltempdata = []
CSV_FILE_PATH = 'EMTEMP.csv'
df = pd.read_csv(CSV_FILE_PATH)
hang,lie = df.shape
path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\EWDATA\\'
figurepath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\figure\\'
for i  in range(0,hang):
    name = df.iloc[i,1]
    RA, DEC, P, prob = df.iloc[i,2], df.iloc[i,3], df.iloc[i,4], df.iloc[i,5]
    file = name+'.txt'
    inputtemper = df.iloc[i,6]
    if inputtemper == 0:
        inputtemper = 5850
    print(file, inputtemper)
    
    data = np.loadtxt(path+file)
    phase = data[:,0]
    datay = data[:,1]-np.mean(data[:,1])

    sx1,sy1 = interone(phase, datay)
    inclcom = inclprediction(sy1)

    T1 =  inputtemper/5850
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
    
        R = [r_squared1, r_squared10, r_squared1l3, r_squared10l3]
        index = np.argmax(R)
        print('index= '+str(index)+'  R2='+str(R[index]))
        
        if R[index] < 0.9:
            continue
    
        plt.figure(1)
        ax = plt.gca()
        ax.plot(x, noisy, '.', c='b')
        #ax.plot(phase, datay, '.', c='b')
        
        if index == 0 :    
            temppre = tupledata(predict1, T1, 0)
            modelindex = 0
            ax.plot(sx1, ligdata1[0],'-m')
            
        if index == 1 :    
            temppre = tupledata(predict10, T1, 1)
            modelindex = 0
            ax.plot(sx1, ligdata10[0],'-m')
            
        if index == 2 : 
            temppre = tupledata(l3predict1, T1, 2)
            modelindex = 1
            ax.plot(sx1, ligdata1l3[0],'-m')
            
        if index == 3 :   
            temppre = tupledata(l3predict10, T1, 3)
            modelindex = 1
            ax.plot(sx1, ligdata10l3[0],'-m')
  
    
        print(temppre)
    
   ##################MCMC#########################
    #初始范围[T，incl,q,f,t2t1,l3]
        init_dist = temppre.copy()
        priors=init_dist.copy()
        ndim = len(priors) #维度数   
        emcee_trace  = run(priors, nwalkers, niter,nburn) #run mcmc
        mu=(emcee_trace.mean(axis=1)) #参数均值
        sigma=(emcee_trace.std(axis=1)) #参数误差
        print('mu=',mu)
        print('sigma=',sigma)
        
    ##################Phoebe###################
        
        try:
            if modelindex == 0:
                times,resultflux, pbdic, pr1, pr2 = plotphoebenol3T(mu, x)
                
            if modelindex == 1:
                times,resultflux, pbdic, pr1, pr2 = plotphoebel3T(mu,x)
                
        except:
            pbdic, pr1, pr2 = 0, 0 ,0
            print('phoebe is error!')
    ##############################################
        doflag = 0
        if not((times == resultflux).all()):
            ax.plot(times, resultflux, '*', c='g') #理论数据
            doflag = 1
            #ax.scatter(times, resultflux, marker='o', c='', edgecolors='g')
            
        pre = predict(mu.reshape(1,-1))
        #ax.plot(x, pre.flatten(),'-r') #理论数据
        ax.plot(x, pre,'-r') #理论数据
        ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
        ax.invert_yaxis() #y轴反向 
        plt.title(name)
        plt.savefig(figurepath+name+'.jpg') #修改2处
        plt.pause(1)
        plt.clf()
        
        if doflag ==1 :
            stdflag = np.std(resultflux)/np.std(pre)
            stdcancha = np.std(resultflux-noisy)
        else:
            stdflag = 0 
            stdcancha = 0
            
        print('stdflag is： '+str(stdflag))
        
        tempalldata = []
        if modelindex == 0:
            tempalldata = [name,  RA, DEC, P, prob, mu[0]*5850, sigma[0]*5850, mu[1]*90, sigma[1]*90, mu[2], sigma[2], mu[3], sigma[3], mu[4], sigma[4], 0, 0, inputtemper, R[index], pbdic, pr1, pr2, stdflag, stdcancha]
        if modelindex == 1:
            tempalldata = [name,  RA, DEC, P, prob, mu[0]*5850, sigma[0]*5850, mu[1]*90, sigma[1]*90, mu[2], sigma[2], mu[3], sigma[3], mu[4], sigma[4], mu[5], sigma[5], inputtemper, R[index], pbdic, pr1, pr2, stdflag, stdcancha]
        alltempdata.append(tempalldata)
    
test = pd.DataFrame(alltempdata) 
test.to_csv('parameter.csv', encoding='gbk',header=0) #修改3处        


