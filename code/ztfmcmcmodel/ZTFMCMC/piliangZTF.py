# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:27:28 2021

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
import phoebe

inclmodel = load_model('incl.hdf5')
model1 = load_model('model1.hdf5')
model10 = load_model('model10.hdf5')
l3model1 = load_model('model1l3.hdf5')
l3model10 = load_model('model10l3.hdf5')
model10mc = load_model('model10mc.hdf5')
l3model10mc = load_model('model10l3mc.hdf5')
predicT = load_model('modelT.hdf5')

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
    
    tup4 = ((predata[2]/100)*0.8, (predata[2]/100)*1.2)
    tup5 = ((predata[3]/100)*0.8, (predata[3]/100)*1.2)
    
    if flag==2 or flag==3:
        tup6 = ((predata[4]/100)*0.7, (predata[4]/100)*1.2)
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
    func1 = interpolate.UnivariateSpline(datax, interdata,s=s*s*num)#?????????????????????
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
    x = np.linspace(0,1,100) #x???
    noisy = np.interp(x, phase, inputdata) #y???
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


def rpars(init_dist):#???ndim ?????????????????????????????????????????????ndim??????
    return [np.random.rand() * (i[1]-i[0]) + i[0] for i in init_dist] 


def lnprior(priors, values):#??????MCMC???????????????????????????????????????
    
    lp = 0.
    for value, prior in zip(values, priors):
        if value >= prior[0] and value <= prior[1]:
            lp+=0
        else:
            lp+=-np.inf 
    return lp


def lnprob(z): #??????????????????
    
    lnp = lnprior(priors,z)#??????MCMC???????????????????????????????????????

    if not np.isfinite(lnp):
            return -np.inf


    output = predict(z)

    lnp = -0.5*np.sum(np.log(2 * np.pi * sigma ** 2)+(output-noisy)**2/(sigma**2)) #??????????????????
      
    return lnp


def run(init_dist, nwalkers, niter,nburn):
    
    ndim = len(init_dist)
    # Generate initial guesses for all parameters for all chains
    p0 = [rpars(init_dist) for i in range(nwalkers)] #?????????ndim*nwalkers???
 #   print(p0)

    # Generate the emcee sampler. Here the inputs provided include the 
    # lnprob function. With this setup, the first parameter
    # in the lnprob function is the output from the sampler (the paramter 
    # positions).
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob) #??????MCMC??????
    pos, prob, state = sampler.run_mcmc(p0, niter) # ??????
    emcee_trace = sampler.chain[:, -nburn:, :].reshape(-1, ndim).T #????????????nburn ???????????????

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
    b.add_dataset('lc', times=times, passband= 'LSST:r')

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
            lumidata = b.compute_pblums()
            pbdic = np.float(lumidata['pblum@secondary@lc01']/lumidata['pblum@primary@lc01'])
        except:
            b.run_compute(ntriangles=6000)
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
    b.add_dataset('lc', times=times, passband= 'LSST:r')
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
            lumidata = b.compute_pblums()
            pbdic = np.float(lumidata['pblum@secondary@lc01']/lumidata['pblum@primary@lc01'])
        except:
            b.run_compute(ntriangles=6000)
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


############################################################################
############################################################################    
filepkl = 'Z:/DingXu/ZTF_jkf/alldata/0002.pkl'
dat = pickle.load(open(filepkl,'rb'))
tot = len(dat)
def loaddata(i):
    idname = dat[i][0]
    name = dat[i][1]
    RA = dat[i][2]
    DEC = dat[i][3]
    P = dat[i][4]
    gmag = dat[i][5]
    rmag = dat[i][6]
    xy = dat[i][7] 
    xy[:,1] = xy[:,1] -np.mean(xy[:,1])
    return name, idname, RA, DEC, P, gmag, rmag, xy[:,0], xy[:,1]
    
def predicttemperature(gmag, ramg):
    if (gmag != 0) and (gmag != 0):
        magrg = [gmag, ramg]
        npmagrg = np.array(magrg)
        inputmag = np.reshape(npmagrg, [1,2])
        temparaturein = predicT.predict(inputmag)
        temparaturein = temparaturein[0][0]
    else:
        temparaturein =  5000
        
    return temparaturein
#########################################################################
#######################################################################


for i in range(0, tot):
    name, idname, RA, DEC, P, gmag, rmag, phase, datay = loaddata(i)
    try:
        sx1,sy1 = interone(phase, datay)
    except:
        continue
    inclcom = inclprediction(sy1)
    #inputtemper = 5786
    inputtemper = predicttemperature(gmag, rmag)
    print('temperature is : '+str(inputtemper))
    
    T1 = inputtemper/5850
    
    
    nparraydata = dataaddT(sy1, T1)
    
    ###############MCMC#######################
    x, noisy, sigma = intertwo(phase, datay)
    nwalkers = 20
    niter = 500
    nburn=200 #?????????????????????????????????
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
    
   ##################MCMC#########################
    #????????????[T???incl,q,f,t2t1,l3]
        init_dist = temppre.copy()
        priors=init_dist.copy()
        ndim = len(priors) #?????????   
        emcee_trace  = run(priors, nwalkers, niter,nburn) #run mcmc
        mu=(emcee_trace.mean(axis=1)) #????????????
        sigma=(emcee_trace.std(axis=1)) #????????????
        print('mu=',mu)
        print('sigma=',sigma)
    
    ##################Phoebe###################
        try:
            if modelindex == 0:
                times,resultflux, pbdic, pr1, pr2 = plotphoebenol3T(mu, x)
        
            if modelindex == 1:
                times,resultflux, pbdic, pr1, pr2 = plotphoebel3T(mu,x)
        
            if pr1 != 0:
                ax.plot(times, resultflux,'-g') #????????????
        except:
            print('phoebe is error!')
    ##############################################
        pre = predict(mu.reshape(1,-1))
        ax.plot(x, pre.flatten(),'-r') #????????????
        ax.yaxis.set_ticks_position('left') #???y???????????????????????????
        ax.invert_yaxis() #y????????? 
        plt.pause(1)
        plt.clf()
    








    
