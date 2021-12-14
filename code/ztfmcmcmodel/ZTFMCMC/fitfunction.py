# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:14:01 2021

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

def plotphoebe(predata,times):
    logger = phoebe.logger('warning')
    incl = predata[0]
    q = predata[1]
    r = predata[2]
    t2t1 = predata[3]
    b = phoebe.default_binary(contact_binary=True)
    #times  = np.linspace(0,1,150)
    times = times
    #b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
    b.add_dataset('lc', times=times, passband= 'ZTF:r')

    b['period@binary'] = 1

    b['incl@binary'] = incl #58.528934
    b['q@binary'] =   q
    b['teff@primary'] = 6500#6500#6500  #6208 
    b['teff@secondary'] = 6500*t2t1#6500*92.307556*0.01#6500*100.08882*0.01 #6087

    b['sma@binary'] = 1#0.05 2.32

    b['requiv@primary'] = r    #0.61845703
    

    #b.add_dataset('mesh', times=[0.25], dataset='mesh01')
    try:
        b.run_compute(irrad_method='none')

        fluxmodel = b['value@fluxes@lc01@model']
        resultflux = -2.5*np.log10(fluxmodel)
        resultflux = resultflux - np.mean(resultflux)
        r2 = b['value@requiv@secondary@component']
        f = b['value@fillout_factor@contact_envelope@envelope@component']
        #plt.figure(0)
        #plt.plot(b['value@times@lc01@model'], resultflux, '.')
        return times,resultflux,r2,f
    except:
        return times,times,0,0
    

def plotphoebenol3(predata,times):
    logger = phoebe.logger('warning')
    incl = predata[0]
    q = predata[1]
    f = predata[2]
    t2t1 = predata[3]
    b = phoebe.default_binary(contact_binary=True)
    times = times
    b.add_dataset('lc', times=times, passband= 'ZTF:r')

    b['period@binary'] = 1

    b['incl@binary'] = incl #58.528934
    b['q@binary'] =   q
    b['teff@primary'] = 6500#6500#6500  #6208 
    b['teff@secondary'] = 6500*t2t1#6500*92.307556*0.01#6500*100.08882*0.01 #6087

    b['sma@binary'] = 1#0.05 2.32
    
    b.flip_constraint('pot', solve_for='requiv@primary')
    b.flip_constraint('fillout_factor', solve_for='pot')
    b['fillout_factor'] = f    #0.61845703
    
    try:
        b.run_compute(irrad_method='none')

        fluxmodel = b['value@fluxes@lc01@model']
        resultflux = -2.5*np.log10(fluxmodel)
        resultflux = resultflux - np.mean(resultflux)
        return times,resultflux
    except:
        return times,times 
 
def plotphoebel3(predata,times):
    logger = phoebe.logger('warning')
    incl = predata[0]
    q = predata[1]
    f = predata[2]
    t2t1 = predata[3]
    l3fra = predata[4]
    b = phoebe.default_binary(contact_binary=True)
    times = times
    b.add_dataset('lc', times=times, passband= 'ZTF:r')
    b.set_value('l3_mode', 'fraction')
    b['period@binary'] = 1

    b['incl@binary'] = incl #58.528934
    b['q@binary'] =   q
    b['teff@primary'] = 6500#6500#6500  #6208 
    b['teff@secondary'] = 6500*t2t1#6500*92.307556*0.01#6500*100.08882*0.01 #6087
    b.set_value('l3_frac', l3fra)
    b['sma@binary'] = 1#0.05 2.32
    
    b.flip_constraint('pot', solve_for='requiv@primary')
    b.flip_constraint('fillout_factor', solve_for='pot')
    b['fillout_factor'] = f    #0.61845703
    
    try:
        b.run_compute(irrad_method='none')

        fluxmodel = b['value@fluxes@lc01@model']
        resultflux = -2.5*np.log10(fluxmodel)
        resultflux = resultflux - np.mean(resultflux)
        return times,resultflux
    except:
        return times,times 

####################################################################
def plotphoebenol3T(predata,times,T1):
    logger = phoebe.logger('warning')
    incl = predata[0]
    q = predata[1]
    f = predata[2]
    t2t1 = predata[3]
    b = phoebe.default_binary(contact_binary=True)
    times = times
    b.add_dataset('lc', times=times, passband= 'LSST:r')

    b['period@binary'] = 1

    b['incl@binary'] = incl #58.528934
    b['q@binary'] =   q
    b['teff@primary'] = T1*5850#6500#6500  #6208 
    b['teff@secondary'] = T1*5850*t2t1#6500*92.307556*0.01#6500*100.08882*0.01 #6087

    b['sma@binary'] = 1#0.05 2.32
    
    b.flip_constraint('pot', solve_for='requiv@primary')
    b.flip_constraint('fillout_factor', solve_for='pot')
    b['fillout_factor'] = f    #0.61845703
    
    try:
        b.run_compute(irrad_method='none')
        print('it is ok')
        lumidata = b.compute_pblums()
        pbdic = np.float(lumidata['pblum@secondary@lc01']/lumidata['pblum@primary@lc01'])
        pr1 = b['requiv@primary']
        pr2 = b['requiv@secondary']
        
        fluxmodel = b['value@fluxes@lc01@model']
        resultflux = -2.5*np.log10(fluxmodel)
        resultflux = resultflux - np.mean(resultflux)
        return times,resultflux, pbdic, pr1, pr2
        #return times,resultflux, 0, 0, 0
    except:
        return times, times, 0, 0, 0          
    
def plotphoebel3T(predata,times, T1):
    logger = phoebe.logger('warning')
    incl = predata[0]
    q = predata[1]
    f = predata[2]
    t2t1 = predata[3]
    l3fra = predata[4]
    b = phoebe.default_binary(contact_binary=True)
    times = times
    b.add_dataset('lc', times=times, passband= 'LSST:r')
    b.set_value('l3_mode', 'fraction')
    b['period@binary'] = 1

    b['incl@binary'] = incl #58.528934
    b['q@binary'] =   q
    b['teff@primary'] = T1*5850#6500#6500  #6208 
    b['teff@secondary'] = T1*5850*t2t1#6500*92.307556*0.01#6500*100.08882*0.01 #6087
    b.set_value('l3_frac', l3fra)
    b['sma@binary'] = 1#0.05 2.32
    
    b.flip_constraint('pot', solve_for='requiv@primary')
    b.flip_constraint('fillout_factor', solve_for='pot')
    b['fillout_factor'] = f    #0.61845703
    
    try:
        b.run_compute(irrad_method='none')
        
        lumidata = b.compute_pblums()
        pbdic = np.float(lumidata['pblum@secondary@lc01']/lumidata['pblum@primary@lc01'])
        pr1 = b['requiv@primary']
        pr2 = b['requiv@secondary']

        fluxmodel = b['value@fluxes@lc01@model']
        resultflux = -2.5*np.log10(fluxmodel)
        resultflux = resultflux - np.mean(resultflux)
        return times,resultflux, pbdic, pr1, pr2
    except:
        return times,times, 0, 0, 0