#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:33:12 2022

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt
import random

logger = phoebe.logger(clevel = 'WARNING')
b = phoebe.default_binary()

times  = np.linspace(0,1,100)
b.add_dataset('lc', times=times, passband= 'TESS:T')
b['period@binary'] = 1
b['sma@orbit'] = 1
m = 0

for count in range(0,100000000):
    try:
        incl = random.uniform(70,90)
        T1 = np.random.randint(4000, 8000)
        T1divT2 = random.uniform(0.5,1)
        q = np.random.uniform(0, 1)
        r1 = random.uniform(0.02,0.487)
        r1r2 = random.uniform(0,1.5)
        ecc = random.uniform(0,0.1)
        
        print('incl=', incl)
        print('temp1=', T1)
        print('temp2=', T1*T1divT2)
        print('r1=', r1)
        print('r2=', r1*r1r2)
        print('ecc=', ecc)
        print('count = ', count)
        print('q = ', q)
        
        
        b['incl@binary'] = incl
        b['q@binary'] = q
        b['teff@primary'] = T1
        b['teff@secondary'] = T1*T1divT2
        b['requiv@primary'] = r1
        if r1*r1r2<0.02:
            continue
        else:
            b['requiv@secondary'] = r1*r1r2
        b['ecc@binary'] = ecc
        
        b.run_compute(irrad_method='none')
        print('it is ok1')
    
        m = m+1
        print('m = ', m)
        file = str(m)+'.lc'
        lightcurvedata = np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T
        mq = [(T1/8000,T1divT2), (incl/90, q), (r1, r1r2), (ecc,0)]
        datamq = np.array(mq)
        print('it is ok2')
    
        resultdata = np.row_stack((lightcurvedata, datamq))
        np.savetxt(file, resultdata)
        
        print('it is ok3')
        
        #plt.plot(b['value@times@lc01@model'], b['value@fluxes@lc01@model'])
        plt.clf()
        plt.figure(1)
        ax = plt.gca()
        ax.plot(b['value@times@lc01@model'], b['value@fluxes@lc01@model'])
        plt.pause(0.1)
    except:
         print('it is error!')