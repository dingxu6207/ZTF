# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 22:06:35 2021

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,100)

#b.add_dataset('lc', times=phoebe.linspace(0,1,150), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', compute_times=phoebe.linspace(0,1,100),passband= 'LSST:r')

b['period@binary'] = 1

b['incl@binary'] =  90   #58.528934
b['q@binary'] =    0.8
b['teff@primary'] =  5800  #6208 
b['teff@secondary'] = 5800#6500*100.08882*0.01 #6087
b.set_value('l3_mode', 'fraction')
l3fra = 0.2
b.set_value('l3_frac', l3fra) 
b['sma@binary'] = 1#0.05 2.32
b.flip_constraint('pot', solve_for='requiv@primary')
b.flip_constraint('fillout_factor', solve_for='pot')
b['fillout_factor'] = 0.5
b.set_value('ecc', component='binary', value=0)
#b.add_feature('spot', component='secondary', feature='spot01', relteff=0.9, radius=20, colat=90, long=180)

#b.add_dataset('mesh', times=[0.25], dataset='mesh01')
b.add_dataset('mesh', compute_times=b.to_time(0.25), columns='teffs')

#b.run_compute(ntriangles=5000)
b.run_compute(irrad_method='none')

#afig, mplfig = b.plot(show=True, legend=True)
afig, mplfig = b.plot(fc='teffs', ec='face', fcmap='plasma', show=True)