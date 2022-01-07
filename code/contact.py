# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 21:38:18 2022

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,200)
b.add_dataset('lc', compute_times=times, passband= 'LSST:r')

b['period@binary'] = 1

b['incl@binary'] =  50   #58.528934
b['q@binary'] =    0.2
b['teff@primary'] =  5850  #6208 
b['teff@secondary'] = 5600#6500*100.08882*0.01 #6087

b['sma@binary'] = 3#0.05 2.32
b.flip_constraint('pot', solve_for='requiv@primary')
b.flip_constraint('fillout_factor', solve_for='pot')
b['fillout_factor'] = 0.3

b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(irrad_method='none')

plt.figure(0)
afig, mplfig = b.plot(show=True, legend=True)

print(b['fillout_factor@contact_envelope'])

np.savetxt('data0.lc', np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T)

fluxmodel = b['value@fluxes@lc01@model']
resultflux = -2.5*np.log10(fluxmodel)
resultflux = resultflux - np.mean(resultflux)
plt.figure(1)
plt.scatter(b['value@times@lc01@model'], resultflux, c='none',marker='o',edgecolors='r', s=80)
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
np.savetxt('datamag.txt', np.vstack((b['value@times@lc01@model'], resultflux)).T)