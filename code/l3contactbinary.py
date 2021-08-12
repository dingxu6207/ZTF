# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 06:37:29 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

lengthd = 100
times  = np.linspace(0,1,lengthd)

#b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', times=times, passband= 'LSST:r')
b.set_value('l3_mode', 'fraction')

constfrac = 84.3345*0.01
#b.set_value('l3_frac', constfrac)
b['l3_frac']= constfrac
b['period@binary'] = 1

b['incl@binary'] = 75.81017     #58.528934
b['q@binary'] =    72.73504*0.01
b['teff@primary'] =  6500#6500#6500#6500  #6208 
b['teff@secondary'] = 6500*96.543274*0.01#6500*0.9069584       

#b['fillout_factor@contact_envelope@envelope@component'] = 0.5

b['sma@binary'] = 1#0.05 2.32
#print(b['sma@binary'])

#b['requiv@primary'] = 0.33427894 #0.61845703
b.flip_constraint('pot', solve_for='requiv@primary')
b.flip_constraint('fillout_factor', solve_for='pot')
b['fillout_factor'] = 11.859107*0.01

b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(irrad_method='none')

plt.figure(0)
afig, mplfig = b.plot(show=True, legend=True)

print(b['fillout_factor@contact_envelope'])



np.savetxt('data0.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T)


fluxes_model = b['fluxes@model'].interp_value(times=times)
fluxcha = fluxes_model-b['value@times@lc01@model']

#print(fluxcha)

path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
#file = 'ztf1.txt' #6677225
file = 'KIC 11769739.txt'
#yuandata = np.loadtxt(file)
yuandata = np.loadtxt(path+file)
#datay = 10**(yuandata[:,1]/(-2.512))
datay = yuandata[:,1]
#datay = -2.5*np.log10(yuandata[:,1])
datay = datay-np.mean(datay)

#datay = datay/np.mean(datay)

fluxmodel = b['value@fluxes@lc01@model']
resultflux = -2.5*np.log10(fluxmodel)
resultflux = resultflux - np.mean(resultflux)
plt.figure(1)
plt.plot(yuandata[:,0], datay, '.')
plt.scatter(b['value@times@lc01@model'], resultflux, c='none',marker='o',edgecolors='r', s=40)
#plt.plot(b['value@times@lc01@model']+0.005, resultflux, '.')
#plt.plot(b['value@times@lc01@model'], -2.5*np.log10(b['value@fluxes@lc01@model'])+0.64, '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)


ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


def calculater(ydata, caldata):
    res_ydata  = np.array(ydata) - np.array(caldata)
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((ydata - np.mean(ydata))**2)
    r_squared  = 1 - (ss_res / ss_tot)
    return r_squared


#R_2 = calculater(datay, resultflux)
#
#print(R_2)

