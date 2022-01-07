# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 20:56:27 2021

@author: dingxu
"""
import phoebe
import numpy as np
import matplotlib.pyplot as plt


logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,100)
b.add_dataset('lc', compute_times=times, passband= 'LSST:r')

b['period@binary'] = 1

b['incl@binary'] =  90   #58.528934
b['q@binary'] =    0.5
b['teff@primary'] =  5800  #6208 
b['teff@secondary'] = 5800#6500*100.08882*0.01 #6087

b['sma@binary'] = 3#0.05 2.32
b.flip_constraint('pot', solve_for='requiv@primary')
b.flip_constraint('fillout_factor', solve_for='pot')
b['fillout_factor'] = 50*0.01

b.set_value('l3_mode', 'fraction')
l3fra = 0.2
b.set_value('l3_frac', l3fra) 

temp = []
plt.figure(0)
for i in np.arange(0, 0.9, 0.1):
    print(i)
    #b['incl@binary'] = i
    l3fra = np.round(i, 1)
    b.set_value('l3_frac', l3fra) 
    b.run_compute(irrad_method='none')
    fluxmodel = b['value@fluxes@lc01@model']
    #resultflux = -2.5*np.log10(fluxmodel)
    timedata = b['value@times@lc01@model']
    plt.plot(timedata, fluxmodel, linewidth = 3, label= str(l3fra))
    plt.xlabel('time', fontsize=14)
    plt.ylabel('flux', fontsize=14)
    plt.legend(loc='upper right', fontsize=10) # 标签位置
#ax = plt.gca()
#ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
#ax.invert_yaxis() #y轴反向


#temp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#temp = [0.1, 0.4, 0.8]
#plt.figure(0)
#for i in range(0,3):
#    print(i)
#    #b['fillout_factor'] = temp[i]
#    #b['q@binary'] = temp[i]
#    b.run_compute(irrad_method='none')
#    fluxmodel = b['value@fluxes@lc01@model']
#    resultflux = -2.5*np.log10(fluxmodel)
#    timedata = b['value@times@lc01@model']
#    plt.plot(timedata, resultflux, linewidth = 3, label= 'q = '+str(temp[i]))
#    plt.xlabel('phase', fontsize=14)
#    plt.ylabel('mag', fontsize=14)
#    plt.legend(loc='upper right', fontsize=10) # 标签位置
#ax = plt.gca()
#ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
#ax.invert_yaxis() #y轴反向
    