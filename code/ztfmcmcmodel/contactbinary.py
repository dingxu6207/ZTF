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

times  = np.linspace(0,1,100)

#b.add_dataset('lc', times=phoebe.linspace(0,1,150), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', compute_times=phoebe.linspace(0,1,100),passband= 'LSST:r')

b['period@binary'] = 1

b['incl@binary'] =  0.71348028*90   #58.528934
b['q@binary'] =    0.40296497
b['teff@primary'] =  1.0338496*5850  #6208 
b['teff@secondary'] = 1.05976131*5850*0.92909411#6500*100.08882*0.01 #6087

b['sma@binary'] = 1#0.05 2.32
b.flip_constraint('pot', solve_for='requiv@primary')
b.flip_constraint('fillout_factor', solve_for='pot')
b['fillout_factor'] = 0.01787572
#b['fillout_factor@contact_envelope'] = 0.5

#b['fillout_factor@contact_envelope@envelope@component'] = 0.5


#print(b['sma@binary'])

#b['requiv@primary'] = 0.5    #0.61845703

'''
b.get_constraint(qualifier='fillout_factor@contact_envelope@envelope@component')
b['fillout_factor@contact_envelope@envelope@component'] = 0.3
b.flip_constraint(qualifier='fillout_factor@contact_envelope@envelope@component', solve_for='requiv@primary')
'''
#b.get_constraint(qualifier='fillout_factor')
#b.flip_constraint(qualifier='requiv@secondary@component', solve_for='requiv@primary@component')
#b.flip_constraint(qualifier='fillout_factor', context='contact_envelope', solve_for='requiv@primary@star@component')
#b['fillout_factor'] = 0.3

#b.add_dataset('mesh', times=[0.25], dataset='mesh01')


try:
    b.run_compute(irrad_method='none')
except:
    b.run_compute(ntriangles=10000)
    

#plt.figure(0)
#afig, mplfig = b.plot(show=True, legend=True)

print(b['fillout_factor@contact_envelope'])

#print(b.filter(context='constraint').qualifiers)
#print(b.filter(context='component', kind='star', component='primary'))
#print(b.filter(component='binary'))
#print(b.filter(context='component'))
#print(b.filter(context='component', kind='envelope'))
#print(b.filter(context='system'))
#print(b.compute_pblums())
#print(b.filter('pblum*'))
print(b.filter('r*'))

print(b.filter('l3*'))

np.savetxt('data0.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T)


#fluxes_model = b['fluxes@model'].interp_value(times=times)
#fluxcha = fluxes_model-b['value@times@lc01@model']

#print(fluxcha)

path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 10226388.txt'
#file = 'V396Mon_Yang2001B.nrm'
#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\nihe\\'
yuandata = np.loadtxt(path+file)
#yuandata = np.loadtxt(path+file)
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
#plt.scatter(b['value@times@lc01@model'], resultflux, c='none',marker='o',edgecolors='r', s=80)
plt.plot(b['value@times@lc01@model'], resultflux, '.')
#plt.plot(b['value@times@lc01@model'], -2.5*np.log10(b['value@fluxes@lc01@model'])+0.64, '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)


ax = plt.gca()
ax.yaxis.set_ticks_position('left') #???y???????????????????????????
ax.invert_yaxis() #y?????????
