# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:44:31 2022

@author: jkf
"""

import numpy as np
import matplotlib.pyplot as plt
dat=np.loadtxt('magtemprature.txt')
CI=dat[:,0]-dat[:,1]
T=dat[:,2]

T1=1/T
p=np.polyfit(CI,T1,1)
z=p[0]*CI+p[1]
plt.figure()
plt.plot(CI,T1,'.')
plt.plot(CI,z,'+')
plt.figure()

plt.plot(CI,T,'.')
plt.plot(CI,1/z,'+')


a2=1/p[0]
a1=p[1]/p[0]
print(a1,a2)


