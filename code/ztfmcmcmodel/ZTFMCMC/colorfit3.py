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


from scipy.optimize import least_squares as ls
from scipy.optimize import minimize
def func(x0):
    return ((x0[1]/(x+x0[0])+x0[2]-y)**2).sum()
x=CI
y=T
x0=[1,1e4,2e3]
result=minimize(func, x0, method='Nelder-Mead',tol=1e-12)
x0=result.x
print(result.x)
#plt.figure()
z2=x0[1]/(x+x0[0])+x0[2]
plt.plot(CI,T,'.')
plt.plot(CI,z2,'+')
