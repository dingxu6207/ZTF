# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:26:43 2022

@author: dingxu
"""

import numpy as np
import corner
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from matplotlib.pyplot import MultipleLocator

data = np.loadtxt('data01.txt')

'''
rang =  [(0.1,2.), (4000.,9000),  (4000.,9000),  (20.,120), (0.,1), (0.,1), (0.3, 0.65), (0.2,0.5), (0.,2)]
figure = corner.corner(data, bins=100,range=rang,labels=[r"$Period$", r"$T1$", r"$T2$", r"$incl$", r"$q$", r"$f$", r"$r1$", r"$r2$", r"$L2/L1$"],
                       show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={"fontsize": 15}, color ='blue')
#figure = corner.corner(npcor, bins=100,labels=[r"$Period$", r"$colorindex$"])
plt.figure(1)
plt.savefig('corner.png')
'''

def func(x, ap, bp):
    return ap*x+bp

def func1(x,  bp):
    return x+bp

period1 = data[:,0]
T11 = data[:,1]
T2 = data[:,2]

data1 = data[data[:,0]<0.5]

period = data1[:,0]
T1 = data1[:,1]

popt, pcov = curve_fit(func, period, T1)

plt.figure(0)
ax = plt.gca()
y_major_locator = MultipleLocator(2000)
ax.yaxis.set_major_locator(y_major_locator)
plt.plot(period1, T11, '.', label='origin data')
plt.axvline(x = 0.5,ls="--",c="green",linewidth=1)#添加垂直直线
plt.xlim(0, 1)
plt.ylim(3000, 10000)

ALLY = period*popt[0] + popt[1]
plt.plot(period, ALLY, '.', c='r', label='fitting data')
plt.legend()
plt.xlabel('period',fontsize=18)
plt.ylabel('T1[k]',fontsize=18)
perr = np.sqrt(np.diag(pcov))
print(perr)

plt.figure(1)
plt.plot(T11, T2, '.', label='origin data')
popt, pcov = curve_fit(func1, T11, T2)
ALLY = T11 + popt[0]
plt.plot(T11, ALLY, '.', c='r', label='fitting data')
plt.xlabel('T1[k]',fontsize=18)
plt.ylabel('T2[k]',fontsize=18)
plt.legend()

def func2(x,  c):
    return x**c

plt.figure(2)
q = data[:,4]
r1 = data[:,6]
r2 = data[:,7]
ratio = r2/r1

popt, pcov = curve_fit(func2, q, ratio)
print(popt[0])
ALLY = q**popt[0]
perr = np.sqrt(np.diag(pcov))
print(perr)
plt.figure(2)
plt.plot(q, ratio, '.', label='origin data')
plt.plot(q, ALLY, '.', c='r', label='fitting data')
plt.xlabel('q',fontsize=18)
plt.ylabel('r2/r1',fontsize=18)
plt.legend()

L2L1 = data[:,8]
plt.figure(3)
plt.plot(q, L2L1, '.', label='origin data')
popt, pcov = curve_fit(func, q, L2L1)
ALLY = q*popt[0] + popt[1]
#plt.ylim(0,1)
plt.xlim(0,1)

plt.plot(q, ALLY, '.', c='r', label='fitting data')
plt.xlabel('q',fontsize=18)
plt.ylabel('L2/L1',fontsize=18)
plt.legend()
