# -*- coding: utf-8 -*-
"""
Created on Fri May 27 23:03:23 2022

@author: dingxu
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import corner
import seaborn as sns
from scipy.optimize import curve_fit

df = pd.read_csv('WUMaCat.csv')

period = df['P']
incl = df['i']
T1 = df['T1']
T2 = df['T2']
R1 = df['r1p']
R2 = df['r2p']
f = df['f']
q = df['q']

data = np.loadtxt('data01.txt')
wq = data[:,4]
wperiod = data[:,0]
wincl = data[:,3]
wT1 = data[:,1]
wT2 = data[:,2]
wr1 = data[:,6]
wr2 = data[:,7]
wf = data[:,5]

plt.figure(0)
sns.kdeplot(period, shade=False,  label='Latković et al.(1941)')
sns.kdeplot(wperiod, shade=False,  label='this paper')
plt.xlim(0,2)
plt.xlabel('period',fontsize=18)
plt.ylabel('density',fontsize=18)

plt.figure(1)
sns.kdeplot(q, shade=False, label='Latković et al.(1941)')
sns.kdeplot(wq, shade=False,  label='this paper')
plt.xlabel('q',fontsize=18)
plt.ylabel('density',fontsize=18)

plt.figure(2)
sns.kdeplot(incl, shade=False,  label='Latković et al.(1941)')
sns.kdeplot(wincl, shade=False,  label='this paper')
plt.xlabel('incl',fontsize=18)
plt.ylabel('density',fontsize=18)

plt.figure(3)
sns.kdeplot(T1, shade=False,  label='Latković et al.(1941)')
sns.kdeplot(wT1, shade=False,  label='this paper')
plt.xlabel('T1',fontsize=18)
plt.ylabel('density',fontsize=18)

plt.figure(4)
sns.kdeplot(T1-T2, shade=False,  label='Latković et al.(1941)')
sns.kdeplot(wT1-wT2, shade=False,  label='this paper')
plt.xlabel('T1-T2',fontsize=18)
plt.ylabel('density',fontsize=18)

plt.figure(5)
sns.kdeplot(f, shade=False,  label='Latković et al.(1941)')
sns.kdeplot(wf, shade=False,  label='this paper')
plt.xlabel('fill-out factor',fontsize=18)
plt.ylabel('density',fontsize=18)

plt.figure(6)
sns.kdeplot(R1, shade=False,  label='Latković et al.(1941)_R1')
sns.kdeplot(R2, shade=False,  label='Latković et al.(1941)_R2')
sns.kdeplot(wr1, shade=False,  label='this paper_R1')
sns.kdeplot(wr2, shade=False,  label='this paper_R2')
plt.xlabel('Relative radius',fontsize=18)
plt.ylabel('density',fontsize=18)