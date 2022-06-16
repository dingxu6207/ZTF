# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:14:02 2022

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

path = ''
lightdata = np.loadtxt(path+'savedata0170T.txt') 
#sns.set()




plt.figure(0)
T = lightdata[:,201]
#sns.kdeplot(T*5850,shade=True)
plt.hist(T*5850, bins=1000)
#plt.title('T1',fontsize=18)
plt.xlabel('T1',fontsize=18)
plt.ylabel('number',fontsize=18)

plt.figure(1)
T2T1 = lightdata[:,202]
#sns.kdeplot(T*5850,shade=True)
plt.hist(T2T1, bins=1000)
#plt.title('T1',fontsize=18)
plt.xlabel('T2T1',fontsize=18)
plt.ylabel('number',fontsize=18)


plt.figure(2)
incldata = lightdata[:,203]
#sns.kdeplot(incldata,shade=True)
plt.hist(incldata*90, bins=1000)
#plt.title('incl',fontsize=18)
plt.xlabel('incl',fontsize=18)
plt.ylabel('number',fontsize=18)


plt.figure(3)
qdata = lightdata[:, 204]
#sns.kdeplot(qdata,shade=True)
plt.hist(qdata, bins=1000)
#plt.title('q',fontsize=18)
plt.xlabel('q',fontsize=18)
plt.ylabel('number',fontsize=18)

plt.figure(4)
r1data = lightdata[:, 205]
#sns.kdeplot(rdata,shade=True)
plt.hist(r1data, bins=1000)
#plt.title('R1',fontsize=18)
plt.xlabel('R1',fontsize=18)
plt.ylabel('number',fontsize=18)


plt.figure(5)
r2r1data = lightdata[:, 206]
#sns.kdeplot(rdata,shade=True)
plt.hist(r2r1data, bins=1000)
#plt.title('R1',fontsize=18)
plt.xlabel('R2R1',fontsize=18)
plt.ylabel('number',fontsize=18)


plt.figure(6)
eccdata = lightdata[:, 207]
#sns.kdeplot(rdata,shade=True)
plt.hist(eccdata, bins=1000)
#plt.title('f',fontsize=18)
plt.xlabel('ecc',fontsize=18)
plt.ylabel('number',fontsize=18)

