# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:37:14 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

lightdata = np.loadtxt('savedatasample0.txt') 
#lightdata = np.loadtxt('alldatasample35.txt') 


import seaborn as sns
sns.set()


lightdata = lightdata[lightdata[:,100] > 20]
#lightdata = lightdata[lightdata[:,101] < 1]
#lightdata = lightdata[lightdata[:,103] < 1.15]
#lightdata = lightdata[lightdata[:,103] > 0.85]
newdata = lightdata[lightdata[:,101] < 0.2]
#d4data = lightdata[lightdata[:,101] > 0.2]
d4data = lightdata[lightdata[:,101] > 0.2]

dfdata = pd.DataFrame(newdata)
#dfdata = dfdata.sample(n=70590)
npdfdata = np.array(dfdata)

df4data = pd.DataFrame(d4data)
#df4data = df4data.sample(n=20000)
np4dfdata = np.array(df4data)

alldata = np.row_stack((np4dfdata, npdfdata))
lightdata = np.copy(alldata)

'''
lightdata = lightdata[lightdata[:,103]<1.04]
lightdata = lightdata[lightdata[:,103]>0.92]
'''
print(len(np4dfdata))
print(len(npdfdata))

plt.figure(0)
incldata = lightdata[:,100]
sns.kdeplot(incldata,shade=True)
#plt.title('incl')
plt.xlabel('incl',fontsize=14)
plt.ylabel('frequency',fontsize=14)

plt.figure(1)
qdata = lightdata[:,101]
sns.kdeplot(qdata,shade=True)
#plt.hist(qdata, bins=4000, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
#plt.title('q')
plt.xlabel('q',fontsize=14)
plt.ylabel('frequency',fontsize=14)

plt.figure(2)
rdata = lightdata[:,102]
sns.kdeplot(rdata,shade=True)
#plt.title('r')
plt.xlabel('f',fontsize=14)
plt.ylabel('frequency',fontsize=14)

plt.figure(3)
tdata = lightdata[:,103]
sns.kdeplot(tdata,shade=True)
#plt.hist(tdata, bins=4000, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
#plt.title('T2/T1')
plt.xlabel('T2/T1',fontsize=14)
plt.ylabel('frequency',fontsize=14)

np.savetxt('alldata35.txt', alldata)




