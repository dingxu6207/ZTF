# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:24:59 2022

@author: dingxu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

filename = 'mcmcerror20.csv'
df = pd.read_csv(filename)

dfdata = df[['INCL','INCLERROR','Q','QERROR','F','FERROR','T2T1','T2T1ERROR','realincl', 'realq', 'realf', 'realt2t1','maxminmag']]

#dfdata.to_csv('testcsv2.csv',encoding='gbk')

##################################################
INCL = dfdata.iloc[:,0]
realINCL = dfdata.iloc[:,8]
sigmaincl = dfdata.iloc[:,1]
resiualincl = INCL-realINCL

print(np.mean(resiualincl), np.std(resiualincl))

plt.figure(0)
plt.hist(resiualincl, bins=500, density = 0)
#plt.hist(sigmaincl, bins=1000, label='sigma incl', density = 0)
plt.xlim(-2,2)
plt.xlabel(r'$\Delta$'+'incl',fontsize=18)
plt.ylabel('number',fontsize=18)
#plt.legend()


#################################################
Q = dfdata.iloc[:,2]
realQ = dfdata.iloc[:,9]
sigmaQ = dfdata.iloc[:,3]
resiualQ = Q-realQ

print(np.mean(resiualQ), np.std(resiualQ))

plt.figure(1)
plt.hist(resiualQ, bins=1000, density = 0)
#plt.hist(sigmaQ, bins=1000, label='sigma Q', density = 1)
plt.xlim(-1,1)
plt.xlabel(r'$\Delta$'+'q',fontsize=18)
plt.ylabel('number',fontsize=18)
#plt.legend()
#####################################################

F = dfdata.iloc[:,4]
realF = dfdata.iloc[:,10]
sigmaF = dfdata.iloc[:,5]
resiualF = F-realF

print(np.mean(resiualF), np.std(resiualF))

plt.figure(2)
plt.hist(resiualF, bins=1000, density = 1)
#plt.hist(sigmaF, bins=1000, label='sigma F', density = 1)
plt.xlim(-0.2,0.2)
plt.xlabel(r'$\Delta$'+'f',fontsize=18)
plt.ylabel('nunber',fontsize=18)
#plt.legend()
#####################################################
T2T1 = dfdata.iloc[:,6]
realT2T1 = dfdata.iloc[:,11]
sigmaT2T1 = dfdata.iloc[:,7]
resiualT2T1 = T2T1-realT2T1

print(np.mean(resiualT2T1), np.std(resiualT2T1))

plt.figure(3)
plt.hist(resiualT2T1, bins=2000, density = 1)
#plt.hist(sigmaT2T1, bins=1000, label='sigma F\T2T1', density = 1)
plt.xlim(-0.02,0.02)
plt.xlabel(r'$\Delta$'+'T2T1',fontsize=18)
plt.ylabel('number',fontsize=18)
#plt.legend()


#############################################################
#dataselect = dfdata[dfdata.iloc[:,2]<6]
#dataselect = dataselect[dataselect.iloc[:,2]>5]
#dataselect = dataselect[dataselect.iloc[:,0]<90]
#errorselec = dataselect.iloc[:,3]/10
#plt.figure(4)
#plt.plot(np.array(dataselect.iloc[:,0]), np.array(errorselec), '.')

