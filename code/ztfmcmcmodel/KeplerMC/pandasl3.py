# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:36:48 2022

@author: dingxu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

filename = 'testcsvl3.csv'
df = pd.read_csv(filename)

dfdata = df[['INCL','INCLERROR','Q','QERROR','F','FERROR','T2T1','T2T1ERROR','l3','l3ERROR',
             'realincl', 'realq', 'realf', 'realt2t1','reall3','maxminmag']]

#dfdata.to_csv('testcsv2.csv',encoding='gbk')

##################################################
INCL = dfdata.iloc[:,0]
realINCL = dfdata.iloc[:,10]
sigmaincl = dfdata.iloc[:,1]
resiualincl = INCL-realINCL

#print(np.mean(resiualincl), np.std(resiualincl))
print(np.round(np.mean(resiualincl),3), np.round(np.std(resiualincl),3))

plt.figure(0)
plt.hist(resiualincl, bins=2000, density = 0)
#plt.hist(sigmaincl, bins=1000, label='sigma incl', density = 0)
plt.xlim(-10, 10)
plt.xlabel(r'$\Delta$'+'incl',fontsize=18)
plt.ylabel('number',fontsize=18)
#plt.legend()
plt.title(r'$\Delta$'+'incl='+str(np.round(np.mean(resiualincl),3))+r'$\pm$'+str(np.round(np.std(resiualincl),3)),fontsize=18)

#################################################
Q = dfdata.iloc[:,2]
realQ = dfdata.iloc[:,11]
sigmaQ = dfdata.iloc[:,3]
resiualQ = Q-realQ

print(np.mean(resiualQ), np.std(resiualQ))

plt.figure(1)
plt.hist(resiualQ, bins=1000, density = 0)
#plt.hist(sigmaQ, bins=1000, label='sigma Q', density = 1)
plt.xlim(-2, 2)
plt.xlabel(r'$\Delta$'+'q',fontsize=18)
plt.ylabel('number',fontsize=18)
#plt.legend()
plt.title(r'$\Delta$'+'q='+str(np.round(np.mean(resiualQ),3))+r'$\pm$'+str(np.round(np.std(resiualQ),3)),fontsize=18)
#####################################################

F = dfdata.iloc[:,4]
realF = dfdata.iloc[:,12]
sigmaF = dfdata.iloc[:,5]
resiualF = F-realF

print(np.mean(resiualF), np.std(resiualF))

plt.figure(2)
plt.hist(resiualF, bins=1000, density = 1)
#plt.hist(sigmaF, bins=1000, label='sigma F', density = 1)
plt.xlim(-0.4,0.4)
plt.xlabel(r'$\Delta$'+'f',fontsize=18)
plt.ylabel('nunber',fontsize=18)
#plt.legend()
plt.title(r'$\Delta$'+'f='+str(np.round(np.mean(resiualF),3))+r'$\pm$'+str(np.round(np.std(resiualF),3)),fontsize=18)

#####################################################
T2T1 = dfdata.iloc[:,6]
realT2T1 = dfdata.iloc[:,13]
sigmaT2T1 = dfdata.iloc[:,7]
resiualT2T1 = T2T1-realT2T1

print(np.mean(resiualT2T1), np.std(resiualT2T1))

plt.figure(3)
plt.hist(resiualT2T1, bins=2000, density = 1)
#plt.hist(sigmaT2T1, bins=1000, label='sigma F\T2T1', density = 1)
plt.xlim(-0.075, 0.075)
plt.xlabel(r'$\Delta$'+'T2/T1',fontsize=18)
plt.ylabel('number',fontsize=18)
#plt.legend()
plt.title(r'$\Delta$'+'T2/T1='+str(np.round(np.mean(resiualT2T1),3))+r'$\pm$'+str(np.round(np.std(resiualT2T1),3)),fontsize=18)
#############################################################
l3 = dfdata.iloc[:,8]
reall3 = dfdata.iloc[:,14]
sigmal3 = dfdata.iloc[:,9]
resiuall3 = l3-reall3

print(np.mean(resiuall3), np.std(resiuall3))

plt.figure(4)
plt.hist(resiuall3, bins=2000, density = 1)
#plt.hist(sigmaT2T1, bins=1000, label='sigma F\T2T1', density = 1)
plt.xlim(-0.2,0.2)
plt.xlabel(r'$\Delta$'+'l3',fontsize=18)
plt.ylabel('number',fontsize=18)
plt.title(r'$\Delta$'+'l3='+str(np.round(np.mean(resiuall3),3))+r'$\pm$'+str(np.round(np.std(resiuall3),3)),fontsize=18)

temperror = []
hang,lie = dfdata.shape
QandRealQ = dfdata[['Q', 'realq']]
npQandrealQ = np.array(QandRealQ)
for i in range(0, hang):
    if npQandrealQ[i,0] > 1:
        error = 1/npQandrealQ[i,0] - 1/npQandrealQ[i,1]
        temperror.append(error)
    else:
        temperror.append(npQandrealQ[i,0]-npQandrealQ[i,1])
        
plt.figure(5)
plt.hist(temperror, bins=2000, density = 1)
plt.xlim(-0.05, 0.05)
plt.xlabel(r'$\Delta$'+'q',fontsize=18)
plt.ylabel('number',fontsize=18)