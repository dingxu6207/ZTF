# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:05:35 2022

@author: dingxu
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import corner
import seaborn as sns
from scipy.optimize import curve_fit

df = pd.read_csv('savedata.csv')
patgfigure = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\paperfigure\\'

savedata = df[['Period','colorindex','T1','incl', 'q','t2t1','f','pr1','pr2','pbdic']]
savedata['T2'] = savedata['T1']*savedata['t2t1']
npcor = np.array(savedata.iloc[0:,0:])
data = npcor[:,[0,1,2,10,3,4,6,7,8,9]] #Period, colorindex, T1, T2, incl, q, f,  pr1, pr2, pbdic


hang,lie = data.shape

for i in range (0,hang): 
    if data[i,5]>1:
        
        data[i,5] = 1/data[i,5]
        
        if data[i,9] != 0:
            data[i,9] = 1/data[i,9]
        
        temp = data[i,2]
        data[i,2] = data[i,3]
        data[i,3] = temp
        
        tempr = data[i,7]
        data[i,7] = data[i,8]
        data[i,8] = tempr
 
def func(x,  c):
    return x**c

       
plt.figure(0)
q = np.loadtxt(patgfigure+'q.txt')
plt.plot(q[:,0], q[:,1], label='Olivera Latković')
sns.kdeplot(data[:,5], shade=False, bw=0.03, label='This paper')
plt.legend()
plt.xlabel('q',fontsize=18)
plt.ylabel('probability density',fontsize=18)


plt.figure(1)
PT2T1 = np.loadtxt(patgfigure+'t2t1.txt')
delt2t1 = data[:,2]-data[:,3]
plt.plot(PT2T1[:,0], PT2T1[:,1], label='Olivera Latković')
sns.kdeplot(delt2t1, shade=False, label='This paper')
plt.legend()
plt.xlabel('T1-T2',fontsize=18)
plt.ylabel('probability density',fontsize=18)


plt.figure(2)
R1 = np.loadtxt(patgfigure+'R1.txt')
R2 = np.loadtxt(patgfigure+'R2.txt')
plt.plot(R1[:,0], R1[:,1], label='Olivera Latković_R1')
plt.plot(R2[:,0], R2[:,1], label='Olivera Latković_R2')
sns.kdeplot(data[:,7], shade=False, bw=0.03, label='This paper_R1')
sns.kdeplot(data[:,8], shade=False, bw=0.03, label='This paper_R2')
plt.xlabel(r"$R$",fontsize=18)
plt.ylabel('probability density',fontsize=18)




plt.figure(3)
period = np.loadtxt(patgfigure+'period.txt')
plt.plot(period[:,0], period[:,1], label='Olivera Latković')
sns.kdeplot(data[:,0], shade=False, label='This paper')
plt.legend()
plt.xlabel('period',fontsize=18)
plt.ylabel('probability density',fontsize=18)


plt.figure(4)
fillout = np.loadtxt(patgfigure+'f.txt')
plt.plot(fillout[:,0], fillout[:,1], label='Olivera Latković')
sns.kdeplot(data[:,6], shade=False, bw=0.03, label='This paper')
plt.legend()
plt.xlabel('Fillout',fontsize=18)
plt.ylabel('probability density',fontsize=18)


plt.figure(5)
T = np.loadtxt(patgfigure+'T.txt')
plt.plot(T[:,0], T[:,1], label='Olivera Latković')
sns.kdeplot(data[:,2], shade=False, label='This paper')
plt.legend()
plt.xlabel('T1',fontsize=18)
plt.ylabel('probability density',fontsize=18)


plt.figure(6)
incl = np.loadtxt(patgfigure+'incl.txt')
plt.plot(incl[:,0], incl[:,1], label='Olivera Latković')
sns.kdeplot(data[:,4], shade=False, label='This paper')
plt.legend()
plt.xlabel('incl',fontsize=18)
plt.ylabel('probability density',fontsize=18)

#plt.figure(3)
#plt.hist(data[:,8], bins=100, density=0)
#
R2R1 = data[:,8]/data[:,7]
plt.figure(7)
plt.plot(data[:,5], R2R1, '.', label='origin data')

popt, pcov = curve_fit(func, data[:,5], R2R1)
print(popt[0])
ALLY = data[:,5]**popt[0]
plt.plot(data[:,5], ALLY, '.', c='r', label='fitting data')
plt.legend()

plt.xlabel('q',fontsize=18)
plt.ylabel('R2/R1',fontsize=18)

#Period, colorindex, T1, T2, incl, q, f,  pr1, pr2, pbdic
plt.figure(8)
figure = corner.corner(data, bins=100, labels=[r"$Period$", r"$colorindex$", r"$T1$", r"$T2$", r"$incl$", r"$q$", r"$f$", r"$pr1$", r"$pr2$", r"$L2/L1$"],
                       title_kwargs={"fontsize": 15}, label_kwargs={"fontsize": 15}, show_titles=True, color ='blue')
plt.savefig('corner.png')
