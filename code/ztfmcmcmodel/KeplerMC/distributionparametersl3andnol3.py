# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 22:39:01 2022

@author: dingxu
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\KeplerMC\\'
lightdata = np.loadtxt(path+'savedata01050.txt') 
lightdatal3 = np.loadtxt(path+'savedata01050l3.txt') 
#sns.set()


plt.figure(0)
T = lightdata[:,100]
Tl3 = lightdatal3[:,100]
sns.kdeplot(T, shade=False, label='nol3')
sns.kdeplot(Tl3, shade=False, label='l3', color ='r')
plt.xlabel('T1',fontsize=18)
plt.ylabel('density',fontsize=18)
plt.legend(loc='upper right')

plt.figure(1)
incldata = lightdata[:,102]
incldatal3 = lightdatal3[:,102]
sns.kdeplot(incldata, shade=False, label='nol3')
sns.kdeplot(incldatal3, shade=False, label='l3', color ='r')
plt.xlabel('incl',fontsize=18)
plt.ylabel('density',fontsize=18)
plt.legend(loc='upper right')

plt.figure(2)
qdata = lightdata[:,103]
qdatal3 = lightdatal3[:,103]
sns.kdeplot(qdata, shade=False, label='nol3')
sns.kdeplot(qdatal3, shade=False, label='l3', color ='r')
plt.xlabel('q',fontsize=18)
plt.ylabel('density',fontsize=18)
plt.legend(loc='upper right')

plt.figure(3)
rdata = lightdata[:,104]
rdatal3 = lightdatal3[:,104]
sns.kdeplot(rdata, shade=False, label='nol3')
sns.kdeplot(rdatal3, shade=False, label='l3', color ='r')
plt.xlabel('f',fontsize=18)
plt.ylabel('density',fontsize=18)
plt.legend(loc='upper right')

plt.figure(4)
tdata = lightdata[:,105]
tdatal3 = lightdatal3[:,105]
sns.kdeplot(tdata, shade=False, label='nol3')
sns.kdeplot(tdatal3, shade=False, label='l3', color ='r')
plt.xlabel('T2/T1',fontsize=18)
plt.ylabel('density',fontsize=18)
plt.legend(loc='upper right')

plt.figure(5)
l3data = lightdatal3[:,106]
sns.kdeplot(l3data, shade=False, label='l3', color ='r')
plt.xlabel('l3',fontsize=18)
plt.ylabel('density',fontsize=18)
plt.legend(loc='upper right')