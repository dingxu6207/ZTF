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

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\TESSSAMPLE\\'
lightdata = np.loadtxt(path+'savedata01050T.txt') 
#sns.set()


plt.figure(0)
T = lightdata[:,100]
#sns.kdeplot(T*5850,shade=True)
plt.hist(T*5850, bins=1000)
plt.title('T1',fontsize=18)
plt.xlabel('T1',fontsize=18)
plt.ylabel('number',fontsize=18)


plt.figure(1)
incldata = lightdata[:,102]
#sns.kdeplot(incldata,shade=True)
plt.hist(incldata, bins=1000)
plt.title('incl',fontsize=18)
plt.xlabel('incl',fontsize=18)
plt.ylabel('number',fontsize=18)

plt.figure(2)
qdata = lightdata[:,103]
#sns.kdeplot(qdata,shade=True)
plt.hist(qdata, bins=1000)
plt.title('q',fontsize=18)
plt.xlabel('q',fontsize=18)
plt.ylabel('number',fontsize=18)

plt.figure(3)
rdata = lightdata[:,104]
#sns.kdeplot(rdata,shade=True)
plt.hist(rdata, bins=1000)
plt.title('f',fontsize=18)
plt.xlabel('f',fontsize=18)
plt.ylabel('number',fontsize=18)

plt.figure(4)
tdata = lightdata[:,105]
#sns.kdeplot(tdata,shade=True)
plt.hist(tdata, bins=1000)
plt.title('T2/T1',fontsize=18)
plt.xlabel('T2/T1',fontsize=18)
plt.ylabel('number',fontsize=18)
