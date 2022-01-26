# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:29:30 2022

@author: dingxu
"""

import pandas as pd
import numpy as np
import corner
import matplotlib.pylab as plt

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\'
file = 'parameter.csv'

df = pd.read_csv(path+file)

df.columns = ['index', 'name', 'RA', 'DEC', 'Period', 'Prob', 'T1', 'T1error', 
              'incl', 'inclerror', 'q', 'qerror', 'f','ferror', 't2t1', 't2t1error',
              'l3', 'l3error', 'inputtemper', 'R2', 'pbdic', 'pr1', 'pr2', 'stdflag', 'stdchancha', 'selectflag', 'r_2']


dfdata = df[df['r_2'] >0.95]
#dfdata = df[df['selectflag'] > 0.9]
dfdata = dfdata[dfdata['selectflag'] < 10]
dfdata = dfdata[dfdata['stdflag'] < 1.02]
dfdata = dfdata[dfdata['stdflag'] > 0.98]
#dfdata = dfdata[dfdata['l3'] < 0.1]


dfdata = dfdata[dfdata['Period'] <2]


qdata = dfdata['q']
npqdatay = np.array(qdata)
npqdata = np.copy(npqdatay)
npqdata[npqdata>1] = 1/npqdata[npqdata>1]

cornordata = dfdata[['Period', 'T1', 'incl', 'q', 't2t1', 'f', 'pr1', 'pr2', 'pbdic']]
#cornordata['qinverse'] = npqdata
#cornordata = dfdata[['Period', 'T1']]
savedata = dfdata[['name','RA','DEC','Period', 'T1','T1error','incl','inclerror','q','qerror','t2t1','t2t1error', 'f','ferror','l3','l3error','pr1', 'pr2', 'pbdic', 'r_2']]
savedata.to_csv('savedata.csv', index=0)

npcor = np.array(cornordata.iloc[0:,0:])

figure = corner.corner(npcor, bins=100,labels=[r"$Period$", r"$T1$", r"$incl$", r"$q$", r"$T2/T1$", r"$f$", r"$r1$", r"$r2$", r"$L2/L1$"],
                       show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={"fontsize": 15}, color ='blue')
#figure = corner.corner(npcor, bins=100,labels=[r"$Period$", r"$colorindex$"])
plt.figure(1)
plt.savefig('corner.png')



plt.figure(2)
plt.hist(npqdata, bins=100)
plt.xlabel('q',fontsize=14)
plt.ylabel('frquentcy',fontsize=14)
plt.title(r"$\mu$"+'='+str(np.round(np.mean(npqdata),2)))

rdata = dfdata['r_2']
nprdatay = np.array(rdata)
plt.figure(3)
plt.hist(nprdatay, bins=100)
plt.xlabel(r"$R^2$",fontsize=18)
plt.ylabel('frquentcy',fontsize=18)
#plt.title(r"$\mu$"+'='+str(np.mean(nprdatay))+' '+r"$\sigma$"+'='+str(np.std(nprdatay)))