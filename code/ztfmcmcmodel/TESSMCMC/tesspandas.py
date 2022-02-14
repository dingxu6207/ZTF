# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:29:30 2022

@author: dingxu
"""

import pandas as pd
import numpy as np
import corner
import matplotlib.pylab as plt
import seaborn as sns
#sns.set()
patgfigure = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\paperfigure\\'
path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\'
file = 'parameter.csv'

df = pd.read_csv(path+file)

df.columns = ['index', 'name', 'RA', 'DEC', 'Period', 'Prob', 'T1', 'T1error', 
              'incl', 'inclerror', 'q', 'qerror', 'f','ferror', 't2t1', 't2t1error',
              'l3', 'l3error', 'inputtemper', 'R2', 'pbdic', 'pr1', 'pr2', 'stdflag', 'stdchancha', 'selectflag', 'r_2']


dfdata = df[df['r_2'] >0]
#dfdata = df[df['selectflag'] > 0.9]
#dfdata = dfdata[dfdata['selectflag'] < 10]
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


q = np.loadtxt(patgfigure+'q.txt')
plt.figure(2)
#plt.hist(npqdata, bins=100, density=1)
plt.plot(q[:,0], q[:,1], label='others')
sns.kdeplot(npqdata, shade=False, bw=0.03, label='ours')
plt.xlabel('q',fontsize=18)
plt.ylabel('probability density',fontsize=18)
#plt.title(r"$\mu$"+'='+str(np.round(np.mean(npqdata),2)))

rdata = dfdata['r_2']
nprdatay = np.array(rdata)
plt.figure(3)
plt.hist(nprdatay, bins=100)
plt.xlabel(r"$R^2$",fontsize=18)
plt.ylabel('frequency',fontsize=18)
#plt.title(r"$\mu$"+'='+str(np.mean(nprdatay))+' '+r"$\sigma$"+'='+str(np.std(nprdatay)))

R = np.loadtxt(patgfigure+'R.txt')
prdata = dfdata['pr1']
npprdatay = np.array(prdata)
plt.figure(4)
#plt.hist(npprdatay, bins=100, density=1)
plt.plot(R[:,0], R[:,1], label='others')
sns.kdeplot(data=npprdatay, shade=False, bw=0.03, label='ours')
plt.xlabel(r"$R$",fontsize=18)
plt.ylabel('probability density',fontsize=18)


period = np.loadtxt(patgfigure+'period.txt')
perioddata = dfdata['Period']
npperioddatay = np.array(perioddata)
plt.figure(5)
plt.plot(period[:,0], period[:,1], label='others')
sns.kdeplot(npperioddatay, shade=False, label='ours')
plt.legend()
plt.xlabel('period',fontsize=18)
plt.ylabel('probability density',fontsize=18)

f = np.loadtxt(patgfigure+'f.txt')
fdata = dfdata['f']
npfdatay = np.array(fdata)
plt.figure(6)
plt.plot(f[:,0], f[:,1], label='others')
sns.kdeplot(npfdatay, shade=False, bw=0.03, label='ours')
plt.legend()
plt.xlabel('Fillout',fontsize=18)
plt.ylabel('probability density',fontsize=18)

T = np.loadtxt(patgfigure+'T.txt')
T1data = dfdata['T1']
npT1datay = np.array(T1data)
plt.figure(7)
plt.plot(T[:,0], T[:,1], label='others')
sns.kdeplot(npT1datay, shade=False, label='ours')
plt.legend()
plt.xlabel('T1',fontsize=18)
plt.ylabel('probability density',fontsize=18)


incl = np.loadtxt(patgfigure+'incl.txt')
incldata = dfdata['incl']
npincldatay = np.array(incldata)
plt.figure(8)
plt.plot(incl[:,0], incl[:,1], label='others')
sns.kdeplot(npincldatay, shade=False, label='ours')
plt.legend()
plt.xlabel('incl',fontsize=18)
plt.ylabel('probability density',fontsize=18)

PT2T1 = np.loadtxt(patgfigure+'t2t1.txt')
t2t1 =  dfdata['t2t1']
npt2t1 = np.array(t2t1)
t1 = T1data*npt2t1
delt2t1 = t1-npT1datay
delt2t1 = delt2t1[delt2t1>-1000]
plt.figure(9)
plt.plot(PT2T1[:,0], PT2T1[:,1], label='others')
sns.kdeplot(delt2t1, shade=False, label='ours')
plt.legend()
plt.xlabel('T1-T2',fontsize=18)
plt.ylabel('probability density',fontsize=18)

