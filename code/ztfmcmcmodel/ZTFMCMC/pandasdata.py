# -*- coding: utf-8 -*-0
"""
Created on Thu Jan 27 21:19:03 2022

@author: dingxu
"""

import pandas as pd
import os
import numpy as np
import corner
import matplotlib.pylab as plt

temp = []
path = 'I:\\backup\\ZTFCODE\\code3\\parameterfile\\'

for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)

       if (strfile[-4:] == '.csv'):
           print(strfile)
           dfone = pd.read_csv(strfile)
           dfone = dfone.iloc[:,0:28]
           dfone.columns = ['index', 'name', 'idname', 'RA', 'DEC', 'Period', 'gmag', 'rmag', 'T1', 'T1error', 
              'incl', 'inclerror', 'q', 'qerror', 'f','ferror', 't2t1', 't2t1error',
              'l3', 'l3error', 'inputtemper', 'R2', 'pbdic', 'pr1', 'pr2', 'stdflag','stdchancha','selectflag']
           temp.append(dfone.iloc[:,0:28])
           

df = pd.concat(temp, axis=0)

dfdata = df[df['stdflag'] < 1.05]
dfdata = dfdata[dfdata['stdflag'] > 0.95]
#dfdata = dfdata[dfdata['selectflag'] > 0.8]
#dfdata = dfdata[dfdata['selectflag'] < 4]
dfdata = dfdata[dfdata['stdchancha'] < 0.04]
#dfdata = dfdata[dfdata['l3'] < 0.1]

dfdata['colorindex'] = dfdata['gmag'] - dfdata['rmag']
dfdata = dfdata[dfdata['Period'] < 1.0]
dfdata = dfdata[dfdata['colorindex'] > -2]

qdata = dfdata['q']
npqdatay = np.array(qdata)
npqdata = np.copy(npqdatay)
npqdata[npqdata>1] = 1/npqdata[npqdata>1]

cornordata = dfdata[['Period', 'colorindex', 'T1', 'incl', 'q', 't2t1', 'f', 'pr1', 'pr2', 'pbdic']]
#cornordata['qinverse'] = npqdata
#cornordata = dfdata[['Period', 'T1']]

npcor = np.array(cornordata.iloc[0:,0:])

figure = corner.corner(npcor, bins=100,labels=[r"$Period$", r"$colorindex$", r"$T1$", r"$incl$", r"$q$", r"$t2t1$", r"$f$", r"$r1$", r"$r2$", r"$L2/L1$"],
                       show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={"fontsize": 15}, color ='blue')
#figure = corner.corner(npcor, bins=100,labels=[r"$Period$", r"$colorindex$"])
plt.figure(1)
plt.savefig('corner.png')



plt.figure(2)
plt.hist(npqdata, bins=100)
plt.xlabel('q',fontsize=14)
plt.ylabel('frquentcy',fontsize=14)
plt.title(r"$\mu$"+'='+str(np.round(np.mean(npqdata),2)))


plt.figure(3)
resiualdata = dfdata['stdchancha']
npredatay = np.array(resiualdata)
plt.hist(npredatay, bins=100)
plt.xlabel('resiual',fontsize=18)
plt.ylabel('frquentcy',fontsize=18)
plt.title(r"$\mu$"+'='+str(np.round(np.mean(resiualdata),2)))


plt.figure(4)
selectdata = dfdata['selectflag']
npsedatay = np.array(selectdata)
plt.hist(npsedatay, bins=100)
plt.xlabel('divnoise',fontsize=18)
plt.ylabel('frquentcy',fontsize=18)
plt.title(r"$\mu$"+'='+str(np.round(np.mean(npsedatay),2)))