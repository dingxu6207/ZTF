# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:29:30 2022

@author: dingxu
"""

import pandas as pd
import numpy as np
import corner
import matplotlib.pylab as plt

path = 'I:\\backup\\ZTFCODE\\code3\\parameterfile\\'
file = 'parameter0.csv'

df = pd.read_csv(path+file)

df.columns = ['index', 'name', 'idname', 'RA', 'DEC', 'Period', 'gmag', 'rmag', 'T1', 'T1error', 
              'incl', 'inclerror', 'q', 'qerror', 'f','ferror', 't2t1', 't2t1error',
              'l3', 'l3error', 'inputtemper', 'R2', 'pbdic', 'pr1', 'pr2', 'stdflag']


dfdata = df[df['stdflag'] < 1.05]
dfdata = dfdata[dfdata['stdflag'] > 0.95]
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

figure = corner.corner(npcor, bins=100,labels=[r"$Period$", r"$colorindex$", r"$T1$", r"$incl$", r"$q$", r"$t2t1$", r"$f$", r"$pr1$", r"$pr2$", r"$pbdic$"], show_titles=True)
#figure = corner.corner(npcor, bins=100,labels=[r"$Period$", r"$colorindex$"])
plt.figure(1)
plt.savefig('corner.png')



plt.figure(2)
plt.hist(npqdata, bins=100)
plt.xlabel('q',fontsize=14)
plt.ylabel('frquentcy',fontsize=14)
plt.title(r"$\mu$"+'='+str(np.mean(npqdata))+' '+r"$\sigma$"+'='+str(np.std(npqdata)))