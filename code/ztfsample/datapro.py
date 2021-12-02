# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:17:03 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import os


'''
filedata = 'E:\\shunbianyuan\\data\\kpdata\\1220.lc'

data = np.loadtxt(filedata)

hangdata = data[:,0][0:100]
liedata = data[:,1][0:100]

ydata = data[100:102,0:2]
ydata = ydata.flatten()

plt.plot(hangdata, liedata, '.')
'''

#path = 'E:\\shunbianyuan\\data\\kpdata\\jiegui\\'
#path = 'E:\\shunbianyuan\\data\\kpdata\\test\\jiegui\\jiegui\\'
path = '/home/dingxu/桌面/ZTFDATATEMPTER/0-10-50Tsampler/'
#path = ''
mypath = []
count = 0
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-3:] == '.lc'):
           mypath.append(strfile)
           print(strfile)
           count = count+1
           print(count)
           

lenpath = len(mypath)
testdata = []
for i in range(lenpath):
    #print(mypath[i])
    try:
        data = np.loadtxt(mypath[i])
        # print(data.shape)
        hangdata = data[:,0][0:100]
        liedata = data[:,1][0:100]

        ydata = data[100:103,0:2]
        ydata = ydata.flatten()
    
        listliedata = list(liedata)
        listydata = list(ydata)
    
        listliedata.extend(listydata)
    
        lightydata = np.array(listliedata)
    #print(lightydata.shape)
    
        testdata.append(lightydata)
        
        print('it is ok'+str(i))
    except:
        print('it is error!')
    
lightdata = np.array(testdata)

savedata = np.savetxt('savedata01050T.txt', lightdata)