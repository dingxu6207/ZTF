# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:33:12 2022

@author: dingxu
"""
# https://zhuanlan.zhihu.com/p/32492090

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy.linalg import lstsq

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\paperfigure\\'
file = 'period.png'
img = Image.open(path+file)

plt.imshow(img, cmap='gray')

#pos = plt.ginput(30)
#
#print(pos)

#ax*x+bx+c =d
'''
2x + 3y = 5
x   + 3y = 3
x   +  y  = 2
'''

a = np.mat([[100.044**3,100.044**2,100.044,1],[150.027**3,150.027**2,150.027,1],[40.0886**2,40.0886,1]])
b = np.mat([4,2,6]).T
x = lstsq(a,b)
print(x)

def derivex(x):
    y = -6.05382203e-05*(x**2)-2.48747513e-02*x+7.09448468
    return y

v = derivex(200)
print(v)
