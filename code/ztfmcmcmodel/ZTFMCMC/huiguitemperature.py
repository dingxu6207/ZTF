# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:24:16 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

magtemprature = np.loadtxt('radectemparature.txt')


magtemprature = magtemprature[~np.isnan(magtemprature[:,2])]

magtemprature = magtemprature[magtemprature[:,0] != 0]

magtemprature = magtemprature[magtemprature[:,1] != 0]

plt.hist(magtemprature[:,2], bins=100)


np.savetxt('magtemprature.txt', magtemprature)