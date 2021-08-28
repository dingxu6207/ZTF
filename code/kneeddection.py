# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:59:58 2021

@author: dingxu
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from kneed import KneeLocator

style.use('seaborn-whitegrid')

x = np.arange(1, 3, 0.01)*np.pi
y = np.cos(x)

# 计算各种参数组合下的拐点
kneedle_cov_inc = KneeLocator(x, 
                      y, 
                      curve='convex', 
                      direction='increasing',
                      online=True)

kneedle_cov_dec = KneeLocator(x, 
                      y, 
                      curve='convex', 
                      direction='decreasing',
                      online=True)

kneedle_con_inc = KneeLocator(x, 
                      y, 
                      curve='concave', 
                      direction='increasing',
                      online=True)

kneedle_con_dec = KneeLocator(x, 
                      y, 
                      curve='concave', 
                      direction='decreasing',
                      online=True)


fig, axe = plt.subplots(2, 2, figsize=[12, 12])

axe[0, 0].plot(x, y, 'k--')
axe[0, 0].annotate(s='Knee Point', xy=(kneedle_cov_inc.knee+0.2, kneedle_cov_inc.knee_y), fontsize=10)
axe[0, 0].scatter(x=kneedle_cov_inc.knee, y=kneedle_cov_inc.knee_y, c='b', s=200, marker='^', alpha=1)
axe[0, 0].set_title('convex+increasing')
axe[0, 0].fill_between(np.arange(1, 1.5, 0.01)*np.pi, np.cos(np.arange(1, 1.5, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe[0, 0].set_ylim(-1, 1)

axe[0, 1].plot(x, y, 'k--')
axe[0, 1].annotate(s='Knee Point', xy=(kneedle_cov_dec.knee+0.2, kneedle_cov_dec.knee_y), fontsize=10)
axe[0, 1].scatter(x=kneedle_cov_dec.knee, y=kneedle_cov_dec.knee_y, c='b', s=200, marker='^', alpha=1)
axe[0, 1].fill_between(np.arange(2.5, 3, 0.01)*np.pi, np.cos(np.arange(2.5, 3, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe[0, 1].set_title('convex+decreasing')
axe[0, 1].set_ylim(-1, 1)

axe[1, 0].plot(x, y, 'k--')
axe[1, 0].annotate(s='Knee Point', xy=(kneedle_con_inc.knee+0.2, kneedle_con_inc.knee_y), fontsize=10)
axe[1, 0].scatter(x=kneedle_con_inc.knee, y=kneedle_con_inc.knee_y, c='b', s=200, marker='^', alpha=1)
axe[1, 0].fill_between(np.arange(1.5, 2, 0.01)*np.pi, np.cos(np.arange(1.5, 2, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe[1, 0].set_title('concave+increasing')
axe[1, 0].set_ylim(-1, 1)

axe[1, 1].plot(x, y, 'k--')
axe[1, 1].annotate(s='Knee Point', xy=(kneedle_con_dec.knee+0.2, kneedle_con_dec.knee_y), fontsize=10)
axe[1, 1].scatter(x=kneedle_con_dec.knee, y=kneedle_con_dec.knee_y, c='b', s=200, marker='^', alpha=1)
axe[1, 1].fill_between(np.arange(2, 2.5, 0.01)*np.pi, np.cos(np.arange(2, 2.5, 0.01)*np.pi), 1, alpha=0.5, color='red')
axe[1, 1].set_title('concave+decreasing')
axe[1, 1].set_ylim(-1, 1)

# 导出图像
plt.savefig('图2.png', dpi=300)