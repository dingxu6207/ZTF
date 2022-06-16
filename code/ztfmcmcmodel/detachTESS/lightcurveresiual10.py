# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:14:44 2022

@author: dingxu
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
from scipy.stats import norm


def quantile(x, q, weights=None): 
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()
    

mpath = ''
dpath = ''
file = 'savedata0170T.txt'



model10mcmc = load_model(mpath+'model202.hdf5')


#data = np.loadtxt(dpath+file)
#np.random.shuffle(data)
#data = data[0:60000,:]
##np.savetxt('savedata01050TN.txt', data)
#np.savetxt('testdata01050.txt', data)
data = np.loadtxt('testdata01050.txt')

hang,lie = data.shape

datax = data[:,0:201]
for i in range(0, hang):
    datax[i,0:201] = -2.5*np.log10(datax[i,0:201])
    datax[i,0:201] = datax[i,0:201] - np.mean(datax[i,0:201])

datay = data[:,[201, 202, 203, 204, 205, 206, 207]]

predict1 = model10mcmc.predict(datay)
#predict1 = model10mcmc(datay)

plt.figure(0)

ax = plt.gca()
ax.plot(datax[0,0:201])
ax.plot(predict1[0],'.')
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)
print(data[0, 201:])
print(np.std(data[0, 0:201]-np.array(predict1[0])))

canchajuzhen = datax[:,0:201] - predict1

temp = []
for i in range(0, hang):
    residata = np.std(canchajuzhen[i,:])
    residata = np.round(residata, 5)
    temp.append(residata)



nptemp = np.array(temp)


num_bins = 1000
plt.figure(1)
#n, bins, patches = plt.hist(temp, num_bins, density=0, alpha=0.5)
n, bins, patches = plt.hist(temp, num_bins)
plt.xlim(-0.001,0.002)
plt.xlabel('mag',fontsize=18)
plt.ylabel('number',fontsize=18)
plt.savefig('resiualmagnol3.png')


q_16, q_50, q_84 = quantile(nptemp, [0.16, 0.5, 0.84])  
print(q_16, q_50, q_84)
plt.axvline(x = q_16, ls = "-", c = "green", linewidth=1)
plt.axvline(x = q_50, ls = "-", c = "green", linewidth=1)
plt.axvline(x = q_84, ls = "-", c = "green", linewidth=1)

plt.title(r'$\sigma=0.00032^{+0.00052}_{-0.00021}$')