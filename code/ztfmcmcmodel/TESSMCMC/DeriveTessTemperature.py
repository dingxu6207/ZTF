# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:51:51 2022

@author: dingxu
"""
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
import numpy as np
import matplotlib.pyplot as plt

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\ztfmcmcmodel\\TESSMCMC\\EWDATA\\'
file = 'EWINFO.csv'

temp = []
data = pd.read_csv(path+file)
hang,lie = data.shape


radectemp = np.loadtxt('radectemparature.txt')
temperature = radectemp[:,5]
listtemp = temperature.tolist()

data['tempature'] = listtemp

data.to_csv('EMTEMP.csv',index=0) #不保存行索引
data = data[data['tempature'] != 0] 
plt.hist(np.around(data.iloc[:,6],4), bins=100, density = 0, facecolor='blue', alpha=0.5)


'''
for i in range (0, hang):
    radata = data.iloc[i,2]
    decdata = data.iloc[i,3]
    coord = SkyCoord(ra = radata, dec = decdata, unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(5/3600, u.deg)
    height = u.Quantity(5/3600, u.deg)
    
    r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
    radectemp = np.transpose((r['ra'], r['dec'], r['phot_g_mean_mag'],  r['phot_bp_mean_mag'], r['phot_rp_mean_mag'], r['teff_val']))
    
    try:
        datatemp = [radata, decdata, radectemp[0][2], radectemp[0][3], radectemp[0][4], radectemp[0][5]]
    except:
        try:
            datatemp = [radata, decdata, radectemp[0][2], radectemp[0][3], radectemp[0][4], 0]
        except:
            datatemp = [radata, decdata, 0, 0, 0, 0]
    temp.append(datatemp)
    
    print(datatemp)
    print('it is ok '+str(i))
    
nptemp = np.array(temp)
np.savetxt('radectemparature5.txt', nptemp)
'''