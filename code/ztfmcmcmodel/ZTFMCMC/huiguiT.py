# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:26:31 2021

@author: dingxu
"""

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

'''#flunction1
data = np.genfromtxt('Table2data.txt',dtype=str)
dataEW = data[data[:,24]=='EW']

savedata = dataEW[:,[2,3,8,9]]

floatdata = np.float32(savedata)
tsavedata = np.savetxt('radecmag.txt', floatdata)
'''

radecmag = np.loadtxt('radecmag.txt')
hang,lie = radecmag.shape

coord = SkyCoord(ra = radecmag[1,0], dec = radecmag[1,1], unit=(u.degree, u.degree), frame='icrs')
width = u.Quantity(1/3600, u.deg)
height = u.Quantity(1/3600, u.deg)

r = Gaia.query_object_async(coordinate=coord, width=width, height=height)

#positions = np.transpose((r['ra'], r['dec'], r['dr2_rv_template_teff'], r['dr2_rv_template_fe_h']))
radectemp = np.transpose((r['ra'], r['dec'], r['teff_val']))

temp = []
for i in range (0, 100000):
    coord = SkyCoord(ra = radecmag[i,0], dec = radecmag[i,1], unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(1/3600, u.deg)
    height = u.Quantity(1/3600, u.deg)
    
    if radecmag[i,2]>18:
        continue
    
    try:
        r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
        radectemp = np.transpose((r['ra'], r['dec'], r['teff_val']))
        datatemp = [radecmag[i,2], radecmag[i,3], radectemp[0][2]]
        temp.append(datatemp)
        print(str(i)+':')
        print(datatemp)
    except:
        try:
            r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
            radectemp = np.transpose((r['ra'], r['dec'], r['teff_val']))
            datatemp = [radecmag[i,2], radecmag[i,3], radectemp[0][2]]
            temp.append(datatemp)
            print(str(i)+':')
            print(datatemp)
        except:
            continue
             
    
    
nptemp = np.array(temp)

np.savetxt('radectemparature.txt', nptemp)

    
    