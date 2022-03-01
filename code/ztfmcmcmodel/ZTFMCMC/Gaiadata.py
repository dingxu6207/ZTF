# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:34:01 2022

@author: dingxu
"""

from astroquery.gaia import Gaia
import numpy as np
import matplotlib.pyplot as plt

Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"

import astropy.units as u

from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

Gaia.ROW_LIMIT = 1000000 

coord = SkyCoord(ra=0, dec=0, unit=(u.degree, u.degree), frame='icrs')

width = u.Quantity(20, u.deg)

height = u.Quantity(20, u.deg)


r = Gaia.query_object_async(coordinate=coord, width=width, height=height)


temp = np.transpose((r['parallax_over_error'], r['visibility_periods_used'],
                     r['parallax'],r['phot_bp_mean_mag'],r['phot_rp_mean_mag'],
                     r['phot_g_mean_mag'] , r['teff_val'],
                     r['phot_g_mean_flux_over_error'],r['phot_bp_mean_flux_over_error'],
                     r['phot_rp_mean_flux_over_error'],r['phot_bp_rp_excess_factor']
                     ))



#phot_g_mean_mag+5*log10(parallax)-10
temp = temp[temp[:,2]>5] 
temp = temp[temp[:,0]>10]
temp = temp[temp[:,1]>8]
temp = temp[temp[:,7]>50]
temp = temp[temp[:,8]>20]
temp = temp[temp[:,9]>20]

yuzhi1 = 1.3+0.06*np.power(temp[:,3]-temp[:,4],2)
yuzhi2 = 1.0+0.015*np.power(temp[:,3]-temp[:,4],2)

temp = temp[temp[:,10]>yuzhi2]
temp = temp[temp[:,10]<yuzhi1]



#temp = temp[temp[:,2] != 0] 
#temp = temp[~np.isnan(temp)]

np.savetxt('temp.txt', temp)

M = temp[:,5]+5*np.log10(temp[:,2])-10

plt.figure(3)
highdataGmag = M
highdataBPRP = temp[:,3]-temp[:,4]

plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=0.5)

plt.xlabel('BP-RP',fontsize=14)
#plt.xlabel('G-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()

ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向









