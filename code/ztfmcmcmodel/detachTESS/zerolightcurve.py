# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:23:48 2022

@author: dingxu
"""

import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
from astropy.timeseries import LombScargle
import shutil
from tensorflow.keras.models import load_model
from scipy.fftpack import fft,ifft
import winsound
from scipy import interpolate
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia


def readfits(fits_file):
    with fits.open(fits_file, mode="readonly") as hdulist:
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
        print(hdulist[0].header['OBJECT'])
        print(hdulist[0].header['RA_OBJ'], hdulist[0].header['DEC_OBJ'])
        
        indexflux = np.argwhere(pdcsap_fluxes > 0)
#        print(sap_fluxes)
        time = tess_bjds[indexflux]
        time = time.flatten()
        flux = pdcsap_fluxes[indexflux]
        flux =  flux.flatten()
        RA = hdulist[0].header['RA_OBJ']
        DEC = hdulist[0].header['DEC_OBJ']
        
        return time, flux, RA, DEC

def zerophse(phases, resultmag):
    listmag = resultmag.tolist()
    listmag.extend(listmag)
    listphrase = phases.tolist()
    listphrase.extend(listphrase+np.max(1))
    
    nplistmag = np.array(listmag)
    sortmag = np.sort(nplistmag)
    maxindex = np.median(sortmag[-1:])
    indexmag = listmag.index(maxindex)
    nplistphrase = np.array(listphrase)
    nplistphrase = nplistphrase-nplistphrase[indexmag]
    nplistmag = np.array(listmag)
    
    phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
    
    return phasemag

def computeperiod(JDtime, targetflux):
   
    ls = LombScargle(JDtime, targetflux, normalization='model')
    frequency, power = ls.autopower(minimum_frequency=0.01,maximum_frequency=40)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower

def pholddata(per, times, fluxes):
    mags = -2.5*np.log10(fluxes)
    mags = mags-np.mean(mags)
    
    lendata =  int((per/26)*1.1*len(times))
     
    time = times[0:lendata]
    mag = mags[0:lendata]
    phases = foldAt(time, per)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = mag[sortIndi]
    return phases, resultmag

def stddata(timedata, fluxdata, P):
    yuanflux = np.copy(fluxdata)
    yuanmag = -2.5*np.log10(yuanflux)
    
    phases, resultmag = pholddata(P, timedata, fluxdata)
    datamag = np.copy(resultmag)
    datanoise = np.diff(datamag,2).std()/np.sqrt(6)
    stddata = np.std(yuanmag)
    return stddata/datanoise


path = 'J:\\EA\\' 
file = 'tess2018206045859-s0001-0000000278706358-0120-s_lc.fits'

tbjd, fluxes, RA, DEC = readfits(path+file)
coord = SkyCoord(ra = RA, dec = DEC, unit=(u.degree, u.degree), frame='icrs')
width = u.Quantity(5/3600, u.deg)
height = u.Quantity(5/3600, u.deg)

r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
radectemp = np.transpose((r['ra'], r['dec'], r['teff_val']))
print(radectemp)


comper, wrongP, maxpower = computeperiod(tbjd, fluxes)
stdodata1 = stddata(tbjd, fluxes, comper)
stdodata2 = stddata(tbjd, fluxes, comper*2)

phases, resultmag = pholddata(comper*2, tbjd, fluxes)
phasemag = zerophse(phases, resultmag)


np.savetxt('phasemag.txt', phasemag)

plt.figure(0)
plt.plot(phases, resultmag, '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14) 
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.figure(1)
plt.plot(phasemag[:,0], phasemag[:,1], '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14) 
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.figure(2)
plt.plot(tbjd, fluxes, '.')
plt.xlabel('tbjd',fontsize=14)
plt.ylabel('FLUX',fontsize=14) 