# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 16:33:31 2022

@author: cqy
"""

from ztfquery import query
import csv
import pandas as pd
from ztfquery import lightcurve

'''
Ras = []
Des = []

# 从match_zhang_ew_10.csv 读取RA DEC
with open('C:/Users/cqy/Documents/physics/mywork/ZTF/match_zhang_ew_10.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        Ras.append(row[36])
        Des.append(row[37])

Ras.remove('col3')
Des.remove('col4')

zquery = query.ZTFQuery()

zquery.load_metadata(radec=[Ras[0],Des[0]])

zquery.download_data("psfcat.fits", show_progress=True, nprocess=4, verbose=True, overwrite=True)
'''
file = 'testcsv.csv'
df = pd.read_csv(file)
path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ZTF\\code\\chengdata\\'
hang ,lie = df.shape

for i in range(0,hang):
    RA = df.iloc[i,3]
    DEC = df.iloc[i,4]
    print(RA,DEC)
    lcq = lightcurve.LCQuery.from_position(RA, DEC, 1)
    data = lcq.data
    data.to_csv(path+df.iloc[i,1]+'.csv')
