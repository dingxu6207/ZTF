# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 12:30:40 2021

@author: dingxu
"""

import numpy as np
import pandas as pd

data = np.genfromtxt('Table2data.txt',dtype=str)
dataEW = data[data[:,24]=='EW']
dataEW10 = dataEW[np.float32(dataEW[:,4])>=10]

name=['ID','SourceID','RAdeg','DEdeg','Per','R21','phi21','T0','gmag','rmag','Per_g','Per_r','Num_g','Num_r',
      'R21_g','R21_r','phi21_g','phi21_r','R2_g','R2_r','Amp_g','Amp_r','log(FAP_g)','log(FAP_r)','Type','Dmin_g','Dmin_r']
listdataEW10 = dataEW10.tolist()
test = pd.DataFrame(columns=name, data=listdataEW10)
test.to_csv('testcsv.csv')