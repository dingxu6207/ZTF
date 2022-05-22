# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:09:43 2022

@author: dingxu
"""

import time
from threading import Thread
import phoebe
import numpy as np
import matplotlib.pyplot as plt
import random

logger = phoebe.logger(clevel = 'WARNING')
b = phoebe.default_binary()

times  = np.linspace(0,1,800)
b.add_dataset('lc', times=times, passband= 'TESS:T')
b['period@binary'] = 1
b['sma@orbit'] = 1
#count = 0
def datacreate(startdata):
    for count in range(startdata, 30000000000):
        try:
            incl = random.uniform(70,90)
            T1 = np.random.randint(4000, 8000)
            T1divT2 = random.uniform(0.5,1)
            q = np.random.uniform(0, 1)
            r1 = random.uniform(0.02,0.487)
            r1r2 = random.uniform(0,1.5)
            ecc = random.uniform(0,0.1)
        
            print('incl=', incl)
            print('temp1=', T1)
            print('temp2=', T1*T1divT2)
            print('r1=', r1)
            print('r2=', r1*r1r2)
            print('ecc=', ecc)
            print('count = ', count)
            print('q = ', q)
        
        
            b['incl@binary'] = incl
            b['q@binary'] = q
            b['teff@primary'] = T1
            b['teff@secondary'] = T1*T1divT2
            b['requiv@primary'] = r1
            if r1*r1r2<0.02:
                continue
            else:
                b['requiv@secondary'] = r1*r1r2
                b['ecc@binary'] = ecc
        
            b.run_compute(irrad_method='none')
            print('it is ok1')
    
            m = count+1
            print('m = ', m)
            file = str(m)+'.lc'
            lightcurvedata = np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T
            mq = [(T1/8000,T1divT2), (incl/90, q), (r1, r1r2), (ecc,0)]
            datamq = np.array(mq)
            print('it is ok2')
    
            resultdata = np.row_stack((lightcurvedata, datamq))
            np.savetxt(file, resultdata)
        
            print('it is ok3')
                    
        except:
            print('it is error!')
    
    




if __name__ == '__main__':
    t1 = Thread(target=datacreate, args=(10000,))
    t2 = Thread(target=datacreate, args=(20000,))
    t3 = Thread(target=datacreate, args=(30000,))
    t4 = Thread(target=datacreate, args=(40000,))
    t5 = Thread(target=datacreate, args=(50000,))
    t6 = Thread(target=datacreate, args=(60000,))
    t7 = Thread(target=datacreate, args=(70000,))
    t8 = Thread(target=datacreate, args=(80000,))
    t9 = Thread(target=datacreate, args=(90000,))
    t10 = Thread(target=datacreate, args=(100000,))
    t11 = Thread(target=datacreate, args=(110000,))
    t12 = Thread(target=datacreate, args=(120000,))
    t13 = Thread(target=datacreate, args=(130000,))
    t14 = Thread(target=datacreate, args=(140000,))
    t15 = Thread(target=datacreate, args=(150000,))
    t16 = Thread(target=datacreate, args=(160000,))
    t17 = Thread(target=datacreate, args=(170000,))
    t18 = Thread(target=datacreate, args=(180000,))
    t19 = Thread(target=datacreate, args=(190000,))
    t20 = Thread(target=datacreate, args=(2000000,))
    t21 = Thread(target=datacreate, args=(2100000,))
    t22 = Thread(target=datacreate, args=(2200000,))
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()
    t10.start()
    t11.start()
    t12.start()
    t13.start()
    t14.start()
    t15.start()
    t16.start()
    t17.start()
    t18.start()
    t19.start()
    t20.start()
    t21.start()
    t22.start()
    
    print("主线程结束")
