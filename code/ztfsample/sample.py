import phoebe
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm_notebook as tqdm
import random

#warnings.filterwarnings('ignore')
logger = phoebe.logger(clevel = 'WARNING')

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,100)
b.add_dataset('lc', times=times, passband= 'LSST:r')

b['period@binary'] = 1
b['sma@orbit'] = 1
b.flip_constraint('pot', solve_for='requiv@primary')
b.flip_constraint('fillout_factor', solve_for='pot')
m = 0


for count in range(0,100000000000):
    try:
        incl = random.uniform(50,90)
        T1 = np.random.randint(4000, 8000)
        T1divT2 = random.uniform(0.8,1.2)
        q = np.random.uniform(0, 10)
        f = random.uniform(0,1)
        
        print('incl=', incl)
        print('temp1=', T1)
        print('temp2=', T1*T1divT2)
        print('q=', q)
        print('f=', f)
        print('count = ', count)
        
        b['fillout_factor'] = f
        b['incl@binary'] = incl
        b['q@binary'] = q
        b['teff@primary'] = T1
        b['teff@secondary'] = T1*T1divT2
        
        b.run_compute(irrad_method='none')
        print('it is ok1')
    
        m = m+1
        print('m = ', m)
        file = str(m)+'.lc'
        lightcurvedata = np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T
        mq = [(T1/5850,0), (incl, q), (f, T1divT2)]
        datamq = np.array(mq)
        print('it is ok2')
    
        resultdata = np.row_stack((lightcurvedata, datamq))
        np.savetxt(file, resultdata)
        
        print('it is ok3')
        
    except:
         print('it is error!')
        
        

