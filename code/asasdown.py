#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scrapy
import urllib
import sys
import os
import numpy as np
import re
from openpyxl import load_workbook
#import goto
#from goto import with_goto
from dominate.tags import label
import requests
from bs4 import BeautifulSoup
import eventlet
#from goto import with_goto



item = 1
k = 0


start_url = 'https://asas-sn.osu.edu/variables?ra=&dec=&radius=0.5&vmag_min=&vmag_max=&amplitude_min=&amplitude_max=&period_min=&period_max=&lksl_min=&lksl_max=&class_prob_min=&class_prob_max=&parallax_over_err_min=&parallax_over_err_max=&name=&variable_type[]=EA&references[]=I&references[]=II&references[]=III&references[]=IV&references[]=V&references[]=IX&sort_by=raj2000&sort_order=asc&show_non_periodic=true&show_without_class=true&asassn_discov_only=false&'


startres = requests.get(start_url)

while item == 1: 
    startres.encoding = 'utf-8'
    startsoup = BeautifulSoup(startres.text, 'html.parser')
    #print(soup)
    tr = startsoup.find_all('tr')
    #print(tr)
    #print(len(tr))
    #print(tr[0])
    
    
    for i in range(len(tr)):
        next_url = 'https://asas-sn.osu.edu'+str(tr[i].a.get('href'))
        filename = re.split('/',next_url)[4]
        #print(filename)
        try:
            with eventlet.Timeout(10,False):#设置超时时间为10秒,超过10秒，跳出with模块
                nextres = requests.get(next_url)
                nextres.encoding = 'utf-8'
                nextsoup = BeautifulSoup(nextres.text, 'html.parser')
                #print(soup)
                a = nextsoup.select('a')
                #print(a) 
                #print(a[14].get('href'))
                final_url = 'https://asas-sn.osu.edu'+str(a[14].get('href'))
                #print(final_url)
                filepath = 'J:\\ASASEA\\'+str(filename)+'.csv'
                urllib.request.urlretrieve(final_url,filename = filepath)
                k = k+1
                print(filepath)
        except:
            print(next_url)
            continue
                
                
                
    li = startsoup.find_all('li')
    #print(li)
    #print(len(li))
    if li[len(li)-1].a.get('href') == 'javascript:void(0)':
        break
    else:
        link = str(li[len(li)-2].a.get('href'))
        sss = re.split('¶',link)
        nextpage = 'https://asas-sn.osu.edu'+sss[0]+'&para'+sss[1]+'&para'+sss[2]
        #print('nextpage:'+str(nextpage))
        start_url = nextpage
        #print(start_url)  
        try:
            with eventlet.Timeout(15,False):
                print(1)
                startres = requests.get(start_url)
                print(2)
                continue
                
            sss2 = re.split('page=',link)
            link2 = sss2[0]+str('page=')+str(int(sss2[1])+1)
            sss3 = re.split('¶',link2)
            nextnextpage = 'https://asas-sn.osu.edu'+sss3[0]+'&para'+sss3[1]+'&para'+sss3[2]
            #print(nextnextpage)   
            start_url = nextnextpage
            print(3)
            startres = requests.get(start_url)
            print(4)
            #print(start_url)
        
        except:
            sss2 = re.split('page=',link)
            link2 = sss2[0]+str('page=')+str(int(sss2[1])+1)
            sss3 = re.split('¶',link2)
            nextnextpage = 'https://asas-sn.osu.edu'+sss3[0]+'&para'+sss3[1]+'&para'+sss3[2]
            #print(nextnextpage)   
            start_url = nextnextpage
            print(5)
            startres = requests.get(start_url)
            print(6)
               
          
    if k == 100:
        break
    

