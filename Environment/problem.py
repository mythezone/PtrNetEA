# -*- encoding: utf-8 -*-
'''
@File    :   tsp.py
@Time    :   2022/06/16 23:05:00
@Author  :   Martoriay 
@Version :   1.0
@Contact :   martoriay@protonmail.com
@License :   (C)Copyright 2021-2022
@Title   :   TSP
'''

import sys
import os
sys.path.append(os.path.abspath('./'))
import numpy as np 

class Problem:
    def __init__(self,problem_file):
        self.problem_file=problem_file
        self.info={}
        self.get_info()

        
    def get_info(self):
        with open(self.problem_file,'r') as f:
            tmp=f.readlines()
        for l in tmp:
            l=l.strip()
            if ':' in l:
                k,v=l.split(':')
                self.info[k.strip()]=v.strip()
            elif l.endswith('SECTION'):
                k='data'
                self.info['data']=[]
            elif l=='EOF':
                break
            else:
                data=l.split(' ')
                data=[int(i.strip())  for i in data if i !=""]
                self.info['data'].append(data)
    
    def data_precess(self):
        print("Please override this method.")
        pass