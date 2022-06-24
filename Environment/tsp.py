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
from Environment.problem import Problem
from Environment.ui import Show
import numpy as np
import random
    
class TSP(Problem):
    def __init__(self,problem_file):
        super().__init__(problem_file)
        self.data_precess()
        self.s=Show()
        self.show=self.s.run
        
    def data_precess(self):
        data=self.info['data']
        cities=np.array([[x,y] for _,x,y in data])
        # x_max,y_max=cities.max(axis=0)
        # x_min,y_min=cities.min(axis=0)
        self.cities=cities
        normalized_cities=(cities-cities.min(axis=0))/(cities.max(axis=0)-cities.min(axis=0))#/np.array([x_max-x_min,y_max-y_min])
        self.normalized_cities=normalized_cities
        self.size=len(self.cities)
        return normalized_cities

    def show_cities(self):
        self.s.plot(self.normalized_cities)
        
        
        
    def show_path_with_cities(self,cities):
        self.s.lines(cities)
        
    def show_path_with_index(self,index):
        cities=self.normalized_cities[index]
        self.show_path_with_cities(cities)
        
    def calculate_with_cities(self,cities):
        res=0
        for i in range(self.size)-1:
            res+=np.sum((cities[i]-cities[i+1])**2)**0.5
        res+=np.sum((cities[-1]-cities[0])**2)**0.5
        return res
    
    def calculate_with_index(self,index):
        cities=self.cities[index]
        return self.calculate_with_cities(cities)


if __name__ == '__main__':
    p=TSP('Benchmark/TSP/tsp/a280.tsp')
    per=np.random.permutation(p.size)
    p.show_path_with_index(per)
    p.show_cities()
    p.show()
        