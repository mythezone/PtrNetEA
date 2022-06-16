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

import tkinter as tk
from tkinter import *
from tkinter import ttk 

class Show:
    def __init__(self,root=None,canvas=None):
        if root==None:
            self.root=tk.Tk()
            self.root.geometry('1000x1000')

        else:
            self.root=root
            self.canvas_size=self.root.width()
            
        self.canvas_size=900
        if canvas==None:
            
            self.canvas=Canvas(self.root,width=self.canvas_size,height=self.canvas_size)
        else:
            self.canvas=canvas
            
        self.canvas.pack()
        
    def plot(self,points,size=4,fill='blue',outline='black',width=1):
        points=np.around(points*800)+5
        for point in points:
            self.canvas.create_oval(point[0]-size//2,point[1]-size//2,point[0]+size//2,point[1]+size//2,fill=fill,outline=outline,width=width)
        
    def lines(self,points,fill='green',width=2,arrow='first',end2start=True):
        points=np.around(points*800)+5
        # args=[]
        # for i in range(len(points)):
        #     p=points[i]
        #     args.extend(p)
        # if end2start:
        #     args.extend(points[0])
        if arrow!=None:
            if arrow=="first":
                arrow=tk.FIRST
            elif arrow=="last":
                arrow=tk.LAST
            else:
                arrow=tk.BOTH
        for i in range(len(points)):
            p=points[i]
            if i==len(points)-1:
                p1=points[0]
            else:
                p1=points[i+1]
            self.canvas.create_line(p[0],p[1],p1[0],p1[1],fill=fill,width=width,arrow=arrow)
        
        
    def run(self,root=None):
        if root==None:
            self.root.mainloop()
        else:
            return self.root
    
        