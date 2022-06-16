# -*- encoding: utf-8 -*-
'''
@File    :   eaLSTM.py
@Time    :   2022/04/13 16:25:07
@Author  :   Muyao ZHONG 
@Version :   1.0
@Contact :   zmy125616515@hotmail.com
@License :   (C)Copyright 2019-2020
@Title   :   
'''

import random
from time import sleep

import numpy as np
import math

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    return values*(1-values)

def tanh_derivative(values): 
    return 1. - values ** 2

# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args): 
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class LstmParam:
    def __init__(self, embedding_dim, hidden_dim,n_layers,drop_out,bidirectional):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        concat_len = embedding_dim + hidden_dim
        self.n_layers=n_layers
        self.drop_out=drop_out
        self.bidirectional=bidirectional
        
        self.wg=[]
        self.wi=[]
        self.wf=[]
        self.wo=[]
        self.bg=[]
        self.bi=[]
        self.bf=[]
        self.bo=[]
        
        self.rwg=[]
        self.rwi=[]
        self.rwf=[]
        self.rwo=[]
        self.rbg=[]
        self.rbi=[]
        self.rbf=[]
        self.rbo=[]
       
        self.init_state()
        
        for i in range(n_layers):
                
            # weight matrices
            self.wg.append( rand_arr(-0.1, 0.1, hidden_dim, concat_len))
            self.wi.append( rand_arr(-0.1, 0.1, hidden_dim, concat_len))
            self.wf.append( rand_arr(-0.1, 0.1, hidden_dim, concat_len))
            self.wo.append( rand_arr(-0.1, 0.1, hidden_dim, concat_len))
            # bias terms
            self.bg.append( rand_arr(-0.1, 0.1, hidden_dim))  
            self.bi.append( rand_arr(-0.1, 0.1, hidden_dim)) 
            self.bf.append( rand_arr(-0.1, 0.1, hidden_dim)) 
            self.bo.append( rand_arr(-0.1, 0.1, hidden_dim)) 
            
            if self.bidirectional:
                # weight matrices
                self.rwg.append( rand_arr(-0.1, 0.1, hidden_dim, concat_len))
                self.rwi.append( rand_arr(-0.1, 0.1, hidden_dim, concat_len))
                self.rwf.append( rand_arr(-0.1, 0.1, hidden_dim, concat_len))
                self.rwo.append( rand_arr(-0.1, 0.1, hidden_dim, concat_len))
                # bias terms
                self.rbg.append( rand_arr(-0.1, 0.1, hidden_dim))  
                self.rbi.append( rand_arr(-0.1, 0.1, hidden_dim)) 
                self.rbf.append( rand_arr(-0.1, 0.1, hidden_dim)) 
                self.rbo.append( rand_arr(-0.1, 0.1, hidden_dim)) 
                
                
    def init_state(self):
        self.g=[]
        self.i=[]
        self.f=[]
        self.o=[]
        self.s=[]
        self.h=[]
        self.bottom_diff_h=[]
        self.bottom_diff_s=[]
        
        self.rg=[]
        self.ri=[]
        self.rf=[]
        self.ro=[]
        self.rs=[]
        self.rh=[]
        self.bottom_diff_rh=[]
        self.bottom_diff_rs=[]
        
        for i in range(self.n_layers):
            self.g.append(np.zeros(self.hidden_dim))
            self.i.append(np.zeros(self.hidden_dim))
            self.f.append(np.zeros(self.hidden_dim))
            self.o.append(np.zeros(self.hidden_dim))
            
            self.s.append(np.zeros(self.hidden_dim))
            self.h.append(np.zeros(self.hidden_dim))
            
            self.bottom_diff_h.append(np.zeros_like(self.h))
            self.bottom_diff_s.append(np.zeros_like(self.s))
            
            if self.bidirectional:
                self.rg.append(np.zeros(self.hidden_dim))
                self.ri.append(np.zeros(self.hidden_dim))
                self.rf.append(np.zeros(self.hidden_dim))
                self.ro.append(np.zeros(self.hidden_dim))
                self.rs.append(np.zeros(self.hidden_dim))
                self.rh.append(np.zeros(self.hidden_dim))
                self.bottom_diff_rh.append(np.zeros_like(self.h))
                self.bottom_diff_rs.append(np.zeros_like(self.s))
        
    def forward_layer(self,x,layer,s_prev=None,h_prev=None):
        # if this is the first lstm node in the network
        if s_prev is None: s_prev = np.zeros_like(self.s[layer])
        if h_prev is None: h_prev = np.zeros_like(self.h[layer])
       
         # concatenate x(t) and h(t-1)
        xc = np.hstack((x, h_prev))
        self.g[layer] = np.tanh(np.dot(self.wg[layer], xc) + self.bg[layer])
        self.i[layer] = sigmoid(np.dot(self.wi[layer], xc) + self.bi[layer])
        self.f[layer] = sigmoid(np.dot(self.wf[layer], xc) + self.bf[layer])
        self.o[layer] = sigmoid(np.dot(self.wo[layer], xc) + self.bo[layer])
        self.s[layer] = self.g[layer] * self.i[layer] + s_prev * self.f[layer]
        self.h[layer] = self.s[layer] * self.o[layer]

class Embedding:
    def __init__(self, input_dim, embedding_dim):
        self.input_dim=input_dim
        self.embedding_dim= embedding_dim
        self.W=rand_arr(-0.1, 0.1, embedding_dim, input_dim)
        self.b=rand_arr(-0.1, 0.1, embedding_dim)
        

class Encoder:
    """Encoder class for Pointer-Net

    Returns:
        Encoder: 
    """
    
    def __init__(self,embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        
        self.h0 = 0.0
        self.c0 = 0.0
        

        
        

