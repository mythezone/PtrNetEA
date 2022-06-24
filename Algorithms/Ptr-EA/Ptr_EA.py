#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Ptr-EA.py
@Time    :   2022/06/20 16:10:32
@Author  :   Mythezone 
@Version :   1.0
@Contact :   zmy125616515@hotmail.com
@License :   (C)Copyright 2021-2022
'''

# content : 
from unicodedata import bidirectional
import torch
import torch.nn as nn 
from torch.nn import Parameter 
import torch.nn.functional as F

class PEncoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        super(PEncoder,self).__init__()
        self.hidden_dim=hidden_dim//2 if bidir else hidden_dim
        self.n_layers=n_layers*2 if bidir else n_layers
        self.bidir=bidir
        self.lstm=nn.LSTM(embedding_dim,
                          self.hidden_dim,
                          self.n_layers,
                          dropout=dropout,
                          bidirectional=bidir)

        self.h0=Parameter(torch.zeros(1),requires_grad=False)
        self.c0=Parameter(torch.zeros(1),requires_grad=False)
        
class PAttention(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(PAttention,self).__init__()
        
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        
        self.input_layer=nn.Linear(input_dim,hidden_dim)
        self.context_linear=nn.Conv1d(input_dim,hidden_dim,1,1)
        self.V=Parameter(torch.FloatTensor(hidden_dim),requires_grad=True)
        self._inf=Parameter(torch.FloatTensor([float('-inf')]),requires_grad=False)
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax()
        
        nn.init.uniform(self.V,-1,1)

class PDecoder(nn.Module):
    def __init__(self,embedding_dim,hidden_dim):
        super(PDecoder,self).__init__()
        
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        
        self.input_to_hidden=nn.Linear(embedding_dim,4*hidden_dim)
        self.hidden_to_hidden=nn.Linear(hidden_dim,4*hidden_dim)
        self.hidden_out=nn.Linear(hidden_dim*2,hidden_dim)
        self.att=PAttention(hidden_dim,hidden_dim)

        self.mask=Parameter(torch.ones(1),requires_grad=False)
        self.runner=Parameter(torch.zeros(1),requires_grad=False)
        

        
class EAFrame:
    """Evolutionary Algorithm Framework:
    1. Population Initiation:
    2. Crossover Operation:
    3. Mutation Operation:
    4. Selection Operation:
    5. Evaluation
    :param pop_size: The population size of the algorithm.
    
    """
    def __init__(self,pop_size):
        pass 
    
    def init_pop(self):
        pass

    def selector(self):
        pass

    def crossover(self,p1,p2):
        pass 
    
    def mutation(self,p):
        pass 
    
    def eval(self):
        pass 
    
    

        