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
import sys
import os
sys.path.append(os.path.abspath('./'))

from unicodedata import bidirectional
import torch
import torch.nn as nn 
from torch.nn import Parameter 
import torch.nn.functional as F

from torch.utils.data import DataLoader
import numpy as np 
from tqdm import tqdm
from torch.autograd import Variable
import random


from Data_Generator import TSPDataset


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
        
    def forward(self,embedded_inputs,hidden):
        embedded_inputs=embedded_inputs.permute(1,0,2)
        outputs,hidden=self.lstm(embedded_inputs,hidden)
        return outputs.permute(1,0,2),hidden

    def init_hidden(self,embedded_inputs):
        batch_size=embedded_inputs.size(0)
        h0=self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,batch_size,self.hidden_dim)
        c0=self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,batch_size,self.hidden_dim)
        return h0,c0 
    
        
class PAttention(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(PAttention,self).__init__()
        
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        
        self.input_linear=nn.Linear(input_dim,hidden_dim)
        self.context_linear=nn.Conv1d(input_dim,hidden_dim,1,1)
        self.V=Parameter(torch.FloatTensor(hidden_dim),requires_grad=True)
        self._inf=Parameter(torch.FloatTensor([float('-inf')]),requires_grad=False)
        self.tanh=nn.Tanh()
        
        
        nn.init.uniform_(self.V,-1,1)
        
    def forward(self,input,context,mask):
        self.softmax=nn.Softmax(dim=1)
        inp=self.input_linear(input).unsqueeze(2).expand(-1,-1,context.size(1))

        context=context.permute(0,2,1)
        ctx=self.context_linear(context)

        V=self.V.unsqueeze(0).expand(context.size(0),-1).unsqueeze(1)

        att=torch.bmm(V,self.tanh(inp+ctx)).squeeze(1)
        if len(att[mask])>0:
            att[mask]=self.inf[mask]
        alpha=self.softmax(att)

        hidden_state=torch.bmm(ctx,alpha.unsqueeze(2)).squeeze(2)
        return hidden_state,alpha

    def init_inf(self,mask_size):
        self.inf=self._inf.unsqueeze(1).expand(*mask_size)


class PDecoder(nn.Module):
    def __init__(self,embedding_dim,hidden_dim):
        super(PDecoder,self).__init__()
        
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        
        self.input_to_hidden=nn.Linear(embedding_dim,4*hidden_dim)
        self.hidden_to_hidden=nn.Linear(hidden_dim,4*hidden_dim)
        self.hidden_out=nn.Linear(hidden_dim*2,hidden_dim)
        self.att=PAttention(hidden_dim,hidden_dim)
        
        # self.mask=Parameter(torch.ones(1),requires_grad=False)
        # self.runner=Parameter(torch.zeros(1),requires_grad=False)
        
    def forward(self,embedded_inputs,decoder_input,hidden,context):
        batch_size=embedded_inputs.size(0)
        input_lenght=embedded_inputs.size(1)
        
        self.mask=torch.ones(1)
        self.runner=torch.zeros(1)

        mask=self.mask.repeat(input_lenght).unsqueeze(0).repeat(batch_size,1)
        self.att.init_inf(mask.size())

        runner=self.runner.repeat(input_lenght)
        for i in range(input_lenght):
            runner.data[i]=i 
            
        runner=runner.unsqueeze(0).expand(batch_size,-1).long()

        outputs=[]
        pointers=[]
        
        def step(x,hidden):
            h,c=hidden
            
            gates=self.input_to_hidden(x)+self.hidden_to_hidden(h)
            input,forget,cell,out=gates.chunk(4,1)

            input=torch.sigmoid(input)
            forget=torch.sigmoid(forget)
            cell=torch.tanh(cell)
            out=torch.sigmoid(out)

            c_t=(forget*c)+(input*cell)
            h_t=out*torch.tanh(c_t)

            hidden_t,output=self.att(h_t,context,torch.eq(mask,0))
            hidden_t=torch.tanh(self.hidden_out(torch.cat((hidden_t,h_t),1)))

            return hidden_t,c_t,output

        for _ in range(input_lenght):
            h_t,c_t,outs=step(decoder_input,hidden)
            hidden=(h_t,c_t)

            masked_outs=outs*mask

            max_probs,indices=masked_outs.max(1)
            one_hot_pointers=(runner==indices.unsqueeze(1).expand(-1,outs.size()[1])).float()
            mask=mask*(1-one_hot_pointers)

            embedding_mask=one_hot_pointers.unsqueeze(2).expand(-1,-1,self.embedding_dim).byte()
            decoder_input=embedded_inputs[embedding_mask.data.bool()].view(batch_size,self.embedding_dim)
            
            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs=torch.cat(outputs).permute(1,0,2)
        pointers=torch.cat(pointers,1)
        return (outputs,pointers),hidden


        
class PtrNet(nn.Module):
    def __init__(self,embedding_dim,
                 hidden_dim,
                 lstm_layters=1,
                 dropout=0,
                 bidir=False):
        super(PtrNet,self).__init__()
        self.embedding_dim=embedding_dim
        self.bidir=bidir
        self.hidden_dim=hidden_dim
        self.embedding=nn.Linear(2,embedding_dim)
        self.encoder=PEncoder(embedding_dim,hidden_dim,lstm_layters,dropout,bidir)
        self.decoder=PDecoder(embedding_dim,hidden_dim)
        self.decoder_input0=Parameter(torch.FloatTensor(embedding_dim),requires_grad=False)

        nn.init.uniform_(self.decoder_input0,-1,1)
        
    def forward(self,inputs):
        batch_size=inputs.size(0)
        input_length=inputs.size(1)

        decoder_input0=self.decoder_input0.unsqueeze(0).expand(batch_size,-1)

        inputs=inputs.view(batch_size*input_length,-1)
        embedded_inputs=self.embedding(inputs).view(batch_size,input_length,-1)

        encoder_hidden0=self.encoder.init_hidden(embedded_inputs)
        encoder_outputs,encoder_hidden=self.encoder(embedded_inputs,encoder_hidden0)

        if self.bidir:
            decoder_hidden0=(torch.cat(tuple(encoder_hidden[0][-2:]),dim=-1),torch.cat(tuple(encoder_hidden[1][-2:])))
        else:
            decoder_hidden0=(encoder_hidden[0][-1],encoder_hidden[1][-1])

        (outputs,pointers),decoder_hidden=self.decoder(embedded_inputs,decoder_input0,decoder_hidden0,encoder_outputs)

        return outputs,pointers

class EAFrame:
    """Evolutionary Algorithm Framework:
    1. Population Initiation:
    2. Crossover Operation:
    3. Mutation Operation:
    4. Selection Operation:
    5. Evaluation
    :param pop_size: The population size of the algorithm.
    
    """
    def __init__(self,pop_size=10,iteration=1000,embedding_dim=64,hidden_dim=64,n_layers=1,dropout=0,bidir=False):
        self.pop_size=pop_size
        self.iteration=iteration
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.n_layers=n_layers
        self.dropout=dropout
        self.bidir=bidir
        self.values=None 
        

    def gen_individual(self):
        return PtrNet(self.embedding_dim,self.hidden_dim,self.n_layers,self.dropout,self.bidir)
    
    def init_pop(self,inputs):
        self.pop=[self.gen_individual() for _ in range(self.pop_size)]
        self.values=self.eval_all(inputs,self.pop)

    def select(self,inputs,models,num):
        values=self.eval_all(inputs,models)
        rank=values.argsort()[:num]
        return models[rank]
    
    def select_by_value(self,values,num):

        nns=nn.Softmax(dim=0)
        values_s=torch.Tensor(values)
        tmp_prob=nns(values_s)
        prob=torch.zeros_like(tmp_prob)
        for i in range(len(tmp_prob)):
            if i==0:
                prob[i]=tmp_prob[i]
            else:
                prob[i]=prob[i-1]+tmp_prob[i]

        res_index=[]
        for i in range(num):
            r=random.random()
            x=0
            while r>prob[x]:
                x+=1
            res_index.append(x)
        return res_index
        
    def select_next(self,values):
        rank=values.argsort()[:self.pop_size]
        return self.pop[rank]

    def crossover(self,m1,m2,cross_probability=0.5):
        p1=m1.state_dict()
        p2=m2.state_dict()
        p={}
        for key in p1.keys():
            if random.random()<cross_probability:
                # p[key]=(p1[key]+p2[key])/2
                p[key]=p2[key]
            else:
                p[key]=p1[key]
        m=PtrNet(self.embedding_dim,self.hidden_dim,self.n_layers,self.dropout,self.bidir)
        m.load_state_dict(p)
        return m
    
    def mutation(self,m,mutation_probability=0.8):
        p=m.state_dict()
        for key in p.keys():
            if random.random()<mutation_probability:
                rp=np.random.randn(*p[key].size())*2
                mask=np.random.rand(*p[key].size())<0.5
                p[key]+=rp*mask

        m.load_state_dict(p)
        return m
    
    def evaluate(self,inputs,model,distances=None):
        if distances is None:
            distances=torch.zeros((inputs.size(0),inputs.size(1),inputs.size(1)))
            for j in range(len(inputs)):
                input=inputs[j]
                for i in range(len(input)):
                    tmp=((input-input[i])**2).sum(axis=1)**0.5
                    distances[j][i]=tmp
                
        results=0.
                
        _,pointers=model(inputs)
        # if list(pointers[0]).count(0)>1:
        #     print("Error in model pointer...")
        for i in range(len(pointers)):
            pointer=pointers[i]
            # pointer.append(pointer[0])
            res=0.
            for j in range(len(pointer)-1):
                res+=distances[i,pointer[j],pointer[j+1]]
            res+=distances[i,pointer[0],pointer[-1]]
            results+=res
        return results/len(inputs)
    
    def evaluate_solutions(self,inputs,solutions):
        distances=torch.zeros((inputs.size(0),inputs.size(1),inputs.size(1)))
        for j in range(len(inputs)):
            input=inputs[j]
            for i in range(len(input)):
                tmp=((input-input[i])**2).sum(axis=1)**0.5
                distances[j][i]=tmp
        pointers=solutions
        results=0.
        for i in range(len(pointers)):
            pointer=pointers[i]
            # pointer.append(pointer[0])
            res=0.
            for j in range(len(pointer)-1):
                res+=distances[i,pointer[j],pointer[j+1]]
            res+=distances[i,pointer[0],pointer[-1]]
            results+=res
        return results/len(inputs)
    
    def eval_all(self,inputs,models):
        res=[]
        distances=torch.zeros((inputs.size(0),inputs.size(1),inputs.size(1)))
        for j in range(len(inputs)):
            input=inputs[j]
            for i in range(len(input)):
                tmp=((input-input[i])**2).sum(axis=1)**0.5
                distances[j][i]=tmp
                
        for i in range(len(models)):
            model=models[i]
            res.append(self.evaluate(inputs,model,distances))
        return res
    
    def next_generation(self,inputs):
        pops=[]
        # 生成新后代
        while len(pops)<self.pop_size*2:
            parent_index=self.select_by_value(self.values,2)
            p1,p2=self.pop[parent_index[0]],self.pop[parent_index[1]]
            p_new=self.crossover(p1,p2)
            if random.random()<0.8:
                self.mutation(p_new)
            pops.append(p_new)
        for i in range(self.pop_size):
            pops.append(self.gen_individual())
        self.values.extend(self.eval_all(inputs,pops))
        self.pop.extend(pops)
        tmp_values=torch.Tensor(self.values)
        pop_index=tmp_values.argsort()
        dup_values=[tmp_values[pop_index[0]]]
        tmp_index=[pop_index[0]]
        
        i=1
        while self.pop_size-len(tmp_index)<len(tmp_index)-i:
            indv=tmp_values[pop_index[i]]
            np_values=np.array(dup_values)
            if all(abs(np_values-indv)>0.01):
                dup_values.append(indv)
                tmp_index.append(pop_index[i])
            i+=1
                    
            
        if len(tmp_index)<self.pop_size:
            tmp_index.extend(pop_index[i:])

        self.pop=[self.pop[i] for i in tmp_index]
        self.values=[self.values[i] for i in tmp_index]
        
            
            
    def run(self,inputs,solutions):
        self.init_pop(inputs)
        best=self.evaluate_solutions(inputs,solutions)
        for i in range(self.iteration):
            self.next_generation(inputs)
            print("%d/%d : Top 5 in pop  is %f|%f|%f|%f|%f<<-%f"%(i+1,self.iteration,self.values[0],self.values[1],self.values[2],self.values[3],self.values[4],best))

        

class hyperparam:
    embedding_size=128
    hiddens=512
    nof_lstms=1
    dropout=0.
    bidir=False
    nof_epoch=100000
    lr=0.0001
    train_size=50
    val_size=5
    test_size=10
    batch_size=5
    nof_points=10
    pop_size=5

    
if __name__ == '__main__':
    params=hyperparam()
    
    model=PtrNet(params.embedding_size,params.hiddens,params.nof_lstms,params.dropout,params.bidir)

    dataset=TSPDataset(params.train_size,params.nof_points)
    torch.save(dataset,'dataset.tch')
    dataloader=DataLoader(dataset,batch_size=params.batch_size,shuffle=True,num_workers=4)
    
    inp=np.array(dataset.data['Points_List'])
    inp=torch.Tensor(inp)

    
    solutions=np.array(dataset.data['Solutions'])

   
    model=EAFrame(params.pop_size,params.nof_epoch)
    best=model.evaluate_solutions(inp,solutions)
    print("The best solution is : %f"%best)
    model.run(inp,solutions)
    
    best_model=model.pop[0]
    torch.save(best_model.state_dict(),'best_model_state_dict.tch')
    
    
        

        