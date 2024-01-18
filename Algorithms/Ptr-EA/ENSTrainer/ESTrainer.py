import torch
from torch.autograd import Variable
import os,sys
sys.path.append('./')

from PointerNet import PointerNet

import numpy as np
import argparse
from tqdm import tqdm

from Data_Generator import TSPDataset
from torch.utils.data import DataLoader

import time
from torch import linalg as linalg

import math
from concurrent.futures import ThreadPoolExecutor



def logger(log,file_path):
    with open(file_path,'a') as f:
        f.write(log+'\n')


def custom_collate_fn(batch):
    batch = [data for data in batch if data is not None]  # 过滤掉为 None 的样本
    return torch.utils.data.dataloader.default_collate(batch)


def calculate_path_length(y, cities):
    """计算TSP实例结果序列的的路径长度

    Args:
        y (int arr): TSP解的城市序列
        cities ([(x11,x12),...,]): 城市坐标

    Returns:
        double : 解的回路长度
    """
    path_length = 0

    for i in range(len(y) - 1):
        city1 = cities[y[i]]
        city2 = cities[y[i+1]]
        distance = math.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
        path_length += distance

    # 添加回到起点的路径长度
    city1 = cities[y[-1]]
    city2 = cities[y[0]]
    distance = math.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
    path_length += distance

    return path_length


def calculate_paths(ys,cities_list):
    total_length=0.0
    for i in range(len(ys)):
        y=ys[i]
        cities=cities_list[i]
        total_length+=calculate_path_length(y,cities)
    return total_length/len(ys)

        
class ENSTrainer:
    def __init__(self,inp_size=50,population_size=5,time_budget=500,task_desc="",cpus=5):
        """指针网络的演化训练器

        Args:
            inp_size (int, optional): 训练数据的节点规模. Defaults to 50.
            population_size (int, optional): 种群数量. Defaults to 5.
            time_budget (int, optional): 训练时间预算. Defaults to 500.
            task_desc (str, optional): 任务描述（用于生成保存模型的文件夹）. Defaults to "".
        """
        self.population_size=population_size
        self.time_budget=time_budget*60
        self.inp_size=inp_size
        self.cpus=cpus

        # hyperparameters
        self.batch_change_sigma=10
        self.epoches=20
        self.t_max=10000
        self.batch_size=256
        self.emb_size=128
        self.hidden_size=512
        self.r=0.99
        self.nof_lstms=5
        self.train_size=1000
        self.test_size=100
        self.sigma=0.08
        
        #init population
        self.population=[self.init_individual() for _ in range(population_size)]
        self.std_devs=[self.init_std_dev(ind_i) for ind_i in self.population]
        self.sigmas=[0.08]*self.population_size
        
        self.folder='./res/task_desc_size-%d_pop-%d/'%(self.inp_size,self.population_size)
        os.makedirs(self.folder,exist_ok=True)
        self.logfile=self.folder+'log.txt'
        
        
    def save_models(self,model,name):
        """保存模型的所有参数

        Args:
            model (_type_): 神经网络
            name (_type_): 文件名
        """ 
        path=self.folder+name 
        torch.save(model.state_dict(),path)

    def init_individual(self):
        """初始化个体

        Returns:
            NN  : 随机生成的指针网络
        """

        nn=PointerNet(self.emb_size, self.hidden_size, self.nof_lstms, dropout=0.,bidir=False)
        return nn 
    
    def init_std_dev(self,ind):
        """初始化指针网络参数的分布

        Args:
            ind (NN): 指针网络
        """
        res=[]
        for param in ind.parameters():
            std_dev=torch.randn_like(param)*0.08
            res.append(std_dev)
        return res 
    
    def mutations(self):
        """生成所有子代

        Returns:
            NNs: 返回所有变异后产生的子代
        """
        offspring=[]
        for index,individual in enumerate(self.population):
            with torch.no_grad():
                for i,param in enumerate(individual.parameters()):
                    noise = torch.randn_like(param)*self.std_devs[index][i]
                    param.add_(noise)
            offspring.append(individual)
            
        return offspring
    
    def evaluate_individual(self,individual, dataset):
        """计算个体在给定batch上的fitness值

        Args:
            individual (NN): 指针网络
            dataset (Batch Instances): 训练数据

        Returns:
            double : 在给定数据上的平均路径长度
        """
        with torch.no_grad():
            o, p = individual(dataset)
            
        return calculate_paths(p,dataset)
    
    def test_population(self,individual,datasets):
        with torch.no_grad():
            o,p = individual(datasets)
            
            
    
    def get_weights(self,individual):
        flattened_weights = torch.tensor([])

        for param_tensor in individual.parameters():
            flattened_weights = torch.cat((flattened_weights, param_tensor.view(-1)))

        return flattened_weights
    
    def calculate_cov_matrix(self,n1,n2):
        """计算两个网络的协方差矩阵（对角阵）

        Args:
            n1 (NN): PtrNet
            n2 (NN): PtrNet

        Returns:
            matrix: 协方差矩阵
        """ 
        v1=self.get_weights(n1)
        v2=self.get_weights(n2)
        covariance_matrix = torch.diag(torch.diag(torch.corrcoef(torch.stack([v1,v2],dim=1))))
        return covariance_matrix
        
    def calculate_distance(self,ind1, ind2):
        """计算两个网络之间的距离

        Args:
            ind1 (NN): PtrNet
            ind2 (NN): PtrNet

        Returns:
            double: 距离
        """
        # cov_matrix=self.calculate_cov_matrix(ind1,ind2)
        v1=self.get_weights(ind1)
        v2=self.get_weights(ind2)
        diff = v1 - v2
        mahalanobis_term = torch.matmul(diff, diff)
        determinant_term = torch.log(linalg.det(torch.stack([v1[:2],v2[:2]],dim=1)) / torch.sqrt(linalg.det(torch.stack([v1[:2],v1[:2]],dim=1))*linalg.det(torch.stack([v2[:2],v2[:2]],dim=1))))

        distance = 0.125 * mahalanobis_term + 0.5 * determinant_term

        return distance
    
    def corr_pi(self,ind1):
        """计算个体在种群中的相关性 

        Args:
            ind1 (NN ): PtrNet

        Returns:
            double : 相关性
        """
        min_distance = float('inf')
        
        for i, ind2 in enumerate(self.population):
            # cov_matrix=self.calculate_cov_matrix(ind1,ind2)
            if i != ind1:
                distance = self.calculate_distance(ind1, ind2)
                if distance < min_distance:
                    min_distance = distance
        return min_distance
    
    def check_change(self,f,c,lm):
        """检查是否要用子代替换父代

        Args:
            f (double): 平均路径长
            c (double): 相关系数
            lm (double): lambda

        Returns:
            Bool: 是否替换
        """
        if f/c<lm:
                return True 
        else:
            return False 
    
    def update_population(self,offspring,batch_data,lm):
        """更新种群中的个体

        Args:
            offspring (NN): PtrNet
            batch_data (tensor): batch数据
            lm (double): lambda

        Returns:
            None: 
        """
        
        change_count=[0]*self.population_size
        if self.parent_fits == []:
            self.parent_fits=[self.evaluate_individual(ind_i,batch_data) for ind_i in self.population]

        offspring_fits=[self.evaluate_individual(ind_i,batch_data) for ind_i in offspring]
        
        for i in range(len(self.population)):
            if self.check_change(offspring_fits[i],self.corr_pi(offspring[i]),lm):
                self.parent_fits[i]=offspring_fits[i]
                self.population[i]=offspring[i]
                change_count[i]+=1

        return change_count
    
    def update_population_parallel(self,offspring,batch_data,lm):
        """更新种群中的个体, 多线程版本

        Args:
            offspring (NN): PtrNet
            batch_data (tensor): batch数据
            lm (double): lambda

        Returns:
            None: 
        """
        
        def thread_execution_parent(index):
            """第一次更新

            Args:
                index (int): index
            """ 
            ind=self.population[index]
            result=self.evaluate_individual(ind,batch_data)
            self.parent_fits[index]=result
            
            
        offspring_fits=[0.0]*self.population_size
        
        
        def thread_execution_offspring(index):
            """更新offspring

            Args:
                index (int): index
            """
            ind=offspring[index]
            result=self.evaluate_individual(ind,batch_data)
            offspring_fits[index]=result
            
            
        change_count=[0]*self.population_size
        
        if self.parent_fits == []:
            self.parent_fits=[0.0]*self.population_size

            with ThreadPoolExecutor(max_workers=self.cpus) as executor:
                futures = [executor.submit(thread_execution_parent,i) for i in range(self.population_size)]
                for future in futures:
                    future.result()
                    
        with ThreadPoolExecutor(max_workers=self.cpus) as executor:           
            futures = [executor.submit(thread_execution_offspring,i) for i in range(self.population_size)]
            for future in futures:
                future.result()
                # self.parent_fits=[self.evaluate_individual(ind_i,batch_data) for ind_i in self.population]
            
        for i in range(len(self.population)):
            if self.check_change(offspring_fits[i],self.corr_pi(offspring[i]),lm):
                self.parent_fits[i]=offspring_fits[i]
                self.population[i]=offspring[i]
                change_count[i]+=1
        return change_count
            
            
    def update_sigma(self,change_count):
        """更新每个个体的sigma

        Args:
            change_count (int) : 替代父代的数量
        """
        for i,c in enumerate(change_count):
            if c/self.epoches<0.2:
                self.sigmas[i] = self.sigmas[i]/self.r 
            elif c/self.epoches >0.2:
                self.sigmas[i] = self.sigmas[i]*self.r 
            else:
                continue
            self.std_devs[i]*=1-self.sigmas[i]

    def train(self):
        """开始训练
        """
        self.start_time=time.time()
        time_used=0
        t=0
        self.parent_fits=[]
        lmbd=0.99
        over=False 
        for epoch in range(self.epoches):
            iterator = tqdm(train_dataloader, unit='Batch')

            for i_batch,sample_batched in enumerate(iterator):
                # iterator.set_description('Batch %i/%i' % (epoch+1, 50000))
                time_used=time.time()-self.start_time
                if time_used>=self.time_budget:
                    over=True 
                    break
                iterator.set_description("%i/%i batch - %d/%d sec: " % (epoch+1,self.epoches,int(time_used),self.time_budget))
                train_batch = Variable(sample_batched['Points'])
                # target_batch = Variable(sample_batched['Solution'])
                # train_batches=torch.unsqueeze(train_batch,dim=0).expand(self.population_size,-1,-1)
                offspring=self.mutations()
                if self.cpus==1:
                    change_count=self.update_population(offspring,train_batch,lmbd)
                else:
                    change_count=self.update_population_parallel(offspring,train_batch,lmbd)

                # change_count=[0]*self.population_size
                # if self.parent_fits == []:
                #     self.parent_fits=[self.evaluate_individual(ind_i,train_batch) for ind_i in self.population]
                # offspring_fits=[self.evaluate_individual(ind_i,train_batch) for ind_i in offspring]
                
                # for i in range(len(self.population)):
                #     if self.check_change(offspring_fits[i],self.corr_pi(offspring[i]),lmbd):
                #         self.parent_fits[i]=offspring_fits[i]
                #         self.population[i]=offspring[i]
                #         change_count[i]+=1
                
                avg_length=torch.mean(torch.tensor(self.parent_fits))
                # print("AVG_L:",avg_length)
                iterator.set_postfix({'avg_length':avg_length.item()})
                logger("%i/%i batch - %d/%d sec: %.2f length" % (epoch+1,self.epoches,int(time_used),self.time_budget,avg_length.item()),self.logfile)
                        
            
            if over:
                break 
            t+=1
            if t%self.epoches==0:
                self.update_sigma(change_count)
            
            
        for i,model in enumerate(self.population):
            print("Saving models: %d/%d"%(i,len(self.population)))
            self.save_models(model,'%d.pth'%i)
            
    def test_best(self):
        """选出在测试集上表现最好的Policy
        """
        iterator=tqdm(test_dataloader,unit="Batch")
        epoch=0
        self.test_best_result=[]
        for i_batch,sample_batched in enumerate(iterator):
            iterator.set_description("batch: %i" % (epoch+1))
            test_batch = Variable(sample_batched['Points'])
            batch_res=[self.evaluate_individual(ind_i,test_batch) for ind_i in self.population]
            self.test_best_result.append(batch_res)
        ma=torch.tensor(self.test_best_result)
        avg_length=torch.mean(ma,dim=0)

        max_mean_index=torch.argmin(avg_length)
        logger("Index of Best Policy:%d with min average length:%.2f"%(max_mean_index,avg_length[max_mean_index].item()),self.logfile)

if __name__ == '__main__':
    
    inp_size=50  #[5,10,15,20,50,100,500,1000]

    #准备数据
    train_dataset = TSPDataset(1000000,inp_size,solve=False)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=256,
                                shuffle=True,
                                num_workers=1)


    test_dataset = TSPDataset(2000,inp_size,solve=False)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=200,
                                shuffle=True,
                                num_workers=1)
    # 参数
    # parser = argparse.ArgumentParser(description="ENS: NCS-PtrNet")
    # # Data
    # parser.add_argument('--train_size', default=1000, type=int, help='Training data size')
    # parser.add_argument('--val_size', default=100, type=int, help='Validation data size')
    # parser.add_argument('--test_size', default=100, type=int, help='Test data size')
    # parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    # # Train
    # parser.add_argument('--nof_epoch', default=50000, type=int, help='Number of epochs')
    # parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # # TSP
    # parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
    # # Network
    # parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
    # parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
    # parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
    # parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
    # parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')

    # # for Test
    # # parser.add_argument('--train_size', default=2000, type=int, help='Training data size')
    # # parser.add_argument('--val_size', default=500, type=int, help='Validation data size')
    # # parser.add_argument('--test_size', default=500, type=int, help='Test data size')
    # # parser.add_argument('--batch_size', default=256, type=int, help='Batch size')

    # params = parser.parse_args()

    trainer=ENSTrainer(inp_size=inp_size,
                       population_size=5,
                       time_budget=500,
                       task_desc='test')
    trainer.train()
    
    trainer.test_best()




