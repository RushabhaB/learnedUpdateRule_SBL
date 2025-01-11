import numpy as np
import torch
from tqdm import tqdm
import os
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import normalize
import scipy
import random

class SBLDataset(Dataset):
    def __init__(
            self,
            SNR,
            mm_type,
            var_noise,
            sparsity,
            n_snapshots,
            seed,
            N_datapoints: int,
            dynamic = False,
            mmv_flag = True,
            device='cpu'
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.var_noise=var_noise
        self.n_snapshots = n_snapshots
        self.SNR=SNR
        self.sparsity = sparsity
        self.N_datapoints = N_datapoints
        self.mmv_flag = mmv_flag
        
        self.matrix_type = mm_type["type"]
        if(mm_type["type"]=="arrayMatrix"):
            self.n_sensors = mm_type["n_sensors"]
            self.n_deg_sep = mm_type["n_deg_sep"]
            self.start_angle = mm_type["start_angle"]
            self.end_angle = mm_type["end_angle"]
            array_manifold = lambda n,theta : np.exp( np.array([i for i in range(n)]) * 1j * np.pi * np.cos(theta))
            self.n_gridpoints = int((self.end_angle-self.start_angle)/self.n_deg_sep)
            self.A = np.zeros(shape=(self.n_sensors,self.n_gridpoints),dtype=complex)
            for i in range(self.n_gridpoints):
                theta = self.start_angle + i*self.n_deg_sep
                self.A[:,i] = (1/np.sqrt(self.n_sensors) )* array_manifold(self.n_sensors,theta * (np.pi/180))
            self.type = complex
        elif(mm_type["type"]== "corr_matrix"):
            self.n_sensors = mm_type["n_sensors"]
            self.n_gridpoints = mm_type["n_gridpoints"]

            self.A = scipy.io.loadmat('matrix_corr_unit_20_100.mat')['A']
            self.type = float
        elif(mm_type["type"] == "random"):
            self.n_sensors = mm_type["n_sensors"]
            self.n_gridpoints = mm_type["n_gridpoints"]
            self.type = float
            self.store_matrix=np.zeros(shape=(self.n_sensors,self.n_gridpoints,self.N_datapoints))
        elif(mm_type["type"] == "c_random"):
            self.n_sensors = mm_type["n_sensors"]
            self.n_gridpoints = mm_type["n_gridpoints"]
            self.type = complex
            self.store_matrix=np.zeros(shape=(self.n_sensors,self.n_gridpoints,self.N_datapoints),dtype=self.type)
        
        if(self.var_noise>0):
            var_source = 10**(self.SNR/10) * self.var_noise
        else:
            var_source = 1
        
        self.data_out = np.zeros(shape=(self.n_sensors,self.n_snapshots,self.N_datapoints),dtype=self.type)
        self.data_in = np.zeros(shape=(self.n_gridpoints,self.N_datapoints))
        self.data_in_mean = np.zeros(shape=(self.n_gridpoints,self.n_snapshots,self.N_datapoints),dtype=self.type)

        N_datapoints_sparsity = int(np.floor(self.N_datapoints/len(self.sparsity)))

        if dynamic is False:
            with tqdm(total=self.N_datapoints) as pbar:
                for i in range(len(self.sparsity)):
                    for n in range(N_datapoints_sparsity):
                    
                        x=np.zeros(shape=(self.n_gridpoints,self.n_snapshots),dtype=self.type) # defining the sparse vector
                        support = np.sort(np.random.randint(low=0,high=self.n_gridpoints-1,size=(self.sparsity[i])))

                        while(len(np.unique(support))!=self.sparsity[i]):
                            support = np.random.randint(low=0,high=self.n_gridpoints-1,size=(self.sparsity[i]))

                        if(mm_type["type"] == "arrayMatrix"):
                            x[support] = np.random.normal(0,np.sqrt(var_source/2),size= (len(support),self.n_snapshots)) + 1j*np.random.normal(0,np.sqrt(var_source/2),size= (len(support),self.n_snapshots))

                            noise = np.random.normal(0,np.sqrt(var_noise/2),size= (self.n_sensors,self.n_snapshots)) + 1j*np.random.normal(0,np.sqrt(var_noise/2),size= (self.n_sensors,self.n_snapshots))

                        elif(mm_type["type"] == "corr_matrix"):
                            
                            x[support] = np.random.normal(0,np.sqrt(var_source),size=(len(support),self.n_snapshots))
                            #x[support] = np.random.uniform(-0.5,0.5,size= (len(support),self.n_snapshots)) 

                            #while(np.any(x[support]>=-0.01) and np.any(x[support]<=0.01)):
                            #    x[support] = np.random.uniform(-0.5,0.5,size= (len(support),self.n_snapshots))
                            
                            #noise = 0
                            noise = np.random.normal(0,np.sqrt(var_noise),size= (self.n_sensors,self.n_snapshots)) 


                        elif(mm_type["type"] == "random"):
                            x[support] = np.random.normal(0,var_source,size=(len(support),self.n_snapshots))
                            #B = self.create_block_diagonal([0.9,0.8,0.7])
                            self.A = np.random.normal(0,1,size=(self.n_sensors,self.n_gridpoints)) 
                            self.A = normalize(self.A,norm='l2',axis=0)
                            self.store_matrix[:,:,n+i*N_datapoints_sparsity] =  self.A
                            noise = np.random.normal(0,var_noise,size= (self.n_sensors,self.n_snapshots))
                        
                        elif(mm_type["type"] == "c_random"):
                            x[support] = np.random.normal(0,var_source/2,size=(len(support),self.n_snapshots)) + 1j *  np.random.normal(0,var_source/2,size=(len(support),self.n_snapshots))
                            #B = self.create_block_diagonal([0.9,0.8,0.7])
                            self.A = np.random.normal(0,1,size=(self.n_sensors,self.n_gridpoints)) + 1j * np.random.normal(0,1,size=(self.n_sensors,self.n_gridpoints)) 
                            self.A = self.A / np.linalg.norm(self.A,axis=0)
                            self.store_matrix[:,:,n+i*N_datapoints_sparsity] =  self.A
                            noise = np.random.normal(0,var_noise/2,size= (self.n_sensors,self.n_snapshots)) + 1j * np.random.normal(0,var_noise/2,size= (self.n_sensors,self.n_snapshots))
                            
                        y=self.A @ x + noise
                        if(np.linalg.norm(y)!=0):
                            self.data_out[:,:,n+i*N_datapoints_sparsity] = y #/ np.linalg.norm(y)
                        if(np.linalg.norm(x)!=0):

                            label = np.zeros(shape=(self.n_gridpoints,))
                            label[support] = 1 #/ sparsity[i]
                            self.data_in_mean[:,:,n+i*N_datapoints_sparsity] = x #/ np.linalg.norm(x)
                            self.data_in[:,n+i*N_datapoints_sparsity] = label

                
                    pbar.update(N_datapoints_sparsity)
    
    def __len__(self):
        return self.N_datapoints
    
    def __getitem__(self,idx):

        if self.mmv_flag:
            data = self.data_out[:,:,idx]
            label = self.data_in[:,idx]
            if (self.matrix_type == "random"):
                matrix = self.store_matrix[:,:,idx]
            elif(self.matrix_type == "corr_matrix"):
                matrix = self.A
            elif(self.matrix_type == "arrayMatrix"):
                matrix = self.A
            elif(self.matrix_type == "c_random"):
                matrix = self.store_matrix[:,:,idx]
            
            data = [data,matrix]

        return  data,[label,self.data_in_mean[:,:,idx]]

    
    def create_block_diagonal(self,value):
        ones = np.ones(shape=(self.n_gridpoints))
        
        A = np.diag(ones,0)

        for i in [1,2,3]:
            off_diag = np.ones(shape=(self.n_gridpoints-i))
            A = A + np.diag(off_diag,i)*value[i-1] + np.diag(off_diag,-i)*value[i-1]
             
        return A
    
class ComboBatchSampler():
    
    def __init__(self, len_dataset, num_dataset,batch_size, drop_last):
        
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.len_dataset = len_dataset
        self.num_dataset = num_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_batches = self.len_dataset // self.batch_size
        if(np.mod(self.len_dataset,self.batch_size)>0):
            self.n_batches += 1

    def __iter__(self):
        
        random.seed(42)

         # define how many batches we will grab
        self.n_batches = self.len_dataset // self.batch_size
        if(np.mod(self.len_dataset,self.batch_size)>0):
            self.n_batches += 1

        # creating samples
        rand_indices = []
        ind = [i for i in range(self.len_dataset)]

        for _ in range(self.num_dataset):
            random.shuffle(ind)
            x=ind.copy()
            if(self.n_batches==1):
                rand_indices.append([x])
            else:
                rand_indices.append([x[i:i+self.batch_size] for i in range(0,self.len_dataset,self.batch_size)])
        
        # define which indicies to use for each batch
        self.dataset_idxs = []
        
        for j in range(self.n_batches):
            loader_inds = list(range(self.num_dataset))
            random.shuffle(loader_inds)
            self.dataset_idxs.append(loader_inds)
        
        # return the batch indicies
        batch = []

        for i in range(self.n_batches):
            for dataset_idx in self.dataset_idxs[i]:
                s = len(rand_indices[dataset_idx][i])
                dat_arr = dataset_idx*np.ones(shape=(s,),dtype=int)
                batch = list(zip(dat_arr,rand_indices[dataset_idx][i]))
                yield (batch)
                batch = []

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_dataset * self.n_batches
        else:
            return self.num_dataset * self.n_batches

# combined dataset class
class CombinationDataset(torch.utils.data.DataLoader):
    def __init__(self, datasets):
        self.datasets = datasets
    def __len__(self):
        return(sum([dataset.__len__() for dataset in self.datasets]))
    def __getitem__(self, indicies):
        dataset_idx = indicies[0]
        data_idx = indicies[1]
        return self.datasets[dataset_idx].__getitem__(data_idx)

if __name__== '__main__':

    from torch.utils.data import DataLoader
    import time

    var_noise = 0.1
    SNR = 10 # dB
    sparsity = [2,3,4]
    n_snapshots=1
    seed=1
    train_N_datapoints = int(len(sparsity)*1000)

    n_sensors = 30
    n_gridpoints = 180
    mm_type_1 = {
        "type": "random",
        "n_sensors" : n_sensors,
        "n_gridpoints": n_gridpoints
    }

    start_angle = 0
    end_angle = 180
    n_deg_sep = 1
    mm_type_2 = {
        "type": "arrayMatrix",
        "n_sensors" : n_sensors,
        "start_angle": start_angle,
        "end_angle":end_angle,
        "n_deg_sep": n_deg_sep
    }

    dataset = SBLDataset(SNR,mm_type_2,var_noise,sparsity,n_snapshots,seed,train_N_datapoints)

    dataloader = DataLoader(dataset,batch_size=16,shuffle=True,num_workers=0,pin_memory=True,drop_last=False)
    #print(dataloader)
    print(len(dataloader))
    tic = time.time()
    for idx, (data,label) in enumerate(dataloader):
        print(data.shape)
        print(idx)
    toc = time.time()
    print(toc-tic)
    






                   




        