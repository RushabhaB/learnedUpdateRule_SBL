import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

    
def blockCompute_T1_T2(gamma,input,var_noise,device,m_type):
    
    if(m_type == "arrayMatrix" or m_type == "c_random"):
        d_type = torch.complex64
    else:
        d_type = torch.float32

    data = input[0].to(d_type).to(device) # need to modify this for multiple snapshots
    n_sensors = data.shape[1]
    batch_I = var_noise*torch.eye(n_sensors,dtype=d_type).unsqueeze(0).to(device)

    if (m_type == "arrayMatrix" or m_type == "corr_matrix"):
        A = input[1][0,:,:].to(d_type).to(device)
        A_T=torch.transpose(torch.conj(A),0,1)
        cov_yy  = torch.einsum('bj,ij,kj->bik',gamma.to(d_type),A,torch.conj(A)) + batch_I
    else:
        A = input[1].to(d_type).to(device)
        A_T=torch.transpose(torch.conj(A),1,2)
        cov_yy  = torch.einsum('bj,bij,bkj->bik',gamma.to(d_type),A,torch.conj(A)) + batch_I
   
    temp_T1 = torch.linalg.solve(cov_yy,data)
    temp_T2 = torch.linalg.solve(cov_yy,A)

    if(m_type == "arrayMatrix" or m_type == "corr_matrix"):
        T_2 = torch.abs(torch.einsum('ij,bji->bi',A_T,temp_T2))
        T_1 = torch.mean(torch.abs(A_T@temp_T1)**2,dim=-1).squeeze()
    else:
        T_2 = torch.abs(torch.einsum('bij,bji->bi',A_T,temp_T2))
        T_1 = torch.mean(torch.abs(torch.bmm(A_T,temp_T1))**2,dim=-1).squeeze()

    del temp_T1
    del temp_T2

    return T_1, T_2


def posteriorMean (gamma,input,var_noise,device,m_type):

    if(m_type == "arrayMatrix" or m_type == "c_random"):
        d_type = torch.complex64
    else:
        d_type = torch.float32
    
    n_sensors = input[0].shape[1]
    gamma = gamma.to(d_type)

    data = input[0].to(d_type).to(device) # need to modify this for multiple snapshots
    batch_I = var_noise*torch.eye(n_sensors,dtype=d_type).unsqueeze(0).to(device)

    if (m_type == "arrayMatrix" or m_type == "corr_matrix"):
        A = input[1][0,:,:].to(d_type).to(device)
        A_T=torch.transpose(torch.conj(A),0,1)
        cov_yy  = torch.einsum('bj,ij,kj->bik',gamma,A,torch.conj(A)) + batch_I
    else:
        A = input[1].to(d_type).to(device)
        A_T=torch.transpose(torch.conj(A),1,2)
        cov_yy  = torch.einsum('bj,bij,bkj->bik',gamma,A,torch.conj(A)) + batch_I

    temp_mu = torch.linalg.solve(cov_yy,data)

    if (m_type == "arrayMatrix" or m_type == "corr_matrix"):
        mean = torch.einsum('bi,ij,bjk->bik',gamma,A_T,temp_mu)
    else:
        mean = torch.einsum('bi,bij,bjk->bik',gamma,A_T,temp_mu)

    del temp_mu

    return mean
          


class BasicIter(nn.Module):
    def __init__(self,n_layers,n_neurons,bias=True,batch_norm=True,skip=False):
        super(BasicIter,self).__init__()

        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.n_layers = n_layers
        self.bn_flag = batch_norm
        self.skip = skip

        for i in range(self.n_layers):
            if(i==0):
                self.fc.append(nn.Linear(3,n_neurons,bias=bias))
                if(self.bn_flag):
                    self.bn.append(nn.BatchNorm1d(n_neurons))
            else:
                self.fc.append(nn.Linear(n_neurons,n_neurons,bias=bias))
                if(self.bn_flag):
                    self.bn.append(nn.BatchNorm1d(n_neurons))
            
            if(self.skip):
               self.skip_conn_1 = nn.Linear(3,n_neurons)

        self.fc.append(nn.Linear(n_neurons,1))
        
    def forward(self,x):
        y = x
        for i in range(self.n_layers):
            y = self.fc[i](y)
            y = F.relu(y)
            if(self.bn_flag):
                y = self.bn[i](y)
            if(i == 1 and self.skip):
                y = y + self.skip_conn_1(x) 
                x1 = y
            if(i==3 and self.skip):
                y = y + x1 
        
        y = self.fc[self.n_layers](y)

        return F.softplus(y) + ((x[:,:,0:1] / x[:,:,1:2]) * x[:,:,2:])
    
class iter_linear_model(nn.Module):
    def __init__(self, n_iter,n_gridpoints):
        super(iter_linear_model,self).__init__()
        self.n_gridpoints = n_gridpoints
        self.n_iter = n_iter
        self.W = nn.Parameter(nn.init.xavier_normal(torch.empty(self.n_gridpoints,self.n_iter+1)),requires_grad=True)
        
    def forward(self,x):
        # x is a n_iter+1 x batch_size x n_gridpoints
        out = torch.einsum('ij,jbi->bi',self.W,x)
        return out 

class full_linear_model(nn.Module):
    def __init__(self, n_iter,n_gridpoints) -> None:
        super(full_linear_model,self).__init__()

        self.n_gridpoints = n_gridpoints
        self.n_iter = n_iter

        self.linear = torch.nn.Sequential(
             torch.nn.Linear(in_features=(self.n_iter+1)*n_gridpoints,out_features=512),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(in_features=512,out_features=512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=512,out_features=n_gridpoints)
        )
    
    def forward(self,gamma_all):

        cat_gamma = torch.cat(tuple(gamma_all[i,:,:] for i in range(self.n_iter+1)),dim=-1)

        return self.linear(cat_gamma)

    
class modelGamma(nn.Module):
    def __init__(self,iter,n_iter,n_layers,n_neurons,n_gridpoints,var_noise,bias,skip,bn=True,same_iter=False,same_gamma=False):
        super(modelGamma,self).__init__()

        self.n_iter = n_iter
        self.n_layers = n_layers
        self.var_noise = var_noise
        self.n_gridpoints=n_gridpoints 
        self.layers_allGamma = nn.ModuleList()
        self.same_iter = same_iter
        self.same_gamma = same_gamma

        for _ in range(self.n_gridpoints):
            self.layers_perGamma = nn.ModuleList()
            for _ in range(self.n_iter):
                self.layers_perGamma.append(iter(self.n_layers,n_neurons,bias,bn,skip))
                if(self.same_iter):
                    break
            self.layers_allGamma.append(self.layers_perGamma)
            if(self.same_gamma):
                break
        

    def forward(self,input):
        self.device = self.get_device
        n_batch = input.shape[0]
        gamma = torch.ones(self.n_iter,n_batch,self.n_gridpoints).float().to(self.device)
        stack_G = torch.ones(n_batch,self.n_gridpoints).float().to(self.device)
        for i in range(self.n_iter):
            T_1,T_2 = blockCompute_T1_T2(stack_G,input,self.var_noise,self.device)
            for j in range(self.n_gridpoints):
                net_input = torch.transpose(torch.stack([T_1[:,j],T_2[:,j],stack_G[:,j]]),1,0).float()

                iter_index = 0 if(self.same_iter) else i
                gamma_index = 0 if(self.same_gamma) else j
                
                G = torch.squeeze(self.layers_allGamma[gamma_index][iter_index](net_input))

                stack_G[:,j] = G
            
        gamma = stack_G
        return gamma

    @property
    def get_device(self):
        return next(self.parameters()).device.type





def BasicIter_2_Layers_1_N_8(n_gridpoints,var_noise):
    return modelGamma(BasicIter,n_iter=2,n_layers=1,n_neurons=8,n_gridpoints=n_gridpoints,var_noise=var_noise,bias=True,skip=False,same_gamma=False,same_iter=False)

def BasicIter_10_Layers_1_N_8(n_gridpoints,var_noise):
    return modelGamma(BasicIter,n_iter=10,n_layers=1,n_neurons=8,n_gridpoints=n_gridpoints,var_noise=var_noise,bias=True,skip=False,same_gamma=False,same_iter=False)

def BasicSameIter_10_Layers_1_N_8(n_gridpoints,var_noise):
    return modelGamma(BasicIter,n_iter=10,n_layers=1,n_neurons=8,n_gridpoints=n_gridpoints,var_noise=var_noise,bias=True,skip=False,same_gamma=False,same_iter=True)

def BasicIter_10_Layers_1_N_8_sameGamma(n_gridpoints,var_noise):
    return modelGamma(BasicIter,n_iter=10,n_layers=1,n_neurons=8,n_gridpoints=n_gridpoints,var_noise=var_noise,bias=True,skip=False,same_gamma=True,same_iter=False)

def BasicIter_3_Layers_2_N_64_no_bn_sameGamma(n_gridpoints,var_noise):
    return modelGamma(BasicIter,n_iter=3,n_layers=2,n_neurons=64,n_gridpoints=n_gridpoints,var_noise=var_noise,bias=True,skip=False,bn=False,same_gamma=True,same_iter=False)

def BasicIter_Layers_2_N_64_no_bn_sameGamma():
    return BasicIter(n_layers=2,n_neurons=64,bias=True,batch_norm=False,skip=False)

def BasicIter_Layers_func_no_bn_sameGamma(n_layers,n_neurons,skip):
    return BasicIter(n_layers=n_layers,n_neurons=n_neurons,bias=True,batch_norm=False,skip=skip)

def BasicIter_10_Layers_2_N_8_sameGamma(n_gridpoints,var_noise):
    return modelGamma(BasicIter,n_iter=10,n_layers=2,n_neurons=8,n_gridpoints=n_gridpoints,var_noise=var_noise,bias=True,skip=False,same_gamma=True,same_iter=False)






model_dict = {
    
    "BasicIter_Layers_func_no_bn_sameGamma": BasicIter_Layers_func_no_bn_sameGamma,

    "BasicIter_Layers_2_N_64_no_bn_sameGamma": BasicIter_Layers_2_N_64_no_bn_sameGamma,

    "BasicIter_3_Layers_2_N_64_no_bn_sameGamma": BasicIter_3_Layers_2_N_64_no_bn_sameGamma,

    "BasicIter_2_Layers_1_N_8": BasicIter_2_Layers_1_N_8,

    "BasicIter_10_Layers_1_N_8": BasicIter_10_Layers_1_N_8,

    "BasicSameIter_10_Layers_1_N_8":BasicSameIter_10_Layers_1_N_8,

    "BasicIter_10_Layers_1_N_8_sameGamma":BasicIter_10_Layers_1_N_8_sameGamma,

    "BasicIter_10_Layers_2_N_8_sameGamma":BasicIter_10_Layers_2_N_8_sameGamma,


}


    