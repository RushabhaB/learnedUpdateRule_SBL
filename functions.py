import torch
import numpy as np 
from model import blockCompute_T1_T2
import sys



def EM_SBL_torch(input,var_noise,device,m_type):
    batch_size = input[0].shape[0]
    n_gridpoints = input[1].shape[2]
    gamma = torch.ones(size=(batch_size,n_gridpoints),dtype=torch.float32).to(device)
    cond = torch.asarray([True] * batch_size,dtype=torch.bool).to(device)
    thresh = 1e-6
    n_iter=0
    while(torch.any(cond)):
        n_iter+=1
        gamma_prev = gamma.clone()
        T_1,T_2 = blockCompute_T1_T2(gamma[cond,:],[input[0][cond,:],input[1][cond,:]],var_noise,device,m_type)
        gamma[cond,:] = (T_1-T_2)*gamma[cond,:]**2 + gamma[cond,:]
        
        if(n_iter>5):
            cond = (torch.linalg.vector_norm(gamma-gamma_prev,dim=1) / torch.linalg.vector_norm(gamma_prev,dim=1) ) > thresh
        if(n_iter>500):
            break
    
    return gamma, n_iter

    

def tipping_SBL_torch(input,var_noise,device,m_type):

    batch_size = input[0].shape[0]
    n_gridpoints = input[1].shape[2]
    gamma = torch.ones(size=(batch_size,n_gridpoints),dtype=torch.float32).to(device)
    cond = torch.asarray([True] * batch_size,dtype=torch.bool).to(device)
    thresh = 1e-6
    n_iter=0
    p=1
    while(torch.any(cond)):
        n_iter+=1
        gamma_prev = gamma.clone()
        T_1,T_2 = blockCompute_T1_T2(gamma[cond,:],[input[0][cond,:],input[1][cond,:]],var_noise,device,m_type)
        gamma[cond,:] = (T_1/T_2)**p * gamma[cond,:]

        if(n_iter>5):
            cond = (torch.linalg.vector_norm(gamma-gamma_prev,dim=1) / torch.linalg.vector_norm(gamma_prev,dim=1)) > thresh
        if(n_iter>500):
            break
    return gamma,n_iter

def pick_top_s(gamma,s,device):
    batch_size = gamma.shape[0]
    gamma_mod = torch.zeros(size=gamma.shape).to(device)
    indices_allBatch = torch.topk(gamma,s).indices
    for i in range(batch_size):
        gamma_mod[i,indices_allBatch[i,:]] = gamma[i,indices_allBatch[i,:]]
    
    return gamma_mod


def check_correct_sparsity(x,target,s):

    if(x.type() == "torch.cuda.ComplexDoubleTensor" \
       or x.type() == "torch.ComplexDoubleTensor"\
        or x.type() == "torch.cuda.ComplexFloatTensor"\
            or x.type() == "torch.ComplexFloatTensor"):
        x = torch.abs(x)
    else:
        x=x

    target_index = torch.sort(torch.topk(torch.abs(target),s).indices).values

    data_index = torch.sort(torch.topk(x,s).indices).values

    bool_index = torch.all(target_index == data_index,dim=1)

    return torch.sum(bool_index)


def convert_label_mean(inputs, targets):

    data = torch.unsqueeze(inputs[:,:,0],axis=-1).to(torch.complex64)
    A = inputs[:,:,1:].to(torch.complex64)
    targets= torch.unsqueeze(targets,axis=-1).to(torch.complex64)
    A_correct = torch.bmm(A,targets)

    post_mean = torch.linalg.lstsq(A_correct,data)

    return post_mean