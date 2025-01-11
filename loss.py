import torch
import numpy as np

ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

def n_log_likelihood(gamma,A,y,var_noise):
    n_sensors = np.shape(A)[0]
    cov_yy = var_noise * np.eye(n_sensors) + A @ gamma @ A.conj().T
    cov_yy_inv = torch.linalg.inv(cov_yy)

    loss = torch.log(torch.det(cov_yy)) + y.T @ cov_yy_inv @ y

    return loss

def complex_mse_loss_sum(output, target):
    dims = len(output.shape)
    n_s = output.shape[2]
    return torch.sum(torch.abs(output - target)**2,dim=tuple([i for i in range(1,dims)])) *1 /n_s

def complex_mse_loss_mean(output, target):
    return torch.mean(torch.abs(output - target)**2,axis=1)

def complex_mse_layer_sum(output,target,weights,n_L):
    dims = len(output.shape)
    loss=torch.sum(torch.abs(output - target)**2,dim=tuple([i for i in range(2,dims)])).float() * 1 / n_L
    loss= torch.matmul(weights ,loss).squeeze()
    return loss

def weighted_cross_entropy(output,target):
    n_iter = output.shape[0]

    target = torch.unsqueeze(target,dim=-1)

    w=torch.unsqueeze(torch.asarray([0.85**i for i in range(n_iter)]),dim=1).T
    weights = torch.fliplr(w).to(output.device).float()

    loss = torch.empty(size=(n_iter,output.shape[1])).to(output.device)
    for i in range(n_iter):
        o = torch.unsqueeze(output[i],dim=-1)
        loss[i] = torch.squeeze(ce_loss(o,target))
    
    loss= torch.matmul(weights ,loss)
    loss=torch.sum(loss,axis=0)

    return loss   



    