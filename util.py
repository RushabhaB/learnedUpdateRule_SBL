import torch
import numpy as np
from model import blockCompute_T1_T2, model_dict, posteriorMean

def test_layer_loss(net,inputs,n_iter,n_gridpoints,device,var_noise,same_gamma,sameIter):
    inputs = inputs.to(device)
    n_batch = inputs.shape[0]
    G = torch.ones(size=(n_batch,n_gridpoints)).float().to(device)


    for i in range(n_iter):
        T1,T2 = blockCompute_T1_T2(G,inputs,var_noise,device)
        for j in range(n_gridpoints):
            net_input = torch.transpose(torch.stack([T1[:,j],T2[:,j],G[:,j]]),1,0).float()

            gamma_ind = 0 if(same_gamma) else j
            iter_ind = 0 if(sameIter) else i

            model_out = net[iter_ind][gamma_ind](net_input)
            G[:,j] = torch.squeeze(model_out,dim=1)
                    
    return G

def test_model_matlab(inputs,targets,var_noise,device,n_iter=15,n_layers=5,n_neurons=64):
    #inputs = torch.tensor(inputs,dtype=torch.complex64)
    inputs = inputs.to(device)
    same_gamma = True
    sameIter = False
    skip = True

    n_gridpoints = inputs.shape[2]-1

    model = torch.nn.ModuleList()
    for i in range(n_iter):
        model_list= torch.nn.ModuleList()
        for j in range(n_gridpoints):
            model_g = model_dict["BasicIter_Layers_func_no_bn_sameGamma"](n_layers=n_layers,n_neurons=n_neurons,skip=skip)
            model_list.append(model_g)
            if(same_gamma):
                break
        model.append(model_list)
        if(sameIter):
            break
    
    model_name="BasicIter_Layers_func_no_bn_sameGamma"+"_Layers_"+ str(n_layers) + "_Iter_"+ str(n_iter)+"_N_" + str(n_neurons) + "_skip_" + "True" + "_mixed_loss" + "_array_n30"
    path = "./checkpoint/sweep_layer_softmax/" + model_name +"/best_model.pt" 
    model_info = torch.load(path,map_location=torch.device(device))
    model.load_state_dict(model_info)
    model.to(device).float()
    model.eval()

    #print(model[0][0].fc[0].weight.type())

    G = torch.ones(size=(targets.shape)).float().to(device)
                
    for i in range(n_iter):
        T1,T2 = blockCompute_T1_T2(G,inputs,var_noise,device=device)
        net_input = torch.concatenate((T1.unsqueeze(-1),T2.unsqueeze(-1),G.unsqueeze(-1)),axis=-1).float()
        
        G = torch.squeeze(model[i][0](net_input),dim=-1)
    
    postMean_model = posteriorMean(G,inputs,var_noise,device)

    return postMean_model

def getCovMatrix(gamma,input,targets,var_noise,m_type,device):

    if(m_type == "arrayMatrix" or m_type == "c_random"):
        d_type = torch.complex64
    else:
        d_type = torch.float32

    data = input[0].to(d_type).to(device) 
    n_sensors = data.shape[1]
    batch_I = var_noise*torch.eye(n_sensors,dtype=d_type).unsqueeze(0).to(device)

    if (m_type == "arrayMatrix" or m_type == "corr_matrix"):
        A = input[1][0,:,:].to(d_type).to(device)
        cov_yy  = torch.einsum('tbj,ij,kj->tbik',gamma.to(d_type),A,torch.conj(A)) + batch_I
        cov_target  = torch.einsum('bj,ij,kj->bik',targets.to(d_type),A,torch.conj(A)) + batch_I
    else:
        A = input[1].to(d_type).to(device)
        cov_yy  = torch.einsum('tbj,bij,bkj->tbik',gamma.to(d_type),A,torch.conj(A)) + batch_I
        cov_target  = torch.einsum('bj,bij,bkj->bik',gamma.to(d_type),A,torch.conj(A)) + batch_I
    
    return cov_yy,cov_target



