import torch 
import numpy as np 
import scipy 
import tqdm
from data import SBLDataset
from train import TrainParam,TrainRegressor
from model import model_dict, posteriorMean
from loss import complex_mse_loss_sum, complex_mse_layer_sum, weighted_cross_entropy, SubspaceDist
import dagshub
from functions import EM_SBL_torch,tipping_SBL_torch, convert_label_mean
from torch.utils.data import DataLoader
import mlflow



def sweep_layer(n_layers,n_neurons,n_iter,mu,max_epochs,skip):
          

    var_noise=1e-3# Do not change this all the models are initilized with this variance of noise
    SNR = 30 # dB 
    # Initializing and creating the dataset for random matrix
    
    sparsity=[i for i in range(1,15)]
    n_sensors=30
    start_angle=30
    end_angle=150
    n_deg_sep=1
    n_snapshots= [1,2,3,5,7,10]
    n_gridpoints = int((end_angle-start_angle)/n_deg_sep)
    train_N_datapoints=len(sparsity)*160000 // len(n_snapshots) # Number of points for every snapshot length
    valid_N_datapoints = len(sparsity)*3000
    seed=2220
    m_flag = 0 # Controls the type of matrix to train the model on 
    if(m_flag == 0):
        mm_type = {
                "type": "arrayMatrix",
                "n_sensors" : n_sensors,
                "start_angle": start_angle,
                "end_angle":end_angle,
            "n_deg_sep": n_deg_sep
        }
    elif(m_flag == 1):
        mm_type= {
            "type": "corr_matrix",
            "n_sensors" : 20,
        "n_gridpoints": 100
        }
        n_sensors=20
        n_gridpoints=100
    else:
        mm_type= {
            "type": "random",
            "n_sensors" : n_sensors,
        "n_gridpoints": n_gridpoints
        }
    

    mmv_mixed_flag = True
    if mmv_mixed_flag:
        traindataset = []
        for i in range(len(n_snapshots)):
            traindataset.append(SBLDataset(SNR,mm_type,var_noise,sparsity,n_snapshots[i],seed+5,train_N_datapoints))
    else:
        traindataset = SBLDataset(SNR,mm_type,var_noise,sparsity,n_snapshots,seed+5,train_N_datapoints)
    validdataset = SBLDataset(SNR,mm_type,var_noise,sparsity,n_snapshots[0],seed+1,valid_N_datapoints)

    criterion= torch.nn.CrossEntropyLoss(reduction='none')
    

    mu=mu# learning rate
    momentum=0.99
    weight_decay=1e-8
    mu_scale=[1,1]
    mu_epoch=[10,max_epochs]
    milestone=[i for i in range(max_epochs,10)]
    batch_size=4096 // 2
    val_batch_size= 4096 
    onecycle=False
    sgd = False
    nesterov=True
    fineTune = False
    
    # Layer param

    
    sameGamma = True
    sameIter = False


    if(torch.backends.mps.is_available()):
        device = torch.device('mps')
        device = "cpu"
    elif(torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    resume=False
    checkpoint_folder="./checkpoint/sweep_layer_softmax_mmv/"
    num_workers=4
    print_every_n_batch=500
    fp16=False
    benchmark = False

    # SEEING HOW TIPPING SBL AND EM-SBL DOES ON THE VALIDATION DATASET

    if(benchmark):
        valid_dataloader = DataLoader(validdataset,batch_size=val_batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=False)
        ce_EM= 0
        ce_p=0
        total_loss_EM=0
        total_loss_p=0
        n_elem=0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_dataloader,1):

                inputs, targets = [inputs[0].to(device),inputs[1].to(device)], targets.to(device)
                
                targets_ce = targets[0]
                targets = targets[1]

                gamma_EM,n_iter_em = EM_SBL_torch(inputs,var_noise,device,mm_type['type'])
                postMean_EM = posteriorMean(gamma_EM,inputs,var_noise,device,mm_type['type'])
                
                gamma_p,n_iter_p = tipping_SBL_torch(inputs,var_noise,device,mm_type['type'])
                postMean_p = posteriorMean(gamma_p,inputs,var_noise,device,mm_type['type'])

                loss_EM = complex_mse_loss_sum(postMean_EM,targets)
                loss_p = complex_mse_loss_sum(postMean_p,targets)
                
                ce_loss_EM = criterion(gamma_EM,targets_ce)
                ce_loss_p = criterion(gamma_p,targets_ce)
                
                total_loss_EM+= torch.sum(loss_EM).item()
                total_loss_p+= torch.sum(loss_p).item()

                ce_EM += torch.sum(ce_loss_EM).item()
                ce_p+= torch.sum(ce_loss_p).item()

                n_elem += loss_EM.numel()
            
            valid_loss_EM= total_loss_EM/n_elem
            valid_loss_p=total_loss_p/n_elem

            print('The validtion loss for EM is : {:1e}, and for tippings algorithm is {:1e}'.format(valid_loss_EM,valid_loss_p))
            print('The CE for EM is : {:0.2f}, and for tippings algorithm is {:}'.format(ce_EM / n_elem,ce_p / n_elem))        
    
    torch.cuda.empty_cache()

    layerWise=True
    
    criterion= torch.nn.CrossEntropyLoss(reduction='none') #torch.nn.BCEWithLogitsLoss(reduction='none') 
    criterion_val = torch.nn.CrossEntropyLoss(reduction='none') #torch.nn.BCEWithLogitsLoss(reduction='none') 
    

    # Defining parameters of model
    model="BasicIter_Layers_func_no_bn_sameGamma"
    name=model+"_Layers_"+ str(n_layers) + "_Iter_"+ str(n_iter)+"_N_" + str(n_neurons) + "_sigProc_skip_" + str(skip) +"_mixed_loss" + "_arr_n30_30dB_2_layer_mixed_snapshots"
    description = "Training model with 2 layers in each iteration and sharing among all the gammas."
    

    tp = TrainParam(
                mu=mu,
                mu_scale=mu_scale,
                mu_epoch=mu_epoch,
                weight_decay=weight_decay,
                momentum=momentum,
                batch_size = batch_size,
                mmv_mixed_flag=mmv_mixed_flag,
                val_batch_size = val_batch_size,
                nesterov = nesterov
                )

    
    c = TrainRegressor(
        name=name,
        net=model_dict[model],
        n_neurons=n_neurons,
        n_layers=n_layers,
        model_description=description,
        layer_wise=layerWise,
        n_iter=n_iter,
        var_noise=var_noise,
        n_gridpoints=n_gridpoints,
        tp=tp,
        trainset=traindataset,
        validationset=validdataset,
        criterion = criterion,
        criterion_val=criterion_val,
        device=device,
        seed=seed,
        resume=resume,
        checkpoint_folder = checkpoint_folder,
        num_workers = num_workers,
        milestone = milestone,
        m_type = mm_type['type'],
        n_L=n_snapshots,
        print_every_n_batch = print_every_n_batch,
        fp16 = fp16,
        onecycle = onecycle,
        sameGamma=sameGamma,
        sameIter=sameIter,
        fineTune=fineTune,
        sgd=sgd,
        skip=skip
    ).train()


if __name__ == '__main__':
    
    n_neurons_vec = [64]
    n_layers_vec = [5]
    n_iter_vec = [15]

    mu = [5e-4]
    max_epoch_vec = [220]
    skip=True
    for i in range(len(n_neurons_vec)):
        for n_layers in n_layers_vec:
            if(n_layers>3):
                skip=True
            for n_iter in n_iter_vec:
                sweep_layer(n_neurons=n_neurons_vec[i],n_layers=n_layers,n_iter=n_iter,mu=mu[i],max_epochs=max_epoch_vec[i],skip=skip)