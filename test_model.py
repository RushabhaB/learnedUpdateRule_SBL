
import torch 
import numpy as np 
import scipy 
import tqdm
from data import SBLDataset
from train import TrainParam,TrainRegressor
from model import model_dict, posteriorMean,blockCompute_T1_T2, full_linear_model,iter_linear_model
from loss import complex_mse_loss_sum
import dagshub
from functions import EM_SBL_torch,tipping_SBL_torch,check_correct_sparsity
from torch.utils.data import DataLoader
import mlflow
import matplotlib.pyplot as plt
import scipy
var_noise=1e-3 # Do not change this all the models are initilized with this variance of noise
var_noise_trad = var_noise
var_noise_model = 1e-3
var_noise_model2 = 1e-3
SNR = 30 # dB 
# Initializing and creating the dataset for random matrix

sparsity=[i for i in range(1,15)]
#sparsity=[5]
n_sensors=30
start_angle=30
end_angle=150
n_deg_sep=1
n_snapshots=3
n_gridpoints = int((end_angle-start_angle)/n_deg_sep)


if(torch.backends.mps.is_available()):
    device = torch.device('mps')
elif(torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = "cpu"



m_flag = 0
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
elif(m_flag == 2):
    mm_type= {
        "type": "random",
        "n_sensors" : n_sensors,
    "n_gridpoints": n_gridpoints
    }
else:
    mm_type= {
        "type": "c_random",
        "n_sensors" : n_sensors,
    "n_gridpoints": n_gridpoints
    }


test_datapoints = 2000
test_batch_size = 512


# Loading the model
n_iter=15
n_layers=5
n_neurons=64
skip=True
model_name="BasicIter_Layers_func_no_bn_sameGamma"+"_Layers_"+ str(n_layers) + "_Iter_"+ str(n_iter)+"_N_" + str(n_neurons) + "_sigProc_skip_" + "True" + "_mixed_loss" + "_arr_n30_30dB_2_layer_2_snapshots" 
path = "./checkpoint/sweep_layer_softmax_mmv/" + model_name +"/best_model.pt" 
model_info = torch.load(path,map_location=torch.device(device))
lin_flag = True
old1 = False
if lin_flag:
    linear_info = torch.load(path+'_linear',map_location=torch.device(device))

n_iter2=15
n_layers2=5
n_neurons2=64
model_name2="BasicIter_Layers_func_no_bn_sameGamma"+"_Layers_"+ str(n_layers2) + "_Iter_"+ str(n_iter2)+"_N_" + str(n_neurons2) +  "_sigProc_skip_" + "True" + "_mixed_loss" + "_arr_n30_30dB_2_layer_mixed_snapshots" 
path2 = "./checkpoint/sweep_layer_softmax_mmv/" + model_name2 +"/best_model.pt" 
model_info2 = torch.load(path2,map_location=torch.device(device))
lin_flag2= True
old = False
if lin_flag2:
    linear_info2 = torch.load(path2+'_linear',map_location=torch.device(device))

same_gamma=True
sameIter=False

# Instantiating model

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
if lin_flag:
    if old1:
        linear = torch.nn.Sequential(
                    torch.nn.Linear(in_features=(n_iter+1)*n_gridpoints,out_features=512).to(device),
                    #torch.nn.ReLU(inplace=True),
                    #torch.nn.Linear(in_features=512,out_features=512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(in_features=512,out_features=n_gridpoints)
                ).to(device)
    else:
        linear = full_linear_model(n_iter=n_iter,n_gridpoints=n_gridpoints)
    
linear.load_state_dict(linear_info)
linear.to(device)
linear.eval()


model.load_state_dict(model_info)
model.to(device)
model.eval()



model2 = torch.nn.ModuleList()

for i in range(n_iter2):
    model_list1= torch.nn.ModuleList()
    for j in range(n_gridpoints):
        model_g1 = model_dict["BasicIter_Layers_func_no_bn_sameGamma"](n_layers=n_layers2,n_neurons=n_neurons2,skip=skip)
        model_list1.append(model_g1)
        if(same_gamma):
            break
    model2.append(model_list1)
    if(sameIter):
        break
if lin_flag2:
    if old:
        linear2 = torch.nn.Sequential(
                    torch.nn.Linear(in_features=(n_iter2+1)*n_gridpoints,out_features=512).to(device),
                    #torch.nn.ReLU(inplace=True),
                    #torch.nn.Linear(in_features=512,out_features=512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(in_features=512,out_features=n_gridpoints)
                ).to(device)
    else:
        linear2 = full_linear_model(n_iter=n_iter2,n_gridpoints=n_gridpoints)

linear2.load_state_dict(linear_info2)
linear2.to(device)
linear2.eval()

model2.load_state_dict(model_info2)
model2.to(device)
model2.eval()


# Evaluating tipping, model and EM-SBL
criterion = complex_mse_loss_sum
mean_prob_EM_av = np.zeros(shape=(len(sparsity),))
mean_prob_p_av = np.zeros(shape=(len(sparsity),))
mean_prob_model_av = np.zeros(shape=(len(sparsity),))
mean_prob_model2_av = np.zeros(shape=(len(sparsity),))

mean_loss_EM_av=np.zeros(shape=(len(sparsity),))
mean_loss_p_av=np.zeros(shape=(len(sparsity),))
mean_loss_model_av=np.zeros(shape=(len(sparsity),))
mean_loss_model2_av=np.zeros(shape=(len(sparsity),))

seed_vec = [3,30000,300]

for seed in seed_vec:

    mean_loss_EM=[]
    mean_loss_p=[]
    mean_loss_model=[]
    mean_loss_model2=[]


    mean_prob_EM=[]
    mean_prob_p=[]
    mean_prob_model=[]
    mean_prob_model2=[]

    for s in sparsity:
        test_dataset = SBLDataset(SNR,mm_type,var_noise,[s],n_snapshots,seed,test_datapoints)
        test_dataloader = DataLoader(test_dataset,batch_size=test_batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=False)

        prob_EM=0
        prob_p=0
        prob_model=0
        prob_model2=0

        total_loss_EM=0
        total_loss_p=0
        total_loss_model=0
        total_loss_model2=0
        

        n_elem=len(test_dataset)


        for batch_idx, (inputs, targets) in enumerate(test_dataloader,1):
                
                inputs, targets = [inputs[0].to(device),inputs[1].to(device)], [targets[0].to(device),targets[1].to(device)]
                if lin_flag:
                    targets_supp = targets[0]
                    targets_mag = targets[1]
                else:
                    targets = targets[1]


                gamma_EM,n_iter_em = EM_SBL_torch(inputs,var_noise_trad,device,mm_type['type'])
                postMean_EM = posteriorMean(gamma_EM,inputs,var_noise,device,mm_type['type'])
                
                gamma_p,n_iter_p = tipping_SBL_torch(inputs,var_noise_trad,device,mm_type['type'])
                postMean_p = posteriorMean(gamma_p,inputs,var_noise,device,mm_type['type'])


                G = torch.ones(size=(targets_supp.shape)).float().to(device)
                if lin_flag:
                    gamma_all = torch.zeros(size=(n_iter+1,targets_supp.shape[0],targets_supp.shape[1])).float().to(device)
                    
                    gamma_all[0,:]=G
                
                for i in range(n_iter):
                    G_prev=G
                    T1,T2 = blockCompute_T1_T2(G,inputs,var_noise_model,device=device,m_type=mm_type['type'])
                    net_input = torch.concatenate((T1.unsqueeze(-1),T2.unsqueeze(-1),G.unsqueeze(-1)),axis=-1).float()
                    G = torch.squeeze(model[i][0](net_input),dim=-1)
                    G = G # - (T1/T2) *G_prev
                    if lin_flag:
                        gamma_all[i+1,:]=G
                
                postMean_model = posteriorMean(G,inputs,var_noise_model,device,mm_type['type'])

                if lin_flag:
                    if old1:
                        cat_gamma = torch.cat(tuple(gamma_all[i,:,:] for i in range(n_iter+1)),dim=-1)
                    else:
                        cat_gamma = gamma_all
                    lin_out = linear(cat_gamma)
                else:
                    lin_out = postMean_model
                
                #scipy.io.savemat("./gamma.mat",{"g":gamma_all.detach().numpy(),"input":inputs.detach().numpy()})
                loss_model = criterion(postMean_model,targets_mag)
                prob_model+=check_correct_sparsity(lin_out,targets_supp,s)
                
                G2 = torch.ones(size=(targets_supp.shape)).float().to(device)
                if lin_flag2:
                    gamma_all2 = torch.zeros(size=(n_iter2+1,targets_supp.shape[0],targets_supp.shape[1])).float().to(device)
                    gamma_all2[0,:] = G2

                for i in range(n_iter2):
                    T1,T2 = blockCompute_T1_T2(G2,inputs,var_noise_model2,device=device,m_type=mm_type['type'])
                    net_input = torch.concatenate((T1.unsqueeze(-1),T2.unsqueeze(-1),G2.unsqueeze(-1)),axis=-1).float()
                    G2 = torch.squeeze(model2[i][0](net_input),dim=-1)
                    if lin_flag2:
                        gamma_all2[i+1,:,:] = G2

                postMean_model2 = posteriorMean(G2,inputs,var_noise_model2,device,mm_type['type'])

                if lin_flag2:
                    if old:
                        cat_gamma2 = torch.cat(tuple(gamma_all2[i,:,:] for i in range(n_iter2+1)),dim=-1)
                        lin_out2 = linear2(cat_gamma2)
                    else:
                        lin_out2 = linear2(gamma_all2)
                else:
                    lin_out2 = postMean_model2

                

            
                loss_EM = criterion(postMean_EM,targets_mag)
                loss_p = criterion(postMean_p,targets_mag)
                loss_model2 = criterion(postMean_model2,targets_mag)
                
                prob_EM+=check_correct_sparsity(gamma_EM,targets_supp,s)
                prob_p+=check_correct_sparsity(gamma_p,targets_supp,s)
                prob_model2+=check_correct_sparsity(lin_out2,targets_supp,s)
                


                total_loss_EM+= torch.sum(loss_EM).item()
                total_loss_p+= torch.sum(loss_p).item()
                total_loss_model+= torch.sum(loss_model).item()
                total_loss_model2+= torch.sum(loss_model2).item()
            
                
            
            
        mean_loss_EM.append(total_loss_EM/n_elem)
        mean_loss_p.append(total_loss_p/n_elem)
        mean_loss_model.append(total_loss_model/n_elem)
        mean_loss_model2.append(total_loss_model2/n_elem)
    

        mean_prob_p.append(prob_p.detach().cpu().numpy()/n_elem)
        mean_prob_EM.append(prob_EM.detach().cpu().numpy()/n_elem)
        mean_prob_model.append(prob_model.detach().cpu().numpy()/n_elem)
        mean_prob_model2.append(prob_model2.detach().cpu().numpy()/n_elem)
    
    mean_prob_p_av = mean_prob_p_av + mean_prob_p
    mean_prob_EM_av = mean_prob_EM_av + mean_prob_EM
    mean_prob_model_av = mean_prob_model_av + mean_prob_model
    mean_prob_model2_av = mean_prob_model2_av + mean_prob_model2

    mean_loss_p_av = mean_loss_p_av + mean_loss_p
    mean_loss_EM_av = mean_loss_EM_av + mean_loss_EM
    mean_loss_model_av = mean_loss_model_av + mean_loss_model
    mean_loss_model2_av = mean_loss_model2_av + mean_loss_model2

np.savez("./data/softmax_mmv/random_real_arr_5_mmv",tip_p= mean_prob_p_av / len(seed_vec), EM_p = mean_prob_EM_av / len(seed_vec) , array_model_p = mean_prob_model2_av / len(seed_vec),array_model1_p = mean_prob_model_av / len(seed_vec),\
         tip_l= mean_loss_p_av /  len(seed_vec), EM_l = mean_loss_EM_av / len(seed_vec), array_model_l = mean_loss_model2_av / len(seed_vec), array_model1_l = mean_loss_model_av / len(seed_vec) )

sparsity = np.array(sparsity)
fig,axs = plt.subplots(1,2,figsize=(12,7))


axs[0].semilogy(sparsity/n_sensors,mean_loss_EM_av /  len(seed_vec),label='EM')
axs[0].semilogy(sparsity/n_sensors,mean_loss_p_av /  len(seed_vec),label='Tipping')
axs[0].semilogy(sparsity/n_sensors,mean_loss_model_av /  len(seed_vec),label='Model Array: mmv 2 snapshot skip')
axs[0].semilogy(sparsity/n_sensors,mean_loss_model2_av/len(seed_vec),label='Model array: mixed mmv skip')
#axs[0].semilogy(sparsity/n_gridpoints,mean_loss_model_2,label='Basic_Iter_3_layer_2')
axs[0].set_ylabel("Mean square error")
axs[0].set_title(f"Array Matrix: Number of sensors ({n_sensors}), Gridpoints (N={n_gridpoints}), Variance ({var_noise}), SNR ({SNR} dB)")
axs[0].legend()
axs[0].set_xlabel("Ratio of number of non-zero elements (d/N)")
axs[0].grid()

axs[1].plot(sparsity/n_sensors,mean_prob_EM_av /  len(seed_vec),label='EM')
axs[1].plot(sparsity/n_sensors,mean_prob_p_av /  len(seed_vec),label='Tipping')
axs[1].plot(sparsity/n_sensors,mean_prob_model_av /  len(seed_vec),label='Model Array: mmv 2 snapshot skip')
axs[1].plot(sparsity/n_sensors,mean_prob_model2_av/ len(seed_vec),label='Model Array: mixed mmv skip ')
#axs[1].set_title("Array Matrix: Number of sensors (40), Gridpoints (60)")
#axs[1].plot(sparsity/n_gridpoints,mean_prob_model_2,label='Basic_Iter_10')
axs[1].set_ylabel("Probability of support recovery")
axs[1].set_xlabel("Ratio of number of non-zero elements (d/N)")
axs[1].legend()
axs[1].grid()

plt.savefig('./Figures/softmax_mmv/arr_matrix_perf_nlayer_softmax_30dB_2_snapshots_skip_mixed_mmv_model.png')
plt.show()


