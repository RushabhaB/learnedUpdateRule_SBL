import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import sys
import os 
import scipy
import time
import math
from model import posteriorMean, blockCompute_T1_T2, iter_linear_model,full_linear_model
from data import CombinationDataset, ComboBatchSampler
from util import getCovMatrix
from loss import complex_mse_loss_sum, complex_mse_layer_sum
from functions import convert_label_mean
import mlflow
import dagshub
import gradflow_check
import random 

def get_or_create_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id

class TrainParam:
    def __init__(self,
                mu,
                mu_scale,
                mu_epoch,
                weight_decay,
                momentum,
                batch_size,
                mmv_mixed_flag,
                val_batch_size,
                nesterov
                ):
        assert len(mu_scale)==len(mu_epoch), "the length of mu_scale and mu_epoch should be the same"        
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.mmv_mixed_flag = mmv_mixed_flag
        self.val_batch_size = val_batch_size
        self.max_epoch = mu_epoch[-1]
        self.mu = mu
        self.mu_scale = mu_scale
        self.mu_epoch = mu_epoch
        self.nesterov = nesterov
        

class TrainRegressor:
    pin_memory = True
    ckpt_filename = 'train.pt'
    def __init__(self,
                name,
                net,
                n_neurons,
                n_layers,
                model_description,
                layer_wise,
                n_iter,
                var_noise,
                n_gridpoints,
                tp,
                trainset,
                validationset,
                criterion,
                criterion_val,
                device,
                seed,
                resume,
                checkpoint_folder,
                num_workers,
                milestone = [],
                print_every_n_batch = 1,
                m_type="arrayMatrix",
                n_L = 1,
                fp16 = False,
                onecycle = False,
                sameGamma=False,
                sameIter=False,
                fineTune = False,
                sgd=False,
                skip=False
                ):
        torch.manual_seed(seed)
        self.criterion = criterion #torch.nn.L1Loss(reduction='none') # MSELoss or L1Loss
        self.criterion_val = criterion_val
        self.var_noise=var_noise
        self.layer_wise = layer_wise
        self.n_gridpoints= n_gridpoints
        self.description = model_description
        self.same_gamma = sameGamma
        self.sameIter = sameIter
        self.n_iter=n_iter
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.skip=skip
        self.m_type = m_type
        self.n_L = n_L

        if(self.layer_wise):
            model_list_all = torch.nn.ModuleList()
            for i in range(self.n_iter):
                model_list= torch.nn.ModuleList()
                for j in range(self.n_gridpoints):
                    model_g = net(self.n_layers,self.n_neurons,self.skip)
                    model_list.append(model_g)
                    if(self.same_gamma):
                        break
                model_list_all.append(model_list)
                if(self.sameIter):
                    break

            self.net = model_list_all
        else:
            net = net(self.n_gridpoints,self.var_noise)
            self.net = net.to(device)
        
        self.tp = tp
        self.n_iter=n_iter
        self.checkpoint_folder = checkpoint_folder
        self.name = name
        self.seed = seed
        self.num_workers = num_workers
        self.milestone = milestone
        self.print_every_n_batch = print_every_n_batch
        self.device = device
        self.net = self.net.to(device).to(torch.float32)
        self.linear_flag = True
        if self.linear_flag:
            self.linear = full_linear_model(self.n_iter,self.n_gridpoints).to(device).to(torch.float32)
            param = list(self.net.parameters())+list(self.linear.parameters())
        else:
            param = list(self.net.parameters())
        self.fp16 = fp16
        self.onecycle = onecycle
        self.sgd =sgd
        self.fineTune = fineTune
        w=torch.unsqueeze(torch.asarray([0.9**i for i in range(self.n_iter)]),dim=1).T
        self.weights = torch.fliplr(w).to(self.device).float()
        
        print("{}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.num_parameters = self.count_parameters()
        print("Number of parameters for the model {}: {:,}".format(name,self.num_parameters))

        self.mu_lambda = lambda i: next(tp.mu_scale[j] for j in range(len(tp.mu_epoch)) if min(tp.mu_epoch[j]//(i+1),1.0) >= 1.0) if i<tp.max_epoch else 0
        self.rho_lambda = lambda i: next(tp.rho_scale[j] for j in range(len(tp.rho_epoch)) if min(tp.rho_epoch[j]//i,1.0) >= 1.0) if i>0 else 0

        if(self.tp.mmv_mixed_flag):
            self.trainloader = self.create_loader(trainset,tp.batch_size,self.num_workers,self.pin_memory)
        else:
            self.trainloader = torch.utils.data.DataLoader(trainset,batch_size=tp.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=False)
        self.validationloader = torch.utils.data.DataLoader(validationset,batch_size=tp.val_batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=False)
        
        if(self.sgd):
            self.optimizer = torch.optim.SGD(self.net.parameters(),lr=tp.mu,momentum=tp.momentum,nesterov=tp.nesterov,weight_decay=tp.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(param,lr=tp.mu,weight_decay=tp.weight_decay)
        
        if self.onecycle is True:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=tp.mu,steps_per_epoch=len(self.trainloader),epochs=tp.max_epoch)
        else:
            #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=self.mu_lambda)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max = tp.max_epoch)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.total_train_time = 0
        self.start_epoch = 1
        self.train_loss = []
        self.validation_loss = []
        self.validation_loss_mse = []
        self.best_validation_loss = sys.float_info.max
        self.best_validation_loss_mse = sys.float_info.max
        self.ckpt_path = self.checkpoint_folder+self.name+'/'+self.ckpt_filename

        if self.fineTune:
            self.fine_tune_model_name = "./checkpoint/sweep_layer_softmax/BasicIter_Layers_func_no_bn_sameGamma_Layers_5_Iter_15_N_32_skip_True_mixed_loss_array_n30_60dB_correct_var_2_layer"
            self.fine_tune_model_name = self.fine_tune_model_name + "/best_model.pt"
            self.fine_tune_linear_model_name = self.fine_tune_model_name + "_linear"

        if resume is True and os.path.isfile(self.ckpt_path) and not self.fineTune:
            print('Resuming {} from a checkpoint at {}'.format(self.name,self.ckpt_path),flush=True)
            self.__load()
        elif self.fineTune:

            print('Fine tuning using best array model {}'.format(self.fine_tune_model_name,flush=True))
            self.__load_fineTune()
            
            (init_validation_loss,init_validation_mse) = self.validation()

            self.init_validation_loss = init_validation_loss
            self.best_validation_loss = init_validation_loss
            self.best_validation_loss_mse = init_validation_mse
            self.__save_net('init_model.pt')
            self.__save(0)
        else:
            print('Ready to train {} from scratch...'.format(self.name),flush=True)

           
            (init_validation_loss,init_validation_mse) = self.validation()

            self.init_validation_loss = init_validation_loss
            self.best_validation_loss = init_validation_loss
            self.best_validation_loss_mse = init_validation_mse
            self.__save_net('init_model.pt')
            self.__save(0)
 
    def __get_lr(self):
            for param_group in self.optimizer.param_groups:
                return param_group['lr']

    def __check_folder(self):
        if not os.path.isdir(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        if not os.path.isdir(self.checkpoint_folder+self.name):
            os.mkdir(self.checkpoint_folder+self.name)

    def __load(self):
        # Load checkpoint.
        checkpoint = torch.load(self.ckpt_path,map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.best_validation_loss = checkpoint['best_validation_loss']
        self.start_epoch = checkpoint['epoch']+1
        self.train_loss = checkpoint['train_loss']
        self.validation_loss = checkpoint['validation_loss']
        self.total_train_time = checkpoint['total_train_time']
        self.init_validation_loss = checkpoint['init_validation_loss']

    def __load_fineTune(self):
        # Load checkpoint.
        checkpoint = torch.load(self.fine_tune_model_name,map_location=self.device)
        self.net.load_state_dict(checkpoint)

        linear_checkpoint = torch.load(self.fine_tune_linear_model_name,map_location=self.device)
        self.linear.load_state_dict(linear_checkpoint)
        

    def __save_net(self,filename):
        self.__check_folder()
        net_path = self.checkpoint_folder+self.name+'/'+filename
        lin_path = self.checkpoint_folder+self.name+'/'+filename + '_linear'
        torch.save(self.net.state_dict(), net_path)
        if self.linear_flag:
            torch.save(self.linear.state_dict(),lin_path)
        print('{} model saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),net_path))
        if self.linear_flag:
            print('{} linear model saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),lin_path))

    def __save(self,epoch):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'init_validation_loss': self.init_validation_loss,
            'best_validation_loss': self.best_validation_loss,
            'epoch': epoch,
            'train_loss': self.train_loss,
            'validation_loss': self.validation_loss,
            'num_param': self.num_parameters,
            'seed': self.seed,
            'mu': self.tp.mu,
            'mu_scale': self.tp.mu_scale,
            'mu_epoch': self.tp.mu_epoch,
            'weight_decay': self.tp.weight_decay,
            'momentum': self.tp.momentum,
            'batch_size': self.tp.batch_size,
            'total_train_time': self.total_train_time,
            }
        self.__check_folder()

        
        torch.save(state, self.ckpt_path)
        print('{} checkpoint saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),self.ckpt_path))
        del state['net'], state['optimizer'], state['scheduler']
        state_path = self.checkpoint_folder+self.name+'/train.mat'
        scipy.io.savemat(state_path,state)
        print('{} state saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),state_path))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        for i in range(self.start_epoch,self.tp.max_epoch+1):
            lr = self.__get_lr()
            num_batch = len(self.trainloader)
            tic = time.time()
            train_loss = self.__train(i)
            toc = time.time()
            self.total_train_time += (toc-tic)
            print('training speed: {:.3f} seconds/epoch'.format(self.total_train_time/i))

            (validation_loss,validation_loss_mse) = self.validation()

            self.train_loss.append(train_loss)
            self.validation_loss.append(validation_loss)
            self.validation_loss_mse.append(validation_loss_mse)
            
            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self.__save_net('best_model.pt')
            
            if validation_loss_mse < self.best_validation_loss_mse:
                self.best_validation_loss_mse = validation_loss_mse
                self.__save_net('best_model.pt')


            print('{} [{}] [Validation] epoch: {}/{} batch: {:3d}/{} lr: {:.1e} loss: {:.6f} mse: {:.6f} best_loss: {:.6f} best_mse_loss: {:.6f}'.format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.name,
                    i,
                    self.tp.max_epoch,
                    num_batch,
                    num_batch,
                    lr,
                    validation_loss,
                    validation_loss_mse,
                    self.best_validation_loss,
                    self.best_validation_loss_mse,
                    self.total_train_time
                    ),flush=True)

            for k in self.milestone:
                if k==i:
                    self.__save_net('epoch_'+str(k)+'_model.pt')
                    self.__save(k)

            if math.isnan(train_loss):
                print("{} NaN train loss... break the training loop".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                break
        if self.start_epoch<self.tp.max_epoch+1:
            self.__save_net('last_model.pt')
            self.__save(i)
            print('end of training at epoch {} for the model saved at {}'.format(i,self.ckpt_path))
        else:
            print('the model '+self.ckpt_path+' has already been trained for '+str(self.tp.max_epoch)+' epochs')
        return self
    
    def create_loader(self,trainset,batch_size,num_workers,pin_memory):
        train_data_combined = CombinationDataset(trainset)
        sampler = ComboBatchSampler(len(trainset[0]),len(trainset),
                                     batch_size=batch_size, drop_last=False) 
        return DataLoader(train_data_combined, batch_sampler=sampler, num_workers=num_workers,pin_memory=pin_memory,drop_last=False)
        
    
    def __train(self,epoch_idx):
        tic = time.time()
        self.net.train()
        accumulated_train_loss = 0
        total = 0
        torch.manual_seed(self.seed+epoch_idx)
        lr = self.__get_lr()
        num_batch = len(self.trainloader)
        for batch_idx, (inputs, targets) in enumerate(self.trainloader,1):
            inputs, targets = inputs, [targets[0].to(self.device),targets[1].to(self.device)]
           
            if self.linear_flag:
                targets_supp = targets[0]
                targets_mag = targets[1]
            else:
                targets = targets[1]

            G = torch.ones(size=(targets_supp.shape)).float().to(self.device)
            gamma_all = torch.zeros(size=(self.n_iter+1,targets_supp.shape[0],targets_supp.shape[1])).to(torch.float32).to(self.device)
            post_mean_vec = torch.zeros(size=(self.n_iter,targets_supp.shape[0],targets_supp.shape[1],targets_mag.shape[2]),dtype=inputs[0].dtype).to(self.device)
            gamma_all[0,:,:] = G
            self.optimizer.zero_grad()

            with torch.autocast(enabled=self.fp16, device_type='cuda', dtype=torch.float16):
                for i in range(self.n_iter):
                    shuffle_indices = torch.randperm(self.n_gridpoints)

                    T1,T2 = blockCompute_T1_T2(G,inputs,self.var_noise,self.device,self.m_type)
                    
                    T1, T2 = T1.unsqueeze(-1), T2.unsqueeze(-1)
                    net_input = torch.concatenate((T1,T2,G.unsqueeze(-1)),axis=-1).float()

                    net_input = net_input[:,shuffle_indices,:]

                    out = torch.squeeze(self.net[i][0](net_input),dim=-1)
                    G[:,shuffle_indices] = out  

                    gamma_all[i+1,:,:] = G 
                    post_mean_vec[i,:,:,:] = posteriorMean(G,inputs,var_noise=self.var_noise,device=self.device,m_type=self.m_type)
                    
                if self.linear_flag:
                    lin_out = self.linear(gamma_all)
                    loss = self.criterion(lin_out,targets_supp) + complex_mse_layer_sum(post_mean_vec,targets_mag,self.weights,targets_mag.shape[2])
                else:
                    loss = self.criterion(post_mean_vec,targets)

                batch_mean_loss = torch.mean(loss) # * 1 / self.n_gridpoints

            if torch.isnan(batch_mean_loss):
                print('Nan train loss detected. The previous train loss: {:.6f}'.format(train_loss))
                return float("nan")

            self.scaler.scale(batch_mean_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            accumulated_train_loss += torch.sum(loss).item()

            total += loss.numel()

            train_loss = accumulated_train_loss/total
            toc = time.time()
            if (batch_idx-1)%self.print_every_n_batch == 0 or batch_idx == num_batch:
                print('{} [{}] [Train] epoch: {}/{} batch: {:3d}/{} lr: {:.1e} loss: {:.6f} | ELA: {:.1f}s'.format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.name,
                    epoch_idx,
                    self.tp.max_epoch,
                    batch_idx,
                    num_batch,
                    lr,
                    train_loss,
                    self.total_train_time+toc-tic
                    ),flush=True)
            if self.onecycle is True:
                self.scheduler.step()
        if self.onecycle is False:
            self.scheduler.step()
        return train_loss
    
    def validation(self):
        self.net.eval()
        accumulated_validation_loss = 0
        accumulated_validation_loss_mse = 0
        total = 0
        total_mse = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.validationloader,1):
                inputs, targets = [inputs[0].to(self.device),inputs[1].to(self.device)],\
                      [targets[0].to(self.device),targets[1].to(self.device)]
                
                targets_supp = targets[0]
                targets_mag = targets[1]

                G = torch.ones(size=(targets_supp.shape)).float().to(self.device)
                #inputs = inputs / torch.max(inputs,dim=1).unsqueeze(1)
                if self.linear_flag:
                    gamma_all = torch.zeros(size=(self.n_iter+1,targets_supp.shape[0],targets_supp.shape[1])).to(torch.float32).to(self.device)
                    gamma_all[0,:,:] = G
                
                
                self.optimizer.zero_grad()
                with torch.autocast(enabled=self.fp16, device_type='cuda', dtype=torch.float16):
                    for i in range(self.n_iter):
                        T1,T2 = blockCompute_T1_T2(G,inputs,self.var_noise,self.device,self.m_type)
                        T1, T2 = T1.unsqueeze(-1), T2.unsqueeze(-1)
                        net_input = torch.concatenate((T1,T2,G.unsqueeze(-1)),axis=-1).float()
                        G = torch.squeeze(self.net[i][0](net_input),dim=-1) 

                        if self.linear_flag:
                            gamma_all[i+1,:,:] = G
                        
                    post_mean = posteriorMean(G,inputs,var_noise=self.var_noise,device=self.device,m_type=self.m_type)
                
                if self.linear_flag:
                    out = self.linear(gamma_all)
                else:
                    out = post_mean
                    targets_supp = targets_mag

                loss = self.criterion_val(out,targets_supp)

                mse_loss = complex_mse_loss_sum(post_mean,targets_mag)
                accumulated_validation_loss += torch.sum(loss).item()
                accumulated_validation_loss_mse += torch.sum(mse_loss).item()
                total += loss.numel()
                total_mse +=mse_loss.numel()
                
        validation_loss = accumulated_validation_loss/total
        validation_loss_mse = accumulated_validation_loss_mse /total_mse
        return validation_loss, validation_loss_mse
    
    def get_grad(self):
        grad=[]
        for param in self.net.parameters():
            grad.append(param.grad)
        
        return grad


