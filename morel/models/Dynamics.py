from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import scipy.spatial
import os
import tarfile

import morel.models.utils as utils


class DynamicsNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons, threshold, dynamics_epochs, dynamics_lr, swa_lr, swa_start, k_swag, activation = nn.ReLU):
        super(DynamicsNet, self).__init__()
        # Validate inputs
        assert input_dim > 0
        assert output_dim > 0
        assert n_neurons > 0

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.k_swag = k_swag
        self.epochs = dynamics_epochs
        self.dynamics_lr = dynamics_lr

        self.device = torch.device('cuda')
        # output : mean
        self.mean = nn.ModuleList()
        self.mean.append(BasicNet(self.input_dim, self.output_dim, self.n_neurons, activation))

        # output : var
        self.var = nn.ModuleList()
        self.var.append(BasicNet(self.input_dim, self.output_dim, self.n_neurons, activation))

        # SWAG store values
        self.first_moment = utils.flatten(self.parameters()).to(self.device) # in tensor
        self.second_moment = torch.square(self.first_moment).to(self.device) # in tensor
        self.D = np.empty((self.first_moment.shape[0],0)) # in numpy
    def forward(self, x):
        m = x
        v = x
        for layer in self.mean:
            m = layer(m)
        for layer in self.var:
            v = layer(v)
        return m, v
    
    def train(self, dataloader, summary_writer = None, comet_experiment = None):    
        print(f'weight {sum(p.numel() for p in self.parameters())}')
        self.cuda()
        hyper_params = {
            "usad_threshold": self.threshold,
            "dynamics_epochs" : self.epochs
            }
        if(comet_experiment is not None):
            comet_experiment.log_parameters(hyper_params)
        
        # Define optimizers and loss functions
        self.optimizer = torch.optim.SGD(self.parameters(), lr = self.dynamics_lr)
        # SWAG store values
        #self.first_moment = utils.flatten(self.parameters()) # in tensor
        #self.second_moment = torch.square(self.first_moment) # in tensor
        #self.D = np.empty((self.first_moment.shape[0],0)) # in numpy
        #print(self.first_moment.size()) #531,461
        #print(self.second_moment.size()) #531,461       
        n_swag = 1
        step = 0
        # Start training loop
        for epoch in range(self.epochs):
            
            # LR Schedule 
            #update_lr = utils.schedule(epoch, lr_init, epochs, swa=True, swa_start=self.swag_start, swa_lr=swa_lr)
            #utils.adjust_learning_rate(self.optimizer, update_lr)

            for i, batch in enumerate(tqdm(dataloader)):
                # Split batch into input and output
                feed, target = batch
                if torch.cuda.is_available():
                    feed, target = feed.cuda(), target.cuda()
                
                self.optimizer.zero_grad()



                #feed forward
                next_state_pred, var = self.forward(feed)

                #backward
                self.loss1 = torch.mean(torch.exp(-var)*torch.square(next_state_pred-target))
                self.loss2 = torch.mean(var)
                self.loss = 0.5*(self.loss1+self.loss2)
                self.loss.backward()
                self.optimizer.step()
        
                # Tensorboard
                if(summary_writer is not None):
                    for j, loss_val in enumerate(loss_vals):
                        summary_writer.add_scalar('Loss/dynamics_{}'.format(j), self.loss, epoch*len(dataloader) + i)
                        summary_writer.add_scalar('Var/dynamics_{}'.format(j), torch.exp(self.v), epoch*len(dataloader) + i)


                if(comet_experiment is not None and i % 10 == 0):
                    for j, loss_val in enumerate(loss_vals):
                        comet_experiment.log_metric('dyn_model_{}_loss'.format(j), loss_val, epoch*len(dataloader) + i)
                        comet_experiment.log_metric('dyn_model_avg_loss'.format(j), sum(loss_vals)/len(loss_vals), epoch*len(dataloader) + i)
            
            if epoch >= 0:
                new_weights = utils.flatten(self.parameters()) # tensor
                self.first_moment = (n_swag*self.first_moment+new_weights)/(n_swag+1) # tensor
                self.second_moment = (n_swag*self.second_moment+new_weights**2)/(n_swag+1) # tensor
                print(f' first_moment is {self.first_moment[10]}')
                print(f'second moment is {self.second_moment[10]}')
                if self.D.shape[1]==self.k_swag:
                    self.D = np.delete(self.D, 0, 1) #delete first col
                new_D = (new_weights-self.first_moment).cpu().detach().numpy()
                print(f'new D is {new_D[10]}')

                #print(f'newD current shape {new_D.shape}')
                #print(f'newD current shape {new_D.reshape(new_D.shape[0],1)}')
                
                self.D = np.append(self.D, new_D.reshape(new_D.shape[0],1), axis = 1) # numpy
                #print(f'D current shape {self.D.shape}')
                n_swag +=1
        print(f'weight {sum(p.numel() for p in self.parameters())}')
        print(f'D is {self.D}')



    def swag(self):
        param_dict = {}
        param_dict['theta_swa'] = self.first_moment
        param_dict['sigma_diag'] = self.second_moment- self.first_moment**2
        param_dict['D'] = self.D
        param_dict['K'] = self.k_swag
        return param_dict        

    def load_swag(self, net):
        param_dict = {}
        param_dict['theta_swa'] = net['theta_swa']
        param_dict['sigma_diag'] = net['sigma_diag']
        param_dict['D'] = net['D']
        param_dict['K'] = net['K']
        return param_dict 

    def usad(self, predictions):
        # Compute the pairwise distances between all predictions
        distances = scipy.spatial.distance_matrix(predictions, predictions)

        # If maximum is greater than threshold, return true
        return (np.amax(distances) > self.threshold)

    def predict(self, x):
        # Generate prediction of next state using dynamics model
        with torch.set_grad_enabled(False):
            return self.forward(x)[0]

    def save(self, save_dir, idx):
        param = self.state_dict()
        param.update(self.swag())
        torch.save(param, os.path.join(save_dir, f'dynamics{idx}.pt'))
    def load(self, load_dir, idx):
        net = torch.load(os.path.join(load_dir, "dynamics.pt"))
        param_dict = self.load_swag(net)
        self.load_state_dict(torch.load(os.path.join(load_dir, f'dynamics{idx}.pt')), strict=False)
        self.cuda()
        return param_dict

class BasicNet(nn.Module):

    def __init__(self, input_dim,  output_dim, n_neurons, activation = nn.ReLU):
        super(BasicNet, self).__init__()

        layers = [nn.Linear(input_dim, n_neurons),
                nn.Linear(n_neurons, n_neurons),
                activation(),
                nn.Linear(n_neurons, n_neurons),
                activation(),
                nn.Linear(n_neurons, output_dim)]
        torch.nn.init.xavier_uniform(layers[0].weight)
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class DynamicsEnsemble():
    def __init__(self, input_dim, output_dim, n_models = 4, n_neurons = 512, threshold = 1.5, n_layers = 2, activation = nn.ReLU, cuda = True):
        self.n_models = 4

        self.threshold = threshold

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.models = []

        for i in range(n_models):
            if(cuda):
                self.models.append(DynamicsNet(input_dim,
                                            output_dim,
                                            n_neurons = n_neurons,
                                            activation = activation).cuda())
            else:
                self.models.append(DynamicsNet(input_dim,
                                            output_dim,
                                            n_neurons = n_neurons,
                                            activation = activation))

    def forward(self, model, x):
        return self.models[model](x)

    def train_step(self, model_idx, feed, target):
        # Reset Gradients
        self.optimizers[model_idx].zero_grad()

        # Feed forward
        next_state_pred, var = self.models[model_idx](feed)
        #output = self.losses[model_idx](next_state_pred, target)

        # Feed forward
        self.loss1 = torch.mean(torch.exp(-var)*torch.square(next_state_pred-target))
        self.loss2 = torch.mean(var)
        self.losses[model_idx] = 0.5*(self.loss1+self.loss2)
        output= self.losses[model_idx]

        # Feed backwards
        output.backward()

        # Weight update
        self.optimizers[model_idx].step()

        # Tensorboard
        return output


    def train(self, dataloader, epochs = 5, optimizer = torch.optim.Adam, loss = nn.MSELoss, summary_writer = None, comet_experiment = None):

        hyper_params = {
            "dynamics_n_models":  self.n_models,
            "usad_threshold": self.threshold,
            "dynamics_epochs" : 5
        }
        if(comet_experiment is not None):
            comet_experiment.log_parameters(hyper_params)

        # Define optimizers and loss functions
        self.optimizers = [None] * self.n_models
        self.losses = [None] * self.n_models

        for i in range(self.n_models):
            self.optimizers[i] = optimizer(self.models[i].parameters())
            #self.losses[i] = loss()

        # Start training loop
        for epoch in range(epochs):
            for i, batch in enumerate(tqdm(dataloader)):
                # Split batch into input and output
                feed, target = batch

                loss_vals = list(map(lambda i: self.train_step(i, feed, target), range(self.n_models)))

                # Tensorboard
                if(summary_writer is not None):
                    for j, loss_val in enumerate(loss_vals):
                        summary_writer.add_scalar('Loss/dynamics_{}'.format(j), loss_val, epoch*len(dataloader) + i)

                if(comet_experiment is not None and i % 10 == 0):
                    for j, loss_val in enumerate(loss_vals):
                        comet_experiment.log_metric('dyn_model_{}_loss'.format(j), loss_val, epoch*len(dataloader) + i)
                        comet_experiment.log_metric('dyn_model_avg_loss'.format(j), sum(loss_vals)/len(loss_vals), epoch*len(dataloader) + i)


    def usad(self, predictions):
        # Compute the pairwise distances between all predictions
        distances = scipy.spatial.distance_matrix(predictions, predictions)

        # If maximum is greater than threshold, return true
        return (np.amax(distances) > self.threshold)

    def predict(self, x):
        # Generate prediction of next state using dynamics model
        with torch.set_grad_enabled(False):
            return torch.stack(list(map(lambda i: self.forward(i, x)[0], range(self.n_models))))

    def save(self, save_dir):
        for i in range(self.n_models):
            torch.save(self.models[i].state_dict(), os.path.join(save_dir, "dynamics_{}.pt".format(i)))

    def load(self, load_dir):
        for i in range(self.n_models):
            self.models[i].load_state_dict(torch.load(os.path.join(load_dir, "dynamics_{}.pt".format(i))))

