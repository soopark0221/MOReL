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
    def __init__(self, input_dim, output_dim, n_neurons, activation):
        super(DynamicsNet, self).__init__()
        # Validate inputs
        assert input_dim > 0
        assert output_dim > 0
        assert n_neurons > 0

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons

        self.device = torch.device('cuda')
        # output : mean
        self.mean = nn.ModuleList()
        self.mean.append(BasicNet(self.input_dim, self.output_dim, self.n_neurons, activation))

        # output : var
        self.var = nn.ModuleList()
        self.var.append(BasicNet(self.input_dim, self.output_dim, self.n_neurons, activation))

    def forward(self, x):
        m = x
        v = x
        for layer in self.mean:
            m = layer(m)
        for layer in self.var:
            v = layer(v)
        return m, v



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
    def __init__(self, input_dim, output_dim, n_neurons, threshold, dynamics_epochs, n_models = 4, n_layers = 2, activation = nn.ReLU, cuda = True):
        self.n_models = n_models

        self.threshold = threshold
        self.epochs = dynamics_epochs

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


    def train(self, dataloader, optimizer = torch.optim.Adam, loss = nn.MSELoss, summary_writer = None, comet_experiment = None):

        hyper_params = {
            "dynamics_n_models":  self.n_models,
            "usad_threshold": self.threshold,
            "dynamics_epochs" : self.epochs
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
        for epoch in range(self.epochs):
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

