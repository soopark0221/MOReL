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

    def forward(self, x):
        m = x
        v = x
        for layer in self.mean:
            m = layer(m)
        for layer in self.var:
            v = layer(v)
        return m, v


    def usad(self, predictions):
        # Compute the pairwise distances between all predictions
        distances = scipy.spatial.distance_matrix(predictions, predictions)

        # If maximum is greater than threshold, return true
        return (np.amax(distances) > self.threshold)

    def predict(self, x):
        # Generate prediction of next state using dynamics model
        with torch.set_grad_enabled(False):
            return self.forward(x)[0]

    def save(self, save_dir):
        torch.save(self.state_dict(), os.path.join(save_dir, "dynamics.pt"))
        
    def load(self, load_dir):
        self.load_state_dict(torch.load(os.path.join(load_dir, "dynamics.pt")))


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

