import argparse
import os
import torch

#self.dynamics.train(dataloader, epochs = 3, summary_writer = self.tensorboard_writer, comet_experiment = self.comet_experiment)
#self.policy.train(env, summary_writer = self.tensorboard_writer, comet_experiment = self.comet_experiment)

# train.py
# 1. batch_size
# 2. obs_dim
# 3. action_dim

# morel.py
# Dynamics Args
# 1. n_neurons
# 2. threshold

class DynamicsNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons = 512, activation = nn.ReLU, threshold = 1.0):

    def train(self, dataloader, epochs, summary_writer = None, comet_experiment = None):
        self.swag_start = 1
        self.k_swag = 2      


        # Define initial LR
        lr_init = 0.01
        swa_lr = 0.01


    def train(self, env, optimizer = torch.optim.Adam,
                        lr =  0.00027,
                        n_steps = 1024,
                        time_steps = 1e6,
                        clip_range = 0.2,
                        entropy_coef = 0.01,
                        value_coef = 0.5,
                        num_batches = 4,
                        gamma = 0.99,
                        lam = 0.95,
                        max_grad_norm = 0.5,
                        num_train_epochs = 4,
                        comet_experiment = None,
                        summary_writer = None,
                        render = False):

parser = argparse.ArgumentParser(description='MOREL/SWAG Training')
parser.add_argument(
    "--batch_size",
    type = int,
    default = 128,
    help = "input batch size ",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--swa_start",
    type=float,
    default=161,
    help="SWA start epoch number (default: 161)",
)
parser.add_argument(
    "--swa_lr", 
    type=float, 
    default=0.02, 
    help="SWA LR (default: 0.02)"
)
parser.add_argument(
    "--swa_c_epochs",
    type=int,
    default=1,
    help="SWA model collection frequency/cycle length in epochs (default: 1)",
)
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")






args = parser.parse_args()

args.device = None

use_cuda = torch.cuda.is_available()

if use_cuda:
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")