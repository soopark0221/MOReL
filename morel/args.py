import argparse
import os
import torch

#self.dynamics.train(dataloader, epochs = 3, summary_writer = self.tensorboard_writer, comet_experiment = self.comet_experiment)
#self.policy.train(env, summary_writer = self.tensorboard_writer, comet_experiment = self.comet_experiment)

#### train.py
# 1. batch_size
# 2. obs_dim
# 3. action_dim

##### morel.py
### Dynamics Args
# n_neurons
# threshold
# dynamics_lr
# swa_lr
# swag_start
# k_swag
# epochs


parser = argparse.ArgumentParser(description='MOREL/SWAG Training')

parser.add_argument(
    "--obs_dim",
    type = int,
    default = 4,
    help = "observation dimension ",
)

parser.add_argument(
    "--action_dim",
    type = int,
    default = 2,
    help = "action dimension ",
)
parser.add_argument(
    "--n_neurons",
    type = int,
    default = 512,
    help = "num of neurons ",
)

parser.add_argument(
    "--threshold",
    type = int,
    default = 1,
    help = "threshold ",
)
parser.add_argument(
    "--batch_size",
    type = int,
    default = 128,
    help = "input batch size ",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--dynamics_lr",
    type=float,
    default=0.01,
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--swa_start",
    type=int,
    default=5,
    help="SWA start epoch number (default: 5)",
)
parser.add_argument(
    "--swa_lr", 
    type=float, 
    default=0.02, 
    help="SWA LR (default: 0.02)"
)

parser.add_argument(
    "--k_swag", 
    type=int, 
    default=3, 
    help="K swag (default: 3)"
)


### Policy Args
# optimizer
# policy_lr
# n_steps
# time_steps
# clip_range
# entropy_coef
# num_batches
# gamma
# lam
# max_grad_norm
# num_train_epochs

parser.add_argument(
    "--policy_lr",
    type=float,
    default=0.00025,
    help="policy lr",
)
parser.add_argument(
    "--n_steps", 
    type=int, 
    default=1024, 
    help="Length of steps for rollouts"
)

parser.add_argument(
    "--time_steps", 
    type=int, 
    default=1e6, 
    help="Total number of steps of experience to collect"
)

parser.add_argument(
    "--clip_range",
    type=float,
    default=0.02,
    help="Clip range for the value function and ratio in policy loss",
)
parser.add_argument(
    "--entropy_coef",
    type=float,
    default=0.01,
    help="Coefficient for entropy loss in total loss function",
)
parser.add_argument(
    "--value_coef",
    type=float,
    default=0.5,
    help="Coefficient for value loss in total loss function",
)

parser.add_argument(
    "--policy_num_batches",
    type=int,
    default=4,
    help="Number of batches to split rollout into",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="Discount factor used for generalized advantage estimate",
)
parser.add_argument(
    "--lam",
    type=float,
    default=0.95,
    help="Discount factor used for generalized advantage estimate",
)
parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=0.5,
    help="Max gradient norm for gradient clipping",
)

parser.add_argument(
    "--policy_num_train_epochs",
    type=int,
    default=4,
    help="Number of epochs to train for after each new experience rollout is generated",
)

args = parser.parse_args()

args.device = None

use_cuda = torch.cuda.is_available()

if use_cuda:
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

