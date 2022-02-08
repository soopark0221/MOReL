
#general

import argparse
import json
import subprocess
import numpy as np
from tqdm import tqdm
import os
import glob
import tarfile
from comet_ml import Experiment

# dataset imports
import gym
import d4rl

from morel.morel import Morel
#from morel_multiswag.morel import Morel_multiswag
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import itertools


class Maze2DDataset(Dataset):

    def __init__(self):
        self.env = gym.make('maze2d-umaze-v1')
        #dataset = dict(itertools.islice(self.env.get_dataset().items(), 64))
        dataset = self.env.get_dataset()

        # Input data
        self.source_observation = dataset["observations"][:-1]
        self.source_action = dataset["actions"][:-1]


        # Output data
        self.target_delta = dataset["observations"][1:] - self.source_observation
        self.target_reward = dataset["rewards"][:-1]

        # Normalize data
        self.delta_mean = self.target_delta.mean(axis=0)
        self.delta_std = self.target_delta.std(axis=0)

        self.reward_mean = self.target_reward.mean(axis=0)
        self.reward_std = self.target_reward.std(axis=0)

        self.observation_mean = self.source_observation.mean(axis=0)
        self.observation_std = self.source_observation.std(axis=0)

        self.action_mean = self.source_action.mean(axis=0)
        self.action_std = self.source_action.std(axis=0)

        self.source_action = (self.source_action - self.action_mean)/self.action_std
        self.source_observation = (self.source_observation - self.observation_mean)/self.observation_std
        self.target_delta = (self.target_delta - self.delta_mean)/self.delta_std
        self.target_reward = (self.target_reward - self.reward_mean)/self.reward_std

        # Get indices of initial states
        self.done_indices = dataset["timeouts"][:-1]
        self.initial_indices = np.roll(self.done_indices, 1)
        self.initial_indices[0] = True

        # Calculate distribution parameters for initial states
        self.initial_obs = self.source_observation[self.initial_indices]
        self.initial_obs_mean = self.initial_obs.mean(axis = 0)
        self.initial_obs_std = self.initial_obs.std(axis = 0)

        # Remove transitions from terminal to initial states
        self.source_action = np.delete(self.source_action, self.done_indices, axis = 0)
        self.source_observation = np.delete(self.source_observation, self.done_indices, axis = 0)
        self.target_delta = np.delete(self.target_delta, self.done_indices, axis = 0)
        self.target_reward = np.delete(self.target_reward, self.done_indices, axis = 0)



    def __getitem__(self, idx):
        feed = torch.FloatTensor(np.concatenate([self.source_observation[idx], self.source_action[idx]])).to("cuda:0")
        target = torch.FloatTensor(np.concatenate([self.target_delta[idx], self.target_reward[idx:idx+1]])).to("cuda:0")

        return feed, target

    def __len__(self):
        return len(self.source_observation)

def upload_assets(comet_experiment, log_dir):
    tar_path = log_dir + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(log_dir, arcname=os.path.basename(log_dir))

    comet_experiment.log_asset(tar_path)
    os.remove(tar_path)

def main(args):
    tensorboard_writer = None
    comet_experiment = None

    if(not args.no_log):
        # Create necessary directories
        if(not os.path.isdir(args.log_dir)):
            os.mkdir(args.log_dir)

        # Create log_dir for run
        run_log_dir = os.path.join(args.log_dir,args.exp_name)
        if(os.path.isdir(run_log_dir)):
            cur_count = len(glob.glob(run_log_dir + "_*"))
            run_log_dir = run_log_dir + "_" + str(cur_count)
        os.mkdir(run_log_dir)

        # Create tensorboard writer if requested

        if(args.tensorboard):
            tensorboard_dir = os.path.join(run_log_dir, "tensorboard")
            writer = SummaryWriter(log_dir = tensorboard_dir)


    # Create comet experiment if requested
    if(args.comet_config is not None):
        with open(args.comet_config, 'r') as f:
            comet_dict = json.load(f)
            comet_experiment = Experiment(
                api_key = comet_dict["api_key"],
                project_name = comet_dict["project_name"],
                workspace = comet_dict["workspace"],
            )
            comet_experiment.set_name(args.exp_name)

            # Get hash for latest git commit for loggings
            last_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").rstrip()
            comet_experiment.log_parameter("git_commit_id", last_commit_hash)     

    # Instantiate dataset
    dynamics_data = Maze2DDataset()

    dataloader = DataLoader(dynamics_data, args.batch_size, shuffle = True)


    agent = Morel(args.obs_dim, args.action_dim, args.n_neurons, args.threshold, args.batch_size, args.dynamics_epochs, args.dynamics_lr, args.swa_lr, args.swa_start, args.k_swag, args.s_swag, \
            args.policy_lr, args.n_steps, args.time_steps, args.clip_range, args.entropy_coef, args.value_coef, args.policy_num_batches, args.gamma, args.lam, args.max_grad_norm, args.policy_num_train_epochs,\
            args.mdl, args.log_dir, args.resume, args.idx, tensorboard_writer = tensorboard_writer, comet_experiment = comet_experiment)
        

    agent.train(dataloader, dynamics_data)

    #if(not args.no_log):
    #agent.save(os.path.join(run_log_dir, "models"))
    if comet_experiment is not None:
        upload_assets(comet_experiment, run_log_dir)

    agent.eval(dynamics_data.env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--mdl', type=str, default='swag')
    parser.add_argument('--log_dir', type=str, default='results/')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--comet_config', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='exp_test')
    parser.add_argument('--no_log', action='store_true')
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
        type = float,
        default = 1.5,
        help = "threshold ",
    )
    parser.add_argument(
        "--idx",
        type = int,
        default = 1,
        help = "ckpt idx",
    )    
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 128,
        help = "input batch size ",
    )
    parser.add_argument(
        "--dynamics_epochs",
        type=int,
        default=5,
        help="number of epochs to train (default: 5)",
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

    parser.add_argument(
        "--s_swag", 
        type=int, 
        default= 10, 
        help="S swag (default: 10)"
    )    

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


    parser.add_argument(
        "--filename",
        type=str,
        default='dynamics',
        help="Dynamics pt filename",
    )
    args = parser.parse_args()
    main(args)







