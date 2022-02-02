import argparse
import os
import torch

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