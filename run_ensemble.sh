#!/bin/bash

##ens 10
python train.py --mdl=ensemble --dynamics_epochs=2 --k_swag=2 --threshold=1 --time_steps=100000 > ens3.txt

##python train.py --mdl=ensemble --dynamics_epochs=2 --k_swag=2 --threshold=1 --time_steps=100000 > ens1.txt
##python train.py --mdl=ensemble --dynamics_epochs=2 --k_swag=2 --threshold=5 --time_steps=100000 > ens2.txt
##python train.py --mdl=ensemble --dynamics_epochs=2 --k_swag=2 --threshold=10 --time_steps=100000 > ens3.txt
