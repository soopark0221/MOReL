#!/bin/bash

python train.py --dynamics_epochs=2 --k_swag=2 --threshold=10 > log8.txt
python train.py --dynamics_epochs=2 --k_swag=2 --threshold=10 --mdl=ensemble > ens1.txt
python train.py --dynamics_epochs=3 --k_swag=3 --threshold=10 > log9.txt
python train.py --dynamics_epochs=4 --k_swag=4 --threshold=10 > log10.txt

##python train.py --dynamics_epochs=2 --k_swag=2 --threshold=3 > log1.txt
##python train.py --dynamics_epochs=2 --k_swag=2 --threshold=5 > log2.txt
##python train.py --dynamics_epochs=2 --k_swag=2 --threshold=7 > log3.txt
##python train.py --dynamics_epochs=2 --k_swag=2 --threshold=10 > log4.txt

##python train.py --dynamics_epochs=2 --k_swag=2 --threshold=15 > log5.txt
##python train.py --dynamics_epochs=10 --k_swag=5 --threshold=10 > log6.txt
##python train.py --dynamics_epochs=10 --k_swag=5 --threshold=15 > log7.txt