#!/bin/bash

python train.py --dynamics_epochs=2 --k_swag=2 --threshold=10 > log5.txt
python train.py --dynamics_epochs=10 --k_swag=5 --threshold=10 > log6.txt
python train.py --dynamics_epochs=10 --k_swag=5 --threshold=15 > log7.txt

##python train.py --dynamics_epochs=2 --k_swag=2 --threshold=3 > log1.txt
##python train.py --dynamics_epochs=2 --k_swag=2 --threshold=5 > log2.txt
##python train.py --dynamics_epochs=2 --k_swag=2 --threshold=7 > log3.txt
##python train.py --dynamics_epochs=2 --k_swag=2 --threshold=10 > log4.txt

