#!/bin/bash

#SBATCH --job-name=we<3ML:)
#SBATCH --output=writeout
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:00:00
#SBATCH --partition=hpc

cd /home/m226252/kcluster
python3 execute.py full 3500 preSVD.npz f t 1 2 1
