#!/bin/bash

#SBATCH --job-name=weLoveML:)
#SBATCH --output=out.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00
#SBATCH --partition=hpc

cd /home/m226252/kcluster
python3 naive.py
