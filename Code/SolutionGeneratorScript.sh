#!/bin/bash

# Slurm sbatch options

#SBATCH -n 48
#   SBATCH --gres=gpu:volta:2
#SBATCH -N 1

#SBATCH -o SolutionGenerator.sh.log-%j
# Loading the required module

module load anaconda/Python-ML-2025a
module load gurobi/gurobi-1102

ln -sf $TMPDIR /tmp/ray_$SLURM_JOB_ID
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID

# Run the script
python SolutionGenerator.py