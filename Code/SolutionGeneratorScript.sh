#!/bin/bash

# Slurm sbatch options

#SBATCH -n 48  #48
#    SBATCH --gres=gpu:volta:2
#SBATCH -N 1

#SBATCH -o SolutionGenerator.sh.log-%j
# Loading the required module

module load anaconda/Python-ML-2025a
module load gurobi/gurobi-1102

ln -sf $TMPDIR /tmp/ray_$SLURM_JOB_ID
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID

# Ray Memory Management Configuration
# Set memory usage threshold to 95% (default is 95%, but good to be explicit if needed)
export RAY_memory_usage_threshold=0.95
# Disable the memory monitor to prevent worker killing
export RAY_memory_monitor_refresh_ms=0

# Run the script
python -u SolutionGenerator.py