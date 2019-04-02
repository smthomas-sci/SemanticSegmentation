#!/bin/bash
#SBATCH --job-name=Augment
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G            
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=out/%A_%a.out
#SBATCH --error=out/%A_%a.err
#SBATCH --array=0

# Activate conda environment
source ~/.bashrc

# Setup array of scripts
commands=(

     "python 04_augment_data.py"
)    

# Execute commands
eval "${commands[${SLURM_ARRAY_TASK_ID}]}"




