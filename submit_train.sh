#!/bin/bash
#SBATCH --job-name=SegTrain
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G            
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:2
#SBATCH --output=out/%A_%a.out
#SBATCH --error=out/%A_%a.err
#SBATCH --array=0

# Activate conda environment
source ~/.bashrc

# Setup array of scripts
commands=(

     "python 05_patch_training.py --batch_size 24 --epochs 15 --learning_rate 0.0001 \
    --dim 256 --num_classes 12 --gpus 2  --log_dir ./logs/ \
    --data /scratch/imb/Simon/SkinCancer/data/5x/TrainingData_90/5x_290_Over \
    "
)    

# Execute commands
eval "${commands[${SLURM_ARRAY_TASK_ID}]}"




