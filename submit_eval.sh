#!/bin/bash
#SBATCH --job-name=SegEval
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

     "python 08_model_evaluation.py --batch_size 12 --dim 256 --num_classes 12 \
    --weights ./weights/5x_290_BS_24_PS_256_C_12_FT_False_E_15_LR_0.0001_WM_Fmodel_ResNet_UNet_less_params_all_12.h5 \
    --data /scratch/imb/Simon/SkinCancer/data/5x/TrainingData/5x_290/
    "
)    

# Execute commands
eval "${commands[${SLURM_ARRAY_TASK_ID}]}"




