#!/bin/bash
#SBATCH --job-name=qlora_train
#SBATCH --output=logs/qlora_train_output.log
#SBATCH --error=logs/qlora_train_error.log
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate qlora_env

srun python train_qlora.py

