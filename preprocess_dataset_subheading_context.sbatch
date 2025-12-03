#!/bin/bash
#SBATCH --job-name=preprocess_dataset
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=500MB
#SBATCH --output=logs/preprocess_dataset_output.log
#SBATCH --error=logs/preprocess_dataset_error.log

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate qlora_env

srun python preprocess_dataset_subheading_context.py
