#!/bin/bash
#SBATCH --job-name=qlora_test_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/test_infer_output.log
#SBATCH --error=logs/test_infer_error.log

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate qlora_env

srun python test_model.py
