#!/bin/bash
#SBATCH --job-name=qlora_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=test_infer_output.log
#SBATCH --error=test_infer_error.log

# Activate your venv directly
module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate qlora_env

# Run inference
srun python test_model.py
