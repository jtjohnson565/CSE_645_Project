#!/bin/bash
#SBATCH --job-name=qlora_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=test_infer_output.log
#SBATCH --error=test_infer_error.log

# Activate your venv directly
source /home/v0kanu01/.conda/envs/qlora_env/lib/python3.10/venv/scripts/common/activate

# Run inference
python test_model.py
