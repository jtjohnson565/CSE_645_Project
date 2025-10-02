#!/bin/bash
#SBATCH --job-name=baseline_prompts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=baseline_prompts_output.log
#SBATCH --error=baseline_prompts_error.log

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate qlora_env

srun python baseline_prompts.py
