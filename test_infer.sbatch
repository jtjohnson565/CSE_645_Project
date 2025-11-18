#!/bin/bash
#SBATCH --job-name=qlora_test_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate qlora_env

srun --output logs/test_infer_output-$1-$2.log --error logs/test_infer_error-$1-$2.log python test_model.py --model_id $1 --checkpoint_step $2 --eval_model $3
