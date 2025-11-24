#!/bin/bash
#SBATCH --job-name=qlora_train_real
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate qlora_env

srun --output logs/qlora_train_real/qlora_train_real_output-$4-$5-${SLURM_JOBID}.log --error logs/qlora_train_real/qlora_train_real_error-$4-$5-${SLURM_JOBID}.log python train_qlora_real.py --epochs $1 --r $2 --alpha $3 --synthetic_model_id $4 --synthetic_checkpoint_steps $5 --output_path real_checkpoints/qlora-output-$4-$5-${SLURM_JOBID}
