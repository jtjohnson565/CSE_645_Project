#!/bin/bash
#SBATCH --job-name=qlora_test_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

nums=($(echo "$1\n" | grep -o -E '[0-9]+'))
output_path="logs/test_infer/test_infer_output"
error_path="logs/test_infer/test_infer_error"

for i in "${nums[@]}"; do
	output_path+="-$i"
	error_path+="-$i"
done

output_path+=".log"
error_path+=".log"

module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda activate qlora_env

srun --output $output_path --error $error_path python test_model.py --checkpoint_path $1 --eval_model $2 --eval_dataset_path $3
