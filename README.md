# CSE 645 Project

### Configuration Instructions:
Login to the LARCC Cluster
```
ssh {User ID}@larcc.hpc.louisville.edu
```
Copy and paste the GitHub repository into a chosen folder
```
scp -r {Local Machine GitHub Repository Path} {User ID}@larcc.hpc.louisville.edu:{Desired Folder Path in LARCC Cluster}
```
Configure miniconda environment using the following instructions after navigating to the folder:
```
module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym
conda create --name qlora\_env
conda activate qlora\_env
pip3 install -r requirements.txt
pip3 install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
```
While in environment, add HuggingFace Token (Also OpenAI Token if Inference Using LangChain Metrics is Desired) to the environment using the following commands:
```
hf auth login
conda env config vars set HF_TOKEN={HuggingFace Token}
conda env config vars set OPENAI_API_KEY={Open AI Token}
```
Exit and login again to cluster to folder to run chosen batch scripts

### Batch Script Commands (Run in CSE_645_Project Directory)
Preprocessing Dataset:
```
sbatch baseline_prompts.sbatch
```
Fine-Tuning using QLoRA:
```
sbatch train_qlora.sbatch {epochs} {rank} {alpha} {bottom n out of 32 decoder blocks excluded from QLoRA injection}
```
Baseline Inferencing:
```
sbatch baseline_prompts.sbatch
```
Fine-Tuned Inferencing:
```
sbatch test_infer.sbatch {model job id} {checkpoint step} {evaluation model name}
```