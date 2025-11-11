\# CSE 645 Project



Configuration Instructions:

 - Login to the LARCC Cluster

     ssh {User ID}@larcc.hpc.louisville.edu

 - Copy and paste the GitHub repository into a chosen folder

     scp -r {Local Machine GitHub Repository Path} {User ID}@larcc.hpc.louisville.edu:{Desired Folder Path in LARCC Cluster}

 - Configure miniconda environment using the following instructions after navigating to the folder:

     module load miniforge3/24.3.0-0-gcc-11.5.0-wkw4vym

     conda create --name qlora\_env

     conda activate qlora\_env

     pip3 install -r requirements.txt

     pip3 install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128

 - While in environment, add HuggingFace Token (Also OpenAI Token if Inference Using LangChain Metrics is Desired) to the environment using the following commands:

     hf auth login

     conda env config vars set HF\_TOKEN={HuggingFace Token}

     conda env config vars set OPENAI\_API\_KEY={Open AI Token}

 - Exit and login again to cluster to folder to run chosen batch scripts

 

