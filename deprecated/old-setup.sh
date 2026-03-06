
module rm rhel7/global
module rm rhel7/default-gpu

if [ ! -d "env" ]; then
	pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
	pip install hydra-core
	pip install wandb
	pip install -r requirements.txt
	huggingface-cli login
	wandb login --relogin $WANDB_API_KEY
else 
	source env/bin/activate
fi
source .env


export PATH="$(pwd)/lib/bin:$PATH"
