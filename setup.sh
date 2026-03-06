EVAL_DATA_DIR=$1

python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install hydra-core
pip install wandb
pip install -r requirements.txt
echo "Dependencies installed"

# Install evaluation pipeline
git submodule update --init
mkdir lib/evaluation-pipeline-2025/evaluation_data
unzip $EVAL_DATA_DIR -d lib/evaluation-pipeline-2025/evaluation_data/
echo "Eval pipeline installed"

# Login
wandb login

echo "Completed setup"