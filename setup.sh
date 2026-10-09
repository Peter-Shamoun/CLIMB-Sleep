EVAL_DATA_DIR=$1
# Install dependencies
pip install --upgrade uv
uv sync # also creates virtual environment
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
echo "Dependencies installed"

# Install evaluation pipeline
git submodule update --init
mkdir lib/evaluation-pipeline-2025/evaluation_data
unzip $EVAL_DATA_DIR -d lib/evaluation-pipeline-2025/evaluation_data/
echo "Eval pipeline installed"

# Login
.venv/bin/wandb login

echo "Completed setup"