#!/bin/bash
# Setup script for vast.ai instances
# Run this once after launching a new instance:
#   bash scripts/setup_vastai.sh

set -e

echo "=== vast.ai Setup ==="

# 1. Install system dependencies
apt-get update -qq
apt-get install -y -qq git wget htop tmux

# 2. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create directories
mkdir -p data tokenizer_data checkpoints

# 4. Download/sync data (if stored remotely)
# Uncomment and configure one of:
#   - HuggingFace Hub:
#     huggingface-cli download <your-repo>/llm-proto-data --local-dir data
#   - S3:
#     aws s3 sync s3://your-bucket/llm-proto/data ./data
#   - rsync from another machine:
#     rsync -avz user@host:/path/to/data ./data

echo ""
echo "Setup complete! Next steps:"
echo "  1. Copy or download tokenized data to ./data/"
echo "  2. Copy tokenizer to ./tokenizer_data/"  
echo "  3. Run training: bash scripts/run_training.sh"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
