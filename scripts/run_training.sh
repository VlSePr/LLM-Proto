#!/bin/bash
# Launch training on vast.ai
# Usage:
#   bash scripts/run_training.sh              # Train tiny model
#   bash scripts/run_training.sh small        # Train small model
#   bash scripts/run_training.sh medium --resume latest  # Resume medium

set -e

MODEL_SIZE="${1:-tiny}"
shift 2>/dev/null || true  # Allow additional args to pass through

echo "=== Training ${MODEL_SIZE} model ==="
echo "Extra args: $@"

# Run inside tmux so training survives SSH disconnects
SESSION_NAME="train_${MODEL_SIZE}"

tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "Session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
    exit 1
}

tmux new-session -d -s "$SESSION_NAME" \
    "python -m src.train \
        --model configs/model_${MODEL_SIZE}.yaml \
        --config configs/training.yaml \
        --wandb_run_name ${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S) \
        $@ \
    2>&1 | tee logs/train_${MODEL_SIZE}.log"

echo "Training started in tmux session: $SESSION_NAME"
echo "  Attach: tmux attach -t $SESSION_NAME"
echo "  Detach: Ctrl+B, D"
echo "  Kill:   tmux kill-session -t $SESSION_NAME"

# Create logs dir
mkdir -p logs
