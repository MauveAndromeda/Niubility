#!/bin/bash
# Fast training for quick testing (10K steps, ~5 minutes)

export ULTRA_STEPS=10000
export ULTRA_SEED=123

echo "Starting fast training (10K steps)..."
python train_ultra.py

echo ""
echo "Training complete! Run evaluation:"
echo "  python eval_ultra.py"
