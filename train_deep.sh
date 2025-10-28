#!/bin/bash
# Deep training for research (500K steps, ~4-8 hours)

export ULTRA_STEPS=500000
export ULTRA_SEED=2024

echo "Starting deep training (500K steps)..."
echo "This will take approximately 4-8 hours on CPU, 1-2 hours on GPU"
python train_ultra.py

echo ""
echo "Training complete! Run evaluation:"
echo "  python eval_ultra.py"
echo "  python seedeval.py"
