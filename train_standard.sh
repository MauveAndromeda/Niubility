#!/bin/bash
# Standard training for production models (100K steps, ~30-60 minutes)

export ULTRA_STEPS=100000
export ULTRA_SEED=42

echo "Starting standard training (100K steps)..."
echo "This will take approximately 30-60 minutes on CPU, 10-15 minutes on GPU"
python train_ultra.py

echo ""
echo "Training complete! Run evaluation:"
echo "  python eval_ultra.py"
echo "  python seedeval.py"
