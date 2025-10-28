#!/bin/bash
# UltraTrader Quick Start Script
# 一键训练、测试、评估完整流程

set -e  # Exit on error

echo "=========================================="
echo "  UltraTrader Quick Start"
echo "=========================================="
echo ""

# Step 1: Run unit tests
echo "[1/4] Running unit tests..."
pytest test_ultra.py -v
if [ $? -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "✗ Tests failed. Please fix errors before training."
    exit 1
fi
echo ""

# Step 2: Quick training (20K steps)
echo "[2/4] Training model (20,000 steps)..."
export ULTRA_STEPS=20000
export ULTRA_SEED=123
python train_ultra.py
echo "✓ Training complete!"
echo ""

# Step 3: Single evaluation
echo "[3/4] Running single evaluation (10 episodes)..."
python eval_ultra.py
echo "✓ Evaluation complete!"
echo ""

# Step 4: Multi-seed evaluation
echo "[4/4] Running multi-seed evaluation (10 seeds × 5 episodes)..."
python seedeval.py
echo "✓ Multi-seed evaluation complete!"
echo ""

echo "=========================================="
echo "  Training & Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - checkpoints/best/best_model.zip"
echo "  - reports/equity_curve_best.png"
echo "  - reports/equity_multiseed.png"
echo "  - reports/seed_stats.csv"
echo ""
echo "Next steps:"
echo "  1. Check reports/ folder for visualizations"
echo "  2. Review seed_stats.csv for robustness metrics"
echo "  3. Adjust training parameters if needed"
echo ""
