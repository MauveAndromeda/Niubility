#!/bin/bash
# Check if UltraTrader is ready to run

echo "========================================"
echo "  UltraTrader Environment Check"
echo "========================================"
echo ""

# Check Python
echo "[1/5] Checking Python..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✓ $PYTHON_VERSION"
else
    echo "✗ Python not found"
    exit 1
fi
echo ""

# Check key packages
echo "[2/5] Checking key packages..."
MISSING=0

for pkg in torch gymnasium stable_baselines3 numpy matplotlib pytest; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
        echo "✓ $pkg ($VERSION)"
    else
        echo "✗ $pkg not installed"
        MISSING=$((MISSING + 1))
    fi
done
echo ""

if [ $MISSING -gt 0 ]; then
    echo "⚠ $MISSING package(s) missing. Run: pip install -r requirements_ultra.txt"
    exit 1
fi

# Check CUDA
echo "[3/5] Checking CUDA availability..."
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" = "True" ]; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    echo "✓ CUDA available (version: $CUDA_VERSION)"
else
    echo "⚠ CUDA not available (will use CPU - slower training)"
fi
echo ""

# Check core files
echo "[4/5] Checking core files..."
FILES="ultra_trading_env.py ultra_advanced_trading_system.py train_ultra.py eval_ultra.py test_ultra.py"
for file in $FILES; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file missing"
        exit 1
    fi
done
echo ""

# Run quick test
echo "[5/5] Running quick environment test..."
if python -c "from ultra_trading_env import UltraTradingEnv; env = UltraTradingEnv(); env.reset(); print('OK')" 2>/dev/null | grep -q "OK"; then
    echo "✓ Environment loads successfully"
else
    echo "✗ Environment failed to load"
    exit 1
fi
echo ""

echo "========================================"
echo "  ✓ All checks passed!"
echo "========================================"
echo ""
echo "You are ready to start training:"
echo "  ./quick_start.sh        (完整流程)"
echo "  ./train_fast.sh         (快速训练)"
echo "  pytest test_ultra.py    (运行测试)"
echo ""
