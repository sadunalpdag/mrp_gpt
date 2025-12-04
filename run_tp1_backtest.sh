#!/bin/bash
# TP1 SCALP MASTER STRATEGY - Backtest Runner
# Run backtest with different configurations

echo "==============================================="
echo "TP1 SCALP MASTER STRATEGY - Backtest Runner"
echo "==============================================="
echo ""

# Check if Python dependencies are installed
echo "Checking dependencies..."
python3 -c "import pandas; import numpy; import requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required dependencies..."
    pip install -q pandas numpy requests
fi

echo "Dependencies OK"
echo ""

# Parse command line arguments
MODE=${1:-full}

case $MODE in
    test)
        echo "Running TEST MODE: Synthetic data validation"
        echo "-------------------------------------------"
        python3 test_tp1_scalp_master.py
        ;;
    
    quick)
        echo "Running QUICK MODE: 3 symbols, 7 days"
        echo "-------------------------------------------"
        python3 -c "
import tp1_scalp_master
tp1_scalp_master.DAYS_BACK = 7
tp1_scalp_master.MAX_SYMBOLS = 3
tp1_scalp_master.main()
"
        ;;
    
    medium)
        echo "Running MEDIUM MODE: 10 symbols, 30 days"
        echo "-------------------------------------------"
        python3 -c "
import tp1_scalp_master
tp1_scalp_master.DAYS_BACK = 30
tp1_scalp_master.MAX_SYMBOLS = 10
tp1_scalp_master.main()
"
        ;;
    
    full)
        echo "Running FULL MODE: All symbols, 90 days"
        echo "-------------------------------------------"
        echo "⚠️  This may take 30-60 minutes depending on API rate limits"
        echo ""
        python3 tp1_scalp_master.py
        ;;
    
    *)
        echo "Usage: $0 [test|quick|medium|full]"
        echo ""
        echo "Modes:"
        echo "  test   - Run test suite with synthetic data (no network required)"
        echo "  quick  - Backtest 3 symbols for 7 days"
        echo "  medium - Backtest 10 symbols for 30 days"
        echo "  full   - Backtest all USDT perpetual futures for 90 days"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "==============================================="
echo "Backtest complete!"
echo "==============================================="
