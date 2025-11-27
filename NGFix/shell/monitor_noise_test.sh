#!/bin/bash
# Monitor noise test progress

echo "=========================================="
echo "Insertion Test with Noise (noise_scale=0.01)"
echo "=========================================="

RESULTS_DIR="/workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000_noise0.01"

if [ -f "${RESULTS_DIR}/summary.csv" ]; then
    echo "Current results:"
    cat "${RESULTS_DIR}/summary.csv"
    COMPLETED=$(tail -n +2 "${RESULTS_DIR}/summary.csv" 2>/dev/null | wc -l)
    echo ""
    echo "Progress: $COMPLETED / 9 tests completed"
else
    echo "Summary file not found yet"
fi

echo ""
echo "Comparison with no noise (efSearch=1000):"
if [ -f "/workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv" ]; then
    echo "No noise results:"
    cat /workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv | head -4
else
    echo "No noise results file not found"
fi

echo ""
echo "Latest log:"
tail -5 /tmp/insertion_with_noise.log 2>/dev/null || echo "No log yet"

echo ""
echo "Running processes:"
ps aux | grep "test_insertion_percentage" | grep -v grep | head -1

