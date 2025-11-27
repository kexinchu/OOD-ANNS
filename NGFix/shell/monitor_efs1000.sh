#!/bin/bash
# Monitor efs=1000 test progress

echo "=========================================="
echo "Insertion Test Progress (efSearch=1000)"
echo "=========================================="
if [ -f /workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv ]; then
    echo "Current results:"
    cat /workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv
    COMPLETED=$(tail -n +2 /workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv 2>/dev/null | wc -l)
    echo ""
    echo "Progress: $COMPLETED / 9 tests completed"
else
    echo "Summary file not found yet"
fi

echo ""
echo "Latest log:"
tail -5 /tmp/insertion_efs1000.log 2>/dev/null || echo "No log yet"

echo ""
echo "Running processes:"
ps aux | grep "test_insertion_percentage" | grep -v grep | head -1

