#!/bin/bash
# Monitor test progress

echo "=========================================="
echo "Test Progress Monitor"
echo "=========================================="
echo ""

echo "Insertion Test:"
echo "==============="
if [ -f /workspace/NGFix/data/t2i-10M/insertion_percentage_results/summary.csv ]; then
    echo "Summary:"
    cat /workspace/NGFix/data/t2i-10M/insertion_percentage_results/summary.csv
    COMPLETED=$(tail -n +2 /workspace/NGFix/data/t2i-10M/insertion_percentage_results/summary.csv | wc -l)
    echo ""
    echo "Completed: $COMPLETED / 9 tests"
else
    echo "Summary file not found yet"
fi

echo ""
echo "Files created:"
ls -1 /workspace/NGFix/data/t2i-10M/insertion_percentage_results/index_insert_*.index 2>/dev/null | wc -l
echo "index files"

echo ""
echo "Deletion Test:"
echo "=============="
if [ -f /workspace/NGFix/data/t2i-10M/deletion_percentage_results/summary.csv ]; then
    echo "Summary:"
    cat /workspace/NGFix/data/t2i-10M/deletion_percentage_results/summary.csv
    COMPLETED=$(tail -n +2 /workspace/NGFix/data/t2i-10M/deletion_percentage_results/summary.csv | wc -l)
    echo ""
    echo "Completed: $COMPLETED / 9 tests"
else
    echo "Summary file not found yet"
fi

echo ""
echo "Files created:"
ls -1 /workspace/NGFix/data/t2i-10M/deletion_percentage_results/index_lazy_delete_*.index 2>/dev/null | wc -l
echo "lazy delete index files"

echo ""
echo "Running Processes:"
echo "=================="
ps aux | grep -E "(test_insertion_percentage|test_deletion_percentage)" | grep -v grep | head -2

echo ""
echo "Latest Log Activity:"
echo "===================="
echo "Insertion (last 3 lines):"
tail -3 /tmp/insertion_final.log 2>/dev/null || echo "No log yet"
echo ""
echo "Deletion (last 3 lines):"
tail -3 /tmp/deletion_final.log 2>/dev/null || echo "No log yet"

