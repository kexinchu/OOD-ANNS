#!/bin/bash
# Monitor script for the 1-hour runtime update test

RESULT_FILE="/workspace/OOD-ANNS/Ours/data/runtime_update_test/runtime_update_results.csv"
LOG_FILE="/workspace/OOD-ANNS/Ours/data/runtime_update_test/nohup.out"
PID_FILE="/workspace/OOD-ANNS/Ours/data/runtime_update_test/test.pid"

echo "=== Runtime Update Test Monitor ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if test is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Test is running (PID: $PID)"
    else
        echo "✗ Test process not found (PID: $PID)"
    fi
else
    # Try to find the process
    PID=$(ps aux | grep test_runtime_update_end2end | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$PID" ]; then
        echo "✓ Test is running (PID: $PID)"
        echo $PID > "$PID_FILE"
    else
        echo "✗ Test process not found"
    fi
fi

echo ""

# Show latest results
if [ -f "$RESULT_FILE" ]; then
    echo "=== Latest Statistics (from CSV) ==="
    tail -5 "$RESULT_FILE" | while IFS=',' read -r timestamp qcount recall ndc latency searches inserts s_inserts f_inserts idx_size; do
        if [ "$timestamp" != "timestamp" ]; then
            echo "Time: $timestamp"
            echo "  Queries: $qcount | Recall: $recall | Latency: ${latency}ms | NDC: $ndc"
            echo "  Searches: $searches | Inserts: $s_inserts (failed: $f_inserts) | Index Size: $idx_size"
            echo ""
        fi
    done
else
    echo "Result file not created yet"
fi

echo ""

# Show recent log activity
if [ -f "$LOG_FILE" ]; then
    echo "=== Recent Log Activity ==="
    tail -3 "$LOG_FILE" | grep -E "(Minute|INSERT|Stopping|Complete)" | tail -3
fi

echo ""
echo "To view full log: tail -f $LOG_FILE"
echo "To view results: cat $RESULT_FILE"

