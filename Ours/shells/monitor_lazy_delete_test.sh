#!/bin/bash

# Monitor lazy delete test progress

TEST_DIR="/workspace/OOD-ANNS/Ours/data/lazy_delete_test"
LOG_FILE="$TEST_DIR/nohup.out"

echo "=== Lazy Delete Test Monitor ==="
echo ""

# Check if test is running
if pgrep -f "test_lazy_delete" > /dev/null; then
    echo "✅ Test is RUNNING"
    echo ""
    PID=$(pgrep -f "test_lazy_delete" | head -1)
    echo "Process ID: $PID"
    echo "CPU Usage: $(ps -p $PID -o %cpu --no-headers)%"
    echo "Memory Usage: $(ps -p $PID -o %mem --no-headers)%"
    echo ""
else
    echo "❌ Test is NOT running"
    echo ""
fi

# Show recent log
if [ -f "$LOG_FILE" ]; then
    echo "=== Recent Log (last 30 lines) ==="
    tail -30 "$LOG_FILE"
    echo ""
fi

# Show output files
echo "=== Output Files ==="
if [ -d "$TEST_DIR" ]; then
    ls -lh "$TEST_DIR"/*.csv 2>/dev/null | tail -5 || echo "No CSV files yet"
else
    echo "Test directory not found"
fi

echo ""
echo "To view full log: tail -f $LOG_FILE"
echo "To stop test: pkill -f test_lazy_delete"

