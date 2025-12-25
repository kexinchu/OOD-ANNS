#!/bin/bash

# Monitor NGFix test and ensure it runs for 60 minutes

TEST_PID_FILE="/tmp/ngfix_test_pid.txt"
RESULT_FILE="/workspace/OOD-ANNS/Ours/data/runtime_update_test/runtime_update_results_ngfix.csv"
LOG_FILE="/workspace/OOD-ANNS/Ours/data/runtime_update_test/nohup_ngfix.out"
EXPECTED_MINUTES=60

# Get PID from running process
PID=$(ps aux | grep test_runtime_update_end2end_ngfix | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "ERROR: Test process not found!"
    exit 1
fi

echo "Monitoring test process PID: $PID"
echo "Expected duration: $EXPECTED_MINUTES minutes"
echo ""

# Function to check test status
check_status() {
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "ERROR: Test process $PID is not running!"
        return 1
    fi
    
    # Count minutes of data
    if [ -f "$RESULT_FILE" ]; then
        MINUTES=$(tail -n +2 "$RESULT_FILE" | wc -l)
        echo "Current progress: $MINUTES / $EXPECTED_MINUTES minutes"
        
        if [ $MINUTES -ge $EXPECTED_MINUTES ]; then
            echo "✅ Test completed! ($MINUTES minutes of data)"
            return 0
        else
            echo "⏳ Test in progress... ($MINUTES / $EXPECTED_MINUTES minutes)"
            return 2
        fi
    else
        echo "⏳ Waiting for first minute of data..."
        return 2
    fi
}

# Monitor loop
while true; do
    check_status
    STATUS=$?
    
    if [ $STATUS -eq 0 ]; then
        echo ""
        echo "=== Test Summary ==="
        tail -5 "$RESULT_FILE"
        echo ""
        echo "Test completed successfully!"
        break
    elif [ $STATUS -eq 1 ]; then
        echo "Test failed or stopped unexpectedly!"
        echo "Last log entries:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    
    sleep 60  # Check every minute
done
