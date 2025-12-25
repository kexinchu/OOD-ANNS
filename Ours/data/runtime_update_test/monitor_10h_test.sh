#!/bin/bash

# Monitor 10-hour test progress

RESULT_FILE="/workspace/OOD-ANNS/Ours/data/runtime_update_test/runtime_update_results_ngfix.csv"
LOG_FILE="/workspace/OOD-ANNS/Ours/data/runtime_update_test/nohup_ngfix.out"
EXPECTED_MINUTES=600

echo "=== 10小时测试监控 ==="
echo "开始时间: $(date)"
echo ""

while true; do
    # Get PID
    PID=$(ps aux | grep test_runtime_update_end2end_ngfix | grep -v grep | awk '{print $2}' | head -1)
    
    if [ -z "$PID" ]; then
        echo "$(date): ERROR - Test process not found!"
        echo "Last log entries:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    
    # Count minutes
    if [ -f "$RESULT_FILE" ]; then
        MINUTES=$(tail -n +2 "$RESULT_FILE" 2>/dev/null | wc -l)
        REMAINING=$((EXPECTED_MINUTES - MINUTES))
        PERCENT=$((MINUTES * 100 / EXPECTED_MINUTES))
        
        echo "$(date): Progress: $MINUTES/$EXPECTED_MINUTES minutes ($PERCENT%) - Remaining: $REMAINING minutes"
        
        if [ $MINUTES -ge $EXPECTED_MINUTES ]; then
            echo ""
            echo "✅ Test completed! ($MINUTES minutes)"
            echo "Final statistics:"
            tail -5 "$RESULT_FILE"
            break
        fi
    else
        echo "$(date): Waiting for first minute of data..."
    fi
    
    sleep 300  # Check every 5 minutes
done

echo ""
echo "Test monitoring complete at $(date)"
