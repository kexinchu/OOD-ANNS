#!/bin/bash
# Unified Test Monitor Script
# Usage:
#   ./monitor.sh [test_type] [options]
#   test_type: motivation | insertion | deletion | noise | all
#   options:
#     --efs EFS_VALUE: For motivation test, specify efSearch value (default: 100)
#     --status: Show current status only (no continuous monitoring)
#     --wait: Wait until test completes (for motivation test)
#     --interval SECONDS: Check interval for continuous monitoring (default: 60)

set -e

TEST_TYPE=""
EFS=100
SHOW_STATUS_ONLY=false
WAIT_FOR_COMPLETION=false
CHECK_INTERVAL=60

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        motivation|insertion|deletion|noise|all)
            TEST_TYPE="$1"
            shift
            ;;
        --efs)
            EFS="$2"
            shift 2
            ;;
        --status)
            SHOW_STATUS_ONLY=true
            shift
            ;;
        --wait)
            WAIT_FOR_COMPLETION=true
            shift
            ;;
        --interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [motivation|insertion|deletion|noise|all] [--efs EFS_VALUE] [--status] [--wait] [--interval SECONDS]"
            exit 1
            ;;
    esac
done

# If no test type specified, show usage
if [ -z "$TEST_TYPE" ]; then
    echo "Usage: $0 [motivation|insertion|deletion|noise|all] [options]"
    echo ""
    echo "Test types:"
    echo "  motivation - Monitor NGFix motivation test"
    echo "  insertion  - Monitor insertion percentage test (efSearch=1000)"
    echo "  deletion   - Monitor deletion percentage test"
    echo "  noise      - Monitor noise test (noise_scale=0.01)"
    echo "  all        - Monitor all tests"
    echo ""
    echo "Options:"
    echo "  --efs EFS_VALUE     - For motivation test, specify efSearch value (default: 100)"
    echo "  --status             - Show current status only (no continuous monitoring)"
    echo "  --wait               - Wait until test completes (for motivation test)"
    echo "  --interval SECONDS   - Check interval for continuous monitoring (default: 60)"
    echo ""
    echo "Examples:"
    echo "  $0 motivation --efs 100 --status    # Check motivation test status"
    echo "  $0 motivation --efs 100 --wait       # Monitor motivation test until completion"
    echo "  $0 insertion                         # Check insertion test status"
    echo "  $0 all                               # Monitor all tests"
    exit 0
fi

# Function to monitor motivation test
monitor_motivation() {
    RESULT_CSV="/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_results_efs${EFS}.csv"
    LOG_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_nohup.log"
    EXPECTED_STAGES=12
    
    echo "=========================================="
    echo "Motivation Test Monitor (efSearch=${EFS})"
    echo "=========================================="
    
    # Check if process is running
    if ps aux | grep -v grep | grep -q "test_ngfix_motivation"; then
        echo "✓ Test is RUNNING"
        PID=$(ps aux | grep -v grep | grep "test_ngfix_motivation" | awk '{print $2}' | head -1)
        echo "  PID: $PID"
    else
        echo "✗ Test is NOT running"
    fi
    
    echo ""
    
    # Check result file
    if [ -f "$RESULT_CSV" ]; then
        echo "=== Current Results ==="
        STAGES=$(tail -n +2 "$RESULT_CSV" 2>/dev/null | wc -l)
        echo "Completed stages: $STAGES/$EXPECTED_STAGES"
        echo ""
        echo "Last 5 results:"
        tail -5 "$RESULT_CSV" | column -t -s',' 2>/dev/null || tail -5 "$RESULT_CSV"
    else
        echo "Result CSV not found yet: $RESULT_CSV"
    fi
    
    echo ""
    
    # Check log file
    if [ -f "$LOG_FILE" ]; then
        echo "=== Recent Log (last 10 lines) ==="
        tail -10 "$LOG_FILE"
    else
        echo "Log file not found: $LOG_FILE"
    fi
    
    # If wait mode, continuously monitor
    if [ "$WAIT_FOR_COMPLETION" = true ] && [ "$SHOW_STATUS_ONLY" = false ]; then
        echo ""
        echo "=== Continuous Monitoring (Ctrl+C to stop) ==="
        while ps aux | grep -v grep | grep -q "test_ngfix_motivation"; do
            if [ -f "$RESULT_CSV" ]; then
                COMPLETED=$(tail -n +2 "$RESULT_CSV" 2>/dev/null | wc -l)
                PERCENTAGE=$((COMPLETED * 100 / EXPECTED_STAGES))
                LAST_STAGE=$(tail -1 "$RESULT_CSV" 2>/dev/null | cut -d',' -f1)
                echo -ne "\r[$(date '+%H:%M:%S')] Progress: $COMPLETED/$EXPECTED_STAGES ($PERCENTAGE%) - Last: $LAST_STAGE"
            else
                echo -ne "\r[$(date '+%H:%M:%S')] Waiting for results file..."
            fi
            sleep $CHECK_INTERVAL
        done
        echo ""
        echo ""
        echo "=== Test Completed ==="
        
        # Generate report if test completed
        if [ -f "$RESULT_CSV" ]; then
            COMPLETED=$(tail -n +2 "$RESULT_CSV" | wc -l)
            if [ $COMPLETED -ge $EXPECTED_STAGES ]; then
                echo "Generating markdown report..."
                python3 /workspace/OOD-ANNS/NGFix/scripts/generate_motivation_report.py \
                    --input "$RESULT_CSV" \
                    --output "/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_report_efs${EFS}.md" 2>&1
                echo "Report generated!"
            fi
        fi
    fi
}

# Function to monitor insertion test
monitor_insertion() {
    echo "=========================================="
    echo "Insertion Test Monitor (efSearch=1000)"
    echo "=========================================="
    
    SUMMARY_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv"
    
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Current results:"
        cat "$SUMMARY_FILE"
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo ""
        echo "Progress: $COMPLETED / 9 tests completed"
    else
        echo "Summary file not found yet: $SUMMARY_FILE"
    fi
    
    echo ""
    echo "Files created:"
    ls -1 /workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/index_insert_*.index 2>/dev/null | wc -l
    echo "index files"
    
    echo ""
    echo "Running processes:"
    ps aux | grep "test_insertion_percentage" | grep -v grep | head -1 || echo "No running processes"
    
    echo ""
    echo "Latest log:"
    tail -5 /tmp/insertion_efs1000.log 2>/dev/null || echo "No log yet"
}

# Function to monitor deletion test
monitor_deletion() {
    echo "=========================================="
    echo "Deletion Test Monitor"
    echo "=========================================="
    
    SUMMARY_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/deletion_percentage_results/summary.csv"
    
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Current results:"
        cat "$SUMMARY_FILE"
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo ""
        echo "Progress: $COMPLETED / 9 tests completed"
    else
        echo "Summary file not found yet: $SUMMARY_FILE"
    fi
    
    echo ""
    echo "Files created:"
    ls -1 /workspace/OOD-ANNS/NGFix/data/t2i-10M/deletion_percentage_results/index_lazy_delete_*.index 2>/dev/null | wc -l
    echo "lazy delete index files"
    
    echo ""
    echo "Running processes:"
    ps aux | grep "test_deletion_percentage" | grep -v grep | head -1 || echo "No running processes"
    
    echo ""
    echo "Latest log:"
    tail -5 /tmp/deletion_final.log 2>/dev/null || echo "No log yet"
}

# Function to monitor noise test
monitor_noise() {
    echo "=========================================="
    echo "Noise Test Monitor (noise_scale=0.01)"
    echo "=========================================="
    
    RESULTS_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs1000_noise0.01"
    SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
    
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Current results:"
        cat "$SUMMARY_FILE"
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo ""
        echo "Progress: $COMPLETED / 9 tests completed"
    else
        echo "Summary file not found yet: $SUMMARY_FILE"
    fi
    
    echo ""
    echo "Comparison with no noise (efSearch=1000):"
    NOISE_FREE_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv"
    if [ -f "$NOISE_FREE_FILE" ]; then
        echo "No noise results:"
        head -4 "$NOISE_FREE_FILE"
    else
        echo "No noise results file not found"
    fi
    
    echo ""
    echo "Running processes:"
    ps aux | grep "test_insertion_percentage" | grep -v grep | head -1 || echo "No running processes"
    
    echo ""
    echo "Latest log:"
    tail -5 /tmp/insertion_with_noise.log 2>/dev/null || echo "No log yet"
}

# Main execution
case $TEST_TYPE in
    motivation)
        monitor_motivation
        ;;
    insertion)
        monitor_insertion
        ;;
    deletion)
        monitor_deletion
        ;;
    noise)
        monitor_noise
        ;;
    all)
        monitor_motivation
        echo ""
        echo ""
        monitor_insertion
        echo ""
        echo ""
        monitor_deletion
        echo ""
        echo ""
        monitor_noise
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac
