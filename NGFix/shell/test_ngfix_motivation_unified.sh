#!/bin/bash
# Unified NGFix Motivation Test Script
# Usage: 
#   ./test_ngfix_motivation_unified.sh [run|check|report|nohup] [--efs VALUE] [--test-size small|full]

set -e

# Default parameters
MEX=48
M=16
efC=500
efC_AKNN=1500
efs=100
K=100
TEST_SIZE="full"  # "small" for testing, "full" for production
ACTION="run"      # run, check, report, nohup

# Data paths
BASE_DATA_DIR="/workspace/RoarGraph/data/t2i-10M"
INDEX_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M"
RESULT_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        run|check|report|nohup)
            ACTION="$1"
            shift
            ;;
        --efs)
            efs="$2"
            shift 2
            ;;
        --test-size)
            TEST_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [run|check|report|nohup] [--efs VALUE] [--test-size small|full]"
            exit 1
            ;;
    esac
done

# Adjust data paths based on test size
if [ "$TEST_SIZE" = "small" ]; then
    # Use smaller dataset for testing
    TEST_QUERY_PATH="${BASE_DATA_DIR}/query.10k.fbin"
    TEST_GT_PATH="${BASE_DATA_DIR}/groundtruth-computed.10k.ibin"
    TRAIN_QUERY_PATH="${BASE_DATA_DIR}/query.train.10M.fbin"  # Still use full for now
    TRAIN_GT_PATH="${BASE_DATA_DIR}/train.gt.bin"
    echo "⚠ TEST MODE: Using small test dataset"
else
    # Full dataset
    TEST_QUERY_PATH="${BASE_DATA_DIR}/query.10k.fbin"
    TEST_GT_PATH="${BASE_DATA_DIR}/groundtruth-computed.10k.ibin"
    TRAIN_QUERY_PATH="${BASE_DATA_DIR}/query.train.10M.fbin"
    TRAIN_GT_PATH="${BASE_DATA_DIR}/train.gt.bin"
fi

# File paths
INPUT_INDEX="${INDEX_DIR}/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}.index"
RESULT_CSV="${RESULT_DIR}/motivation_test_results_efs${efs}.csv"
REPORT_MD="${RESULT_DIR}/motivation_test_report_efs${efs}.md"
LOG_FILE="${RESULT_DIR}/motivation_test_nohup.log"
EXPECTED_STAGES=12

mkdir -p ${RESULT_DIR}
cd /workspace/OOD-ANNS/NGFix

# Function: Run the test
run_test() {
    echo "=== NGFix Motivation Test ==="
    echo "Input Index: ${INPUT_INDEX}"
    echo "Test Query: ${TEST_QUERY_PATH}"
    echo "Test GT: ${TEST_GT_PATH}"
    echo "Train Query: ${TRAIN_QUERY_PATH}"
    echo "Train GT: ${TRAIN_GT_PATH}"
    echo "Result CSV: ${RESULT_CSV}"
    echo "efSearch: ${efs}"
    echo ""

    # Check if files exist
    for file in "${INPUT_INDEX}" "${TEST_QUERY_PATH}" "${TEST_GT_PATH}" "${TRAIN_QUERY_PATH}"; do
        if [ ! -f "$file" ]; then
            echo "Error: File not found: $file"
            exit 1
        fi
    done

    echo "Starting motivation test..."
    ./build/test/test_ngfix_motivation \
        --test_query_path ${TEST_QUERY_PATH} \
        --test_gt_path ${TEST_GT_PATH} \
        --train_query_path ${TRAIN_QUERY_PATH} \
        --train_gt_path ${TRAIN_GT_PATH} \
        --index_path ${INPUT_INDEX} \
        --metric ip_float \
        --K ${K} \
        --efs ${efs} \
        --efC_AKNN ${efC_AKNN} \
        --result_path ${RESULT_CSV}

    if [ $? -eq 0 ]; then
        echo ""
        echo "=== Test completed successfully ==="
        generate_report
    else
        echo "Error: Test failed!"
        exit 1
    fi
}

# Function: Check test status
check_test() {
    echo "=== Motivation Test Status ==="
    echo ""

    # Check if process is running
    if ps aux | grep -v grep | grep -q "test_ngfix_motivation"; then
        echo "✓ Test is RUNNING"
        PID=$(ps aux | grep -v grep | grep "test_ngfix_motivation" | awk '{print $2}' | head -1)
        echo "  PID: $PID"
    else
        echo "✗ Test is NOT running"
    fi

    echo ""

    # Check log file
    if [ -f "$LOG_FILE" ]; then
        echo "=== Recent Log (last 20 lines) ==="
        tail -20 "$LOG_FILE"
        echo ""
    else
        echo "Log file not found: $LOG_FILE"
    fi

    echo ""

    # Check result file
    if [ -f "$RESULT_CSV" ]; then
        echo "=== Current Results ==="
        echo "Total lines in CSV: $(wc -l < "$RESULT_CSV")"
        echo ""
        echo "Last 5 results:"
        tail -5 "$RESULT_CSV" | column -t -s',' 2>/dev/null || cat "$RESULT_CSV"
        echo ""
        
        # Count completed stages
        STAGES=$(tail -n +2 "$RESULT_CSV" 2>/dev/null | wc -l)
        echo "Completed stages: $STAGES / $EXPECTED_STAGES"
    else
        echo "Result CSV not found yet: $RESULT_CSV"
    fi
}

# Function: Generate report
generate_report() {
    if [ ! -f "$RESULT_CSV" ]; then
        echo "Error: Result file not found: $RESULT_CSV"
        exit 1
    fi

    echo "Generating markdown report..."
    python3 /workspace/OOD-ANNS/NGFix/scripts/generate_motivation_report.py \
        --input ${RESULT_CSV} \
        --output ${REPORT_MD} 2>&1

    if [ $? -eq 0 ] && [ -f "$REPORT_MD" ]; then
        echo "✓ Report generated: $REPORT_MD"
    else
        echo "⚠ Failed to generate report"
    fi
}

# Function: Run with nohup
run_nohup() {
    echo "=== Starting Motivation Test in Background ==="
    
    # Check if already running
    if ps aux | grep -v grep | grep -q "test_ngfix_motivation"; then
        echo "⚠ Test is already running!"
        PID=$(ps aux | grep -v grep | grep "test_ngfix_motivation" | awk '{print $2}' | head -1)
        echo "  PID: $PID"
        exit 1
    fi

    # Run with nohup
    nohup bash -c "
        cd /workspace/OOD-ANNS/NGFix
        ./shell/test_ngfix_motivation_unified.sh run --efs ${efs} --test-size ${TEST_SIZE}
    " > ${LOG_FILE} 2>&1 &

    echo "Test started in background with PID: $!"
    echo "Log file: ${LOG_FILE}"
    echo "Results will be saved to: ${RESULT_CSV}"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f ${LOG_FILE}"
    echo "  ./shell/test_ngfix_motivation_unified.sh check --efs ${efs}"
}

# Function: Generate final report
generate_final_report() {
    echo "=== Final Check and Report Generation ==="
    echo ""

    # Check if test is still running
    if ps aux | grep -v grep | grep -q "test_ngfix_motivation"; then
        echo "⚠ Test is still running..."
        PID=$(ps aux | grep -v grep | grep "test_ngfix_motivation" | awk '{print $2}' | head -1)
        echo "  PID: $PID"
        echo ""
        check_test
        echo ""
        echo "Please wait for the test to complete, then run this script again with 'report' action."
        exit 0
    fi

    echo "✓ Test process has finished"
    echo ""

    # Check results
    if [ ! -f "$RESULT_CSV" ]; then
        echo "✗ Result file not found: $RESULT_CSV"
        exit 1
    fi

    COMPLETED=$(tail -n +2 "$RESULT_CSV" 2>/dev/null | wc -l)
    echo "Completed stages: $COMPLETED / $EXPECTED_STAGES"
    echo ""

    if [ $COMPLETED -lt $EXPECTED_STAGES ]; then
        echo "⚠ Warning: Only $COMPLETED / $EXPECTED_STAGES stages completed"
        echo "Results may be incomplete."
        echo ""
    fi

    echo "=== All Results ==="
    cat "$RESULT_CSV" | column -t -s',' 2>/dev/null || cat "$RESULT_CSV"
    echo ""

    # Generate report
    generate_report

    if [ -f "$REPORT_MD" ]; then
        echo ""
        echo "=== Report Preview (first 50 lines) ==="
        head -50 "$REPORT_MD"
    fi
}

# Main execution
case $ACTION in
    run)
        run_test
        ;;
    check)
        check_test
        ;;
    report)
        generate_final_report
        ;;
    nohup)
        run_nohup
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Usage: $0 [run|check|report|nohup] [--efs VALUE] [--test-size small|full]"
        exit 1
        ;;
esac

