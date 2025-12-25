#!/bin/bash
# Unified Motivation Test Script
# Usage: 
#   ./test_motivation.sh [--efs VALUE] [--nohup] [--test-mode]
#   --efs: efSearch value (default: 100)
#   --nohup: Run in background with nohup
#   --test-mode: Use small dataset for testing (100K train queries instead of 10M)

set -e

# Default parameters
MEX=48
M=16
efC=500
efC_AKNN=1500
efs=100
K=100
USE_NOHUP=false
TEST_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --efs)
            efs="$2"
            shift 2
            ;;
        --nohup)
            USE_NOHUP=true
            shift
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--efs VALUE] [--nohup] [--test-mode]"
            exit 1
            ;;
    esac
done

# Data paths
BASE_DATA_DIR="/workspace/RoarGraph/data/t2i-10M"
INDEX_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M"
RESULT_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results"

# Create result directory
mkdir -p ${RESULT_DIR}

# Input index
INPUT_INDEX="${INDEX_DIR}/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}.index"

# Test data (10K queries)
TEST_QUERY_PATH="${BASE_DATA_DIR}/query.10k.fbin"
TEST_GT_PATH="${BASE_DATA_DIR}/groundtruth-computed.10k.ibin"

# Train data - use small dataset in test mode
if [ "$TEST_MODE" = true ]; then
    echo "⚠ TEST MODE: Using 100K train queries instead of 10M"
    # For test mode, we'll limit the train queries in the C++ code
    # For now, we'll use the same path but the code will handle limiting
    TRAIN_QUERY_PATH="${BASE_DATA_DIR}/query.train.10M.fbin"
    TRAIN_GT_PATH="${BASE_DATA_DIR}/train.gt.bin"
    TRAIN_LIMIT=100000  # 100K for testing
else
    TRAIN_QUERY_PATH="${BASE_DATA_DIR}/query.train.10M.fbin"
    TRAIN_GT_PATH="${BASE_DATA_DIR}/train.gt.bin"
    TRAIN_LIMIT=10000000  # 10M for full test
fi

# Output files
RESULT_CSV="${RESULT_DIR}/motivation_test_results_efs${efs}.csv"
LOG_FILE="${RESULT_DIR}/motivation_test_nohup_efs${efs}.log"

echo "=== NGFix Motivation Test ==="
echo "efSearch: ${efs}"
echo "Test Mode: ${TEST_MODE}"
echo "Input Index: ${INPUT_INDEX}"
echo "Test Query: ${TEST_QUERY_PATH}"
echo "Test GT: ${TEST_GT_PATH}"
echo "Train Query: ${TRAIN_QUERY_PATH}"
echo "Train GT: ${TRAIN_GT_PATH}"
if [ "$TEST_MODE" = true ]; then
    echo "Train Limit: ${TRAIN_LIMIT} (test mode)"
fi
echo "Result CSV: ${RESULT_CSV}"
echo ""

# Check if files exist
if [ ! -f "${INPUT_INDEX}" ]; then
    echo "Error: Input index not found: ${INPUT_INDEX}"
    exit 1
fi

if [ ! -f "${TEST_QUERY_PATH}" ]; then
    echo "Error: Test query file not found: ${TEST_QUERY_PATH}"
    exit 1
fi

if [ ! -f "${TEST_GT_PATH}" ]; then
    echo "Error: Test GT file not found: ${TEST_GT_PATH}"
    exit 1
fi

if [ ! -f "${TRAIN_QUERY_PATH}" ]; then
    echo "Error: Train query file not found: ${TRAIN_QUERY_PATH}"
    exit 1
fi

# Build command
cd /workspace/OOD-ANNS/NGFix
CMD="./build/test/test_ngfix_motivation \
--test_query_path ${TEST_QUERY_PATH} \
--test_gt_path ${TEST_GT_PATH} \
--train_query_path ${TRAIN_QUERY_PATH} \
--train_gt_path ${TRAIN_GT_PATH} \
--index_path ${INPUT_INDEX} \
--metric ip_float \
--K ${K} \
--efs ${efs} \
--efC_AKNN ${efC_AKNN} \
--result_path ${RESULT_CSV}"

if [ "$TEST_MODE" = true ]; then
    # In test mode, we need to modify the C++ code or use a wrapper
    # For now, we'll just run with a note that test mode needs code modification
    echo "⚠ Note: Test mode requires code modification to limit train queries"
    echo "   For now, running full test. Modify test_ngfix_motivation.cc to support --train_limit"
fi

# Run test
if [ "$USE_NOHUP" = true ]; then
    echo "Starting test in background with nohup..."
    echo "Log file: ${LOG_FILE}"
    nohup bash -c "$CMD" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "Test started with PID: $PID"
    echo ""
    echo "To monitor progress:"
    echo "  ./shell/monitor_all.sh --motivation --efs ${efs}"
    echo ""
    echo "To check status:"
    echo "  ./shell/monitor_all.sh --motivation --efs ${efs} --status"
else
    echo "Starting motivation test..."
    $CMD
    
    if [ $? -eq 0 ]; then
    echo ""
    echo "=== Test completed successfully ==="
    echo "Results saved to: ${RESULT_CSV}"
    echo ""
    echo "Generating markdown report..."
    python3 /workspace/OOD-ANNS/NGFix/scripts/generate_motivation_report.py \
        --input ${RESULT_CSV} \
        --output ${RESULT_DIR}/motivation_test_report_efs${efs}.md 2>&1
    
    if [ $? -eq 0 ] && [ -f "${RESULT_DIR}/motivation_test_report_efs${efs}.md" ]; then
        echo "✓ Report generated: ${RESULT_DIR}/motivation_test_report_efs${efs}.md"
    else
        echo "⚠ Report generation failed or file not created"
    fi
    else
        echo "Error: Test failed!"
        exit 1
    fi
fi

