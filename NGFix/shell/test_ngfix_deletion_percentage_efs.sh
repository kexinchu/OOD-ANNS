#!/bin/bash
# Test script for NGFix deletion percentage with different efSearch values
# Tests deletion of 1%, 2%, 3%, 4%, 5%, 10%, 15% of NGFix addition edges
# Tests at efSearch values: 100, 200, 300, 400, 500, 1000, 1500, 2000
# Measures: recall, latency, ndc

set -e

MEX=48
M=16
efC=500
efC_AKNN=1500
K=100

# Data paths
BASE_DATA_DIR="/workspace/RoarGraph/data/t2i-10M"
INDEX_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M"
RESULT_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/deletion_percentage_efs_results"

# Create result directory
mkdir -p ${RESULT_DIR}

# Input index
INPUT_INDEX="${INDEX_DIR}/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}.index"

# Test data
TEST_QUERY_PATH="${BASE_DATA_DIR}/query.10k.fbin"
TEST_GT_PATH="${BASE_DATA_DIR}/groundtruth-computed.10k.ibin"

# Output files
RESULT_CSV="${RESULT_DIR}/deletion_percentage_efs_results.csv"
LOG_FILE="${RESULT_DIR}/deletion_percentage_efs_test.log"

echo "=== NGFix Deletion Percentage Test (Multiple efSearch) ==="
echo "Input Index: ${INPUT_INDEX}"
echo "Test Query: ${TEST_QUERY_PATH}"
echo "Test GT: ${TEST_GT_PATH}"
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

# Function to run the test
run_test() {
    cd /workspace/OOD-ANNS/NGFix
    ./build/test/test_ngfix_deletion_percentage_efs \
        --test_query_path ${TEST_QUERY_PATH} \
        --test_gt_path ${TEST_GT_PATH} \
        --index_path ${INPUT_INDEX} \
        --metric ip_float \
        --K ${K} \
        --result_path ${RESULT_CSV} 2>&1 | tee ${LOG_FILE}
}

echo "Starting deletion percentage test with multiple efSearch values..."
echo "This will test deletion percentages: 1%, 2%, 3%, 4%, 5%, 10%, 15%"
echo "At efSearch values: 100, 200, 300, 400, 500, 1000, 1500, 2000"
echo ""
echo "This may take a while..."
echo ""

run_test

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Test completed successfully ==="
    echo "Results saved to: ${RESULT_CSV}"
    echo "Log saved to: ${LOG_FILE}"
    echo ""
    echo "Summary of results:"
    echo "-------------------"
    head -20 ${RESULT_CSV}
    echo "..."
    echo ""
    echo "Total lines: $(wc -l < ${RESULT_CSV})"
else
    echo ""
    echo "=== Test failed ==="
    echo "Check log file: ${LOG_FILE}"
    exit 1
fi




