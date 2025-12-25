#!/bin/bash

MEX=48
M=16
efC=500
efC_AKNN=1500
efC_delete=500
K=100

# Data paths
BASE_DATA_DIR="/workspace/RoarGraph/data/t2i-10M"
INDEX_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M"
RESULT_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/lazy_vs_real_deletion_results"

# Create result directory if it doesn't exist
mkdir -p ${RESULT_DIR}

# Input index (should be a NGFix index with addition edges)
INPUT_INDEX="${INDEX_DIR}/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_10M.index"

# Test data (10K queries)
TEST_QUERY_PATH="${BASE_DATA_DIR}/query.10k.fbin"
TEST_GT_PATH="${BASE_DATA_DIR}/groundtruth-computed.10k.ibin"

# Output result file
RESULT_CSV="${RESULT_DIR}/lazy_vs_real_deletion_enhanced_results.csv"

echo "=== Enhanced Lazy Deletion vs Real Deletion Comparison Test ==="
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

# Run the enhanced comparison test
echo "Starting enhanced lazy vs real deletion comparison test..."
echo "This test will run with multiple efSearch values: 100, 200, 300, 400, 500"
echo "And will track max_candidate_set_size for each query"
echo ""
cd /workspace/OOD-ANNS/NGFix
./build/test/test_lazy_vs_real_deletion_enhanced \
--test_query_path ${TEST_QUERY_PATH} \
--test_gt_path ${TEST_GT_PATH} \
--index_path ${INPUT_INDEX} \
--metric ip_float \
--K ${K} \
--efC_AKNN ${efC_AKNN} \
--efC_delete ${efC_delete} \
--result_path ${RESULT_CSV}

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Test completed successfully ==="
    echo "Results saved to: ${RESULT_CSV}"
    echo ""
    echo "First few lines of results:"
    head -20 ${RESULT_CSV}
else
    echo "Error: Test failed!"
    exit 1
fi


