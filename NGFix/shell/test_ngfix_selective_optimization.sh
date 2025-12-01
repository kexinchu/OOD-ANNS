#!/bin/bash

MEX=48
M=16
efC=500
K=100
NUM_TEST_QUERIES=10000

# Base HNSW index (M=16, not optimized with NGFix)
BASE_INDEX_PATH=/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index

# Full NGFix index (control group - optimized with all queries)
FULL_NGFIX_INDEX_PATH=/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN1500.index

# Data paths
TRAIN_QUERY_PATH=/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin
TRAIN_GT_PATH=/workspace/RoarGraph/data/t2i-10M/train.gt.bin
TEST_QUERY_PATH=/workspace/RoarGraph/data/t2i-10M/query.10k.fbin
TEST_GT_PATH=/workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin

# Result directory
RESULT_DIR=/workspace/OOD-ANNS/NGFix/data/t2i-10M/selective_optimization_results
mkdir -p $RESULT_DIR

echo "=== NGFix Selective Optimization Test ==="
echo "Base Index: $BASE_INDEX_PATH"
echo "Full NGFix Index (Control): $FULL_NGFIX_INDEX_PATH"
echo "Train Queries: $TRAIN_QUERY_PATH"
echo "Train GT: $TRAIN_GT_PATH"
echo "Test Queries: $TEST_QUERY_PATH"
echo "Test GT: $TEST_GT_PATH"
echo "Result Directory: $RESULT_DIR"
echo ""

# Run the test
taskset -c 1 test/test_ngfix_selective_optimization \
--base_index_path $BASE_INDEX_PATH \
--train_query_path $TRAIN_QUERY_PATH \
--train_gt_path $TRAIN_GT_PATH \
--test_query_path $TEST_QUERY_PATH \
--test_gt_path $TEST_GT_PATH \
--metric ip_float \
--full_ngfix_index_path $FULL_NGFIX_INDEX_PATH \
--result_dir $RESULT_DIR \
--K $K \
--num_test_queries $NUM_TEST_QUERIES

echo ""
echo "Done! Results saved to $RESULT_DIR/ngfix_selective_optimization_results.csv"

