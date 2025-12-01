#!/bin/bash

MEX=48
M=16
efC=500
K=100
NUM_QUERIES=1000

# Use base graph (not NGFix processed)
INDEX_PATH=/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index
TRAIN_QUERY_PATH=/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin
TRAIN_GT_PATH=/workspace/RoarGraph/data/t2i-10M/train.gt.bin
TEST_QUERY_PATH=/workspace/RoarGraph/data/t2i-10M/query.10k.fbin
TEST_GT_PATH=/workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin
RESULT_PATH=/workspace/OOD-ANNS/NGFix/data/t2i-10M/hardness_predictor_results.json

echo "Testing Hardness Predictor..."
echo "Index: $INDEX_PATH"
echo "Train Queries: $TRAIN_QUERY_PATH"
echo "Train GT: $TRAIN_GT_PATH"
echo "Test Queries: $TEST_QUERY_PATH"
echo "Test GT: $TEST_GT_PATH"
echo "Output: $RESULT_PATH"

taskset -c 1 test/test_hardness_predictor \
--train_query_path $TRAIN_QUERY_PATH \
--train_gt_path $TRAIN_GT_PATH \
--test_query_path $TEST_QUERY_PATH \
--test_gt_path $TEST_GT_PATH \
--metric ip_float \
--K $K \
--num_queries $NUM_QUERIES \
--index_path $INDEX_PATH \
--result_path $RESULT_PATH

echo "Done! Results saved to $RESULT_PATH"

