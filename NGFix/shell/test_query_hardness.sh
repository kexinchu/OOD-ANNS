#!/bin/bash

MEX=48
M=16
efC=500
K=100
NUM_QUERIES=1000

# Use base graph (not NGFix processed)
INDEX_PATH=/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index
QUERY_PATH=/workspace/RoarGraph/data/t2i-10M/query.10k.fbin
GT_PATH=/workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin
RESULT_PATH=/workspace/OOD-ANNS/NGFix/data/t2i-10M/query_hardness_results.json

echo "Testing query hardness detection..."
echo "Index: $INDEX_PATH"
echo "Queries: $QUERY_PATH"
echo "Output: $RESULT_PATH"

taskset -c 1 test/test_query_hardness \
--test_query_path $QUERY_PATH \
--test_gt_path $GT_PATH \
--metric ip_float \
--K $K \
--num_queries $NUM_QUERIES \
--index_path $INDEX_PATH \
--result_path $RESULT_PATH

echo "Done! Results saved to $RESULT_PATH"

