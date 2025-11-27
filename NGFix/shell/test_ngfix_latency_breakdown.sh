#!/bin/bash

# Test script for NGFix latency breakdown
# This script tests the latency comparison between EH matrix calculation and other logic

MEX=48
M=16
efC=500

# Base graph path (10M data)
BASE_GRAPH_PATH=/workspace/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index

# Train GT path (contains ground truth for graph enhancement)
TRAIN_GT_PATH=/workspace/RoarGraph/data/t2i-10M/train.gt.bin

# Output path for results
OUTPUT_PATH=/workspace/NGFix/data/t2i-10M/ngfix_latency_breakdown_results.json

# Number of queries to test (1k by default)
NUM_QUERIES=1000

# Metric
METRIC=ip_float

echo "=========================================="
echo "NGFix Latency Breakdown Test"
echo "=========================================="
echo "Base graph: ${BASE_GRAPH_PATH}"
echo "Train GT: ${TRAIN_GT_PATH}"
echo "Number of queries: ${NUM_QUERIES}"
echo "Metric: ${METRIC}"
echo "Output: ${OUTPUT_PATH}"
echo "=========================================="
echo ""

# Check if base graph exists
if [ ! -f "${BASE_GRAPH_PATH}" ]; then
    echo "Error: Base graph not found at ${BASE_GRAPH_PATH}"
    echo "Please build the base graph first using build_hnsw_bottom"
    exit 1
fi

# Check if train GT exists
if [ ! -f "${TRAIN_GT_PATH}" ]; then
    echo "Error: Train GT file not found at ${TRAIN_GT_PATH}"
    exit 1
fi

# Run the test
./build/test/test_ngfix_latency_breakdown \
    --base_graph_path ${BASE_GRAPH_PATH} \
    --train_gt_path ${TRAIN_GT_PATH} \
    --metric ${METRIC} \
    --num_queries ${NUM_QUERIES} \
    --output_path ${OUTPUT_PATH}

echo ""
echo "Test completed. Results saved to: ${OUTPUT_PATH}"

