#!/bin/bash

# Test script for runtime insert with hard query detection

BASE_INDEX_PATH="/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M16_efC500_MEX48.index"
TRAIN_GT_PATH=""
TRAIN_QUERY_PATH=""
OUTPUT_DIR="/workspace/OOD-ANNS/Ours/data/t2i-10M"
METRIC="ip_float"
K=100
EF_SEARCH=100
RECALL_THRESHOLD=0.92
MAX_QUERIES=0  # 0 means process all

# Try to find train.gt.bin
if [ -f "/workspace/RoarGraph/data/t2i-10M/train.gt.bin" ]; then
    TRAIN_GT_PATH="/workspace/RoarGraph/data/t2i-10M/train.gt.bin"
elif [ -f "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train.gt.bin" ]; then
    TRAIN_GT_PATH="/workspace/OOD-ANNS/NGFix/data/t2i-10M/train.gt.bin"
fi

# Try to find train_query.fbin
if [ -f "/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin" ]; then
    TRAIN_QUERY_PATH="/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin"
elif [ -f "/workspace/RoarGraph/data/t2i-10M/train_query.fbin" ]; then
    TRAIN_QUERY_PATH="/workspace/RoarGraph/data/t2i-10M/train_query.fbin"
elif [ -f "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train_query.fbin" ]; then
    TRAIN_QUERY_PATH="/workspace/OOD-ANNS/NGFix/data/t2i-10M/train_query.fbin"
fi

echo "=== Runtime Insert Test Configuration ==="
echo "Base index: $BASE_INDEX_PATH"
echo "Train GT: $TRAIN_GT_PATH"
echo "Train Query: $TRAIN_QUERY_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Metric: $METRIC"
echo "K: $K"
echo "efSearch: $EF_SEARCH"
echo "Recall threshold: $RECALL_THRESHOLD"
echo ""

# Check if base index exists
if [ ! -f "$BASE_INDEX_PATH" ]; then
    echo "Error: Base index not found: $BASE_INDEX_PATH"
    exit 1
fi

# Check if train GT exists
if [ -z "$TRAIN_GT_PATH" ] || [ ! -f "$TRAIN_GT_PATH" ]; then
    echo "Error: Train GT file not found"
    echo "Please provide --train_gt_path argument or ensure file exists"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="./test/test_runtime_insert"
CMD="$CMD --base_index_path $BASE_INDEX_PATH"
CMD="$CMD --train_gt_path $TRAIN_GT_PATH"
CMD="$CMD --output_index_dir $OUTPUT_DIR"
CMD="$CMD --metric $METRIC"
CMD="$CMD --K $K"
CMD="$CMD --efSearch $EF_SEARCH"
CMD="$CMD --recall_threshold $RECALL_THRESHOLD"
if [ "$MAX_QUERIES" -gt 0 ]; then
    CMD="$CMD --max_queries $MAX_QUERIES"
fi

if [ -n "$TRAIN_QUERY_PATH" ] && [ -f "$TRAIN_QUERY_PATH" ]; then
    echo "Note: Train query file found, will test recall"
else
    echo "Warning: Train query file not found, will proceed without recall testing"
fi

echo ""
echo "Running test..."
echo ""

cd /workspace/OOD-ANNS/Ours/build
$CMD

echo ""
echo "=== Test completed ==="

