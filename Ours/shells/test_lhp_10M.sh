#!/bin/bash

# Test script for LHP comparison with 10M base data and 10M train queries

BASE_DATA_SIZE=10000000  # 10M
QUERY_SIZE=10000000      # 10M
OUTPUT_DIR="/workspace/OOD-ANNS/Ours/data/comparison_10M"
METRIC="ip_float"

# Try to find data files
BASE_DATA_PATH=""
QUERY_DATA_PATH=""

# Try base data
if [ -f "/workspace/RoarGraph/data/t2i-10M/base.10M.fbin" ]; then
    BASE_DATA_PATH="/workspace/RoarGraph/data/t2i-10M/base.10M.fbin"
elif [ -f "/workspace/RoarGraph/data/t2i-10M/base.fbin" ]; then
    BASE_DATA_PATH="/workspace/RoarGraph/data/t2i-10M/base.fbin"
elif [ -f "/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.fbin" ]; then
    BASE_DATA_PATH="/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.fbin"
fi

# Try query data
if [ -f "/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin" ]; then
    QUERY_DATA_PATH="/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin"
elif [ -f "/workspace/RoarGraph/data/t2i-10M/query.train.fbin" ]; then
    QUERY_DATA_PATH="/workspace/RoarGraph/data/t2i-10M/query.train.fbin"
fi

echo "=== LHP Comparison Test (10M Base + 10M Query) ==="
echo "Base data: $BASE_DATA_PATH"
echo "Query data: $QUERY_DATA_PATH"
echo "Base data size: $BASE_DATA_SIZE"
echo "Query size: $QUERY_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo ""

if [ -z "$BASE_DATA_PATH" ] || [ ! -f "$BASE_DATA_PATH" ]; then
    echo "Error: Base data file not found"
    exit 1
fi

if [ -z "$QUERY_DATA_PATH" ] || [ ! -f "$QUERY_DATA_PATH" ]; then
    echo "Error: Query data file not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run test
cd /workspace/OOD-ANNS/Ours/build

./test/test_lhp_comparison \
    --base_data_path "$BASE_DATA_PATH" \
    --query_data_path "$QUERY_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --metric "$METRIC" \
    --base_data_size $BASE_DATA_SIZE \
    --query_size $QUERY_SIZE

echo ""
echo "=== Test completed ==="
echo "Results saved to: $OUTPUT_DIR"

