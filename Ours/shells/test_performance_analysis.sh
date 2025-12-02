#!/bin/bash

# Performance analysis test script for Ours implementation
# Usage: ./test_performance_analysis.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OURS_DIR="$SCRIPT_DIR/.."
NGFIX_DIR="$OURS_DIR/../NGFix"
DATA_DIR="$NGFIX_DIR/data/t2i-10M"
ROARGRAPH_DATA_DIR="/workspace/RoarGraph/data/t2i-10M"

echo "=== Building Ours ==="
cd "$OURS_DIR"
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo ""
echo "=== Checking Data Files ==="
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please ensure t2i-10M data is available"
    exit 1
fi

# Check for required files (use available index file)
INDEX_FILE=""
if [ -f "$DATA_DIR/base.index" ]; then
    INDEX_FILE="$DATA_DIR/base.index"
elif [ -f "$DATA_DIR/t2i_10M_HNSWBottom_M16_efC500_MEX48.index" ]; then
    INDEX_FILE="$DATA_DIR/t2i_10M_HNSWBottom_M16_efC500_MEX48.index"
elif [ -f "$DATA_DIR/t2i_10M_HNSWBottom_M16_efC500_MEX48_8M.index" ]; then
    INDEX_FILE="$DATA_DIR/t2i_10M_HNSWBottom_M16_efC500_MEX48_8M.index"
else
    echo "Error: No base index file found in $DATA_DIR"
    exit 1
fi

# Check for query and GT files
TRAIN_QUERY_FILE=""
TRAIN_GT_FILE=""

if [ -f "$ROARGRAPH_DATA_DIR/query.train.10M.fbin" ]; then
    TRAIN_QUERY_FILE="$ROARGRAPH_DATA_DIR/query.train.10M.fbin"
elif [ -f "$DATA_DIR/train_query.fbin" ]; then
    TRAIN_QUERY_FILE="$DATA_DIR/train_query.fbin"
else
    echo "Error: Train query file not found"
    exit 1
fi

if [ -f "$ROARGRAPH_DATA_DIR/train.gt.bin" ]; then
    TRAIN_GT_FILE="$ROARGRAPH_DATA_DIR/train.gt.bin"
elif [ -f "$DATA_DIR/train_gt.ibin" ]; then
    TRAIN_GT_FILE="$DATA_DIR/train_gt.ibin"
else
    echo "Error: Train GT file not found"
    exit 1
fi

echo "Using files:"
echo "  Index: $INDEX_FILE"
echo "  Train Query: $TRAIN_QUERY_FILE"
echo "  Train GT: $TRAIN_GT_FILE"
echo ""

echo "=== Running Performance Analysis ==="
RESULT_DIR="$OURS_DIR/results"
mkdir -p "$RESULT_DIR"

# Determine metric type (default to ip_float for text2image)
METRIC="ip_float"

# Run performance analysis
./test/test_performance_analysis \
    --base_index_path "$INDEX_FILE" \
    --train_query_path "$TRAIN_QUERY_FILE" \
    --train_gt_path "$TRAIN_GT_FILE" \
    --metric "$METRIC" \
    --result_dir "$RESULT_DIR" \
    --K 100 \
    --num_queries 1000

echo ""
echo "=== Performance Analysis Completed ==="
echo "Results saved to: $RESULT_DIR"
echo "Check $RESULT_DIR/performance_breakdown.csv for detailed results"

