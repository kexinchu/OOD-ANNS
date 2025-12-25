#!/bin/bash

# Test script for lazy delete mechanism

# Default parameters
BASE_INDEX_PATH=""
QUERY_DATA_PATH=""
OUTPUT_DIR="/workspace/OOD-ANNS/Ours/data/lazy_delete_test"
METRIC="ip_float"
NUM_QUERIES=1000
K=100
EF_SEARCH=200

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base_index_path)
            BASE_INDEX_PATH="$2"
            shift 2
            ;;
        --query_data_path)
            QUERY_DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --num_queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --ef_search)
            EF_SEARCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build directory
BUILD_DIR="/workspace/OOD-ANNS/Ours/build"
TEST_BINARY="$BUILD_DIR/test/test_lazy_delete"

# Check if binary exists
if [ ! -f "$TEST_BINARY" ]; then
    echo "Error: Test binary not found. Building..."
    cd "$BUILD_DIR" || exit 1
    make test_lazy_delete || {
        echo "Build failed. Please check the build directory."
        exit 1
    }
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="$TEST_BINARY"

if [ -n "$BASE_INDEX_PATH" ]; then
    CMD="$CMD --base_index_path $BASE_INDEX_PATH"
fi

if [ -n "$QUERY_DATA_PATH" ]; then
    CMD="$CMD --query_data_path $QUERY_DATA_PATH"
fi

CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --metric $METRIC"
CMD="$CMD --num_queries $NUM_QUERIES"
CMD="$CMD --k $K"
CMD="$CMD --ef_search $EF_SEARCH"

echo "=== Running Lazy Delete Test ==="
echo "Command: $CMD"
echo ""

# Run test
$CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Test completed successfully ==="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "No CSV files found"
else
    echo ""
    echo "=== Test failed with exit code $EXIT_CODE ==="
    exit $EXIT_CODE
fi

