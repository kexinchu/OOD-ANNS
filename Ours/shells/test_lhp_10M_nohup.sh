#!/bin/bash

# Test script for LHP comparison with 10M base data and 10M train queries
# Run with nohup to ensure it completes

BASE_DATA_SIZE=10000000  # 10M
QUERY_SIZE=10000000      # 10M
OUTPUT_DIR="/workspace/OOD-ANNS/Ours/data/comparison_10M"
METRIC="ip_float"
LOG_FILE="/workspace/OOD-ANNS/Ours/data/comparison_10M/test_log.txt"

# Try to find data files
BASE_DATA_PATH="/workspace/RoarGraph/data/t2i-10M/base.10M.fbin"
QUERY_DATA_PATH="/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin"

echo "=== LHP Comparison Test (10M Base + 10M Query) ==="
echo "Base data: $BASE_DATA_PATH"
echo "Query data: $QUERY_DATA_PATH"
echo "Base data size: $BASE_DATA_SIZE"
echo "Query size: $QUERY_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

if [ ! -f "$BASE_DATA_PATH" ]; then
    echo "Error: Base data file not found: $BASE_DATA_PATH"
    exit 1
fi

if [ ! -f "$QUERY_DATA_PATH" ]; then
    echo "Error: Query data file not found: $QUERY_DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run test with nohup
cd /workspace/OOD-ANNS/Ours/build

echo "Starting test at $(date)" | tee -a "$LOG_FILE"
echo "Command: ./test/test_lhp_comparison --base_data_path $BASE_DATA_PATH --query_data_path $QUERY_DATA_PATH --output_dir $OUTPUT_DIR --metric $METRIC --base_data_size $BASE_DATA_SIZE --query_size $QUERY_SIZE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

./test/test_lhp_comparison \
    --base_data_path "$BASE_DATA_PATH" \
    --query_data_path "$QUERY_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --metric "$METRIC" \
    --base_data_size $BASE_DATA_SIZE \
    --query_size $QUERY_SIZE \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "=== Test completed at $(date) with exit code $EXIT_CODE ===" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"

exit $EXIT_CODE


