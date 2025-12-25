#!/bin/bash

# Test script for comparing fix node cost between NGFix and Group-EH methods (nohup version)

BASE_INDEX_PATH=""
INSERT_DATA_PATH=""
OUTPUT_DIR="/workspace/OOD-ANNS/Ours/data/fix_node_cost_test"
METRIC="ip_float"
NUM_INSERT_OPS=10000
K=100
EFC=200

# Try to find 8M base index (for insert test)
if [ -f "/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M16_efC500_MEX48_8M.index" ]; then
    BASE_INDEX_PATH="/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M16_efC500_MEX48_8M.index"
elif [ -f "/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M16_efC500_MEX48_AKNN1500_8M.index" ]; then
    BASE_INDEX_PATH="/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M16_efC500_MEX48_AKNN1500_8M.index"
elif [ -f "/workspace/OOD-ANNS/Ours/data/comparison_10M/base.index" ]; then
    BASE_INDEX_PATH="/workspace/OOD-ANNS/Ours/data/comparison_10M/base.index"
fi

# Try to find insert data
if [ -f "/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.additional.2M.fbin" ]; then
    INSERT_DATA_PATH="/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.additional.2M.fbin"
elif [ -f "/workspace/RoarGraph/data/t2i-10M/base.10M.fbin" ]; then
    INSERT_DATA_PATH="/workspace/RoarGraph/data/t2i-10M/base.10M.fbin"
elif [ -f "/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.fbin" ]; then
    INSERT_DATA_PATH="/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.fbin"
fi

echo "=== Insert Node Cost Comparison Test (nohup) ==="
echo "Base index: $BASE_INDEX_PATH"
echo "Insert data: $INSERT_DATA_PATH"
echo "Number of insert operations: $NUM_INSERT_OPS"
echo "K: $K"
echo "efC: $EFC"
echo "Output dir: $OUTPUT_DIR"
echo "Metric: $METRIC"
echo ""

if [ -z "$BASE_INDEX_PATH" ] || [ ! -f "$BASE_INDEX_PATH" ]; then
    echo "Error: Base index file not found"
    exit 1
fi

if [ -z "$INSERT_DATA_PATH" ] || [ ! -f "$INSERT_DATA_PATH" ]; then
    echo "Error: Insert data file not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run test in background with nohup
cd /workspace/OOD-ANNS/Ours/build

nohup ./test/test_fix_node_cost \
    --base_index_path "$BASE_INDEX_PATH" \
    --insert_data_path "$INSERT_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --metric "$METRIC" \
    --num_insert_ops $NUM_INSERT_OPS \
    --k $K \
    --efC $EFC > "$OUTPUT_DIR/test_fix_node_cost_nohup.log" 2>&1 &

echo "Test started in background. PID: $!"
echo "Log file: $OUTPUT_DIR/test_fix_node_cost_nohup.log"
echo "Results will be saved to: $OUTPUT_DIR/insert_node_results.csv"
echo ""
echo "To check progress: tail -f $OUTPUT_DIR/test_fix_node_cost_nohup.log"

