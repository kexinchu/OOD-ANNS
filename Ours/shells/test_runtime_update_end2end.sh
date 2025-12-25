#!/bin/bash

# End-to-end test: QPS=400 search + QPS=100 insert with connectivity enhancement

set -e

# Default paths (adjust as needed)
# BASE_INDEX_PATH="${BASE_INDEX_PATH:-/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M16_efC500_MEX48_AKNN1500_10M.index}"
BASE_INDEX_PATH="${BASE_INDEX_PATH:-/workspace/OOD-ANNS/Ours/data/comparison_10M/partial_rebuild.index}"
TRAIN_QUERY_PATH="${TRAIN_QUERY_PATH:-/workspace/OOD-ANNS/Ours/data/comparison_10M/query.test.8M.fbin}"
TRAIN_GT_PATH="${TRAIN_GT_PATH:-/workspace/OOD-ANNS/Ours/data/comparison_10M/train.gt.8M.bin}"
ADDITIONAL_VECTOR_PATH="${ADDITIONAL_VECTOR_PATH:-/workspace/RoarGraph/data/t2i-10M/base.additional.10M.fbin}"
RESULT_DIR="${RESULT_DIR:-./data/runtime_update_test}"
RESULT_FILE="${RESULT_FILE:-runtime_update_results.csv}"  # Result filename (without path)
METRIC="${METRIC:-ip_float}"
K="${K:-100}"
DURATION_MINUTES="${DURATION_MINUTES:-300}"  # 5 hours = 300 minutes

# Create result directory
mkdir -p "$RESULT_DIR"

cd /workspace/OOD-ANNS/Ours/build

echo "=== Building Runtime Update End-to-End Test ==="
make test_runtime_update_end2end -j$(nproc)

echo ""
echo "=== Configuration ==="
echo "Base index: $BASE_INDEX_PATH"
echo "Train queries: $TRAIN_QUERY_PATH"
echo "Train GT: $TRAIN_GT_PATH"
echo "Additional vectors: $ADDITIONAL_VECTOR_PATH"
echo "Result dir: $RESULT_DIR"
echo "K: $K"
echo "ef_search: 1000"
echo "Search QPS: 1000"
echo "Insert QPS: 126"
echo "Delete QPS: 14 (9:1 ratio with insert)"
echo "Duration: $DURATION_MINUTES minutes"
echo ""

echo "=== Running Runtime Update End-to-End Test ==="
cd /workspace/OOD-ANNS/Ours/build

# Handle absolute vs relative paths
if [[ "$BASE_INDEX_PATH" = /* ]]; then
    # Absolute path, use as is
    INDEX_PATH="$BASE_INDEX_PATH"
else
    # Relative path, prepend ../
    INDEX_PATH="../$BASE_INDEX_PATH"
fi

nohup ./test/test_runtime_update_end2end \
    --base_index_path "$INDEX_PATH" \
    --train_query_path "$TRAIN_QUERY_PATH" \
    --train_gt_path "$TRAIN_GT_PATH" \
    --additional_vector_path "$ADDITIONAL_VECTOR_PATH" \
    --metric "$METRIC" \
    --result_dir "../$RESULT_DIR" \
    --K "$K" \
    --duration_minutes "$DURATION_MINUTES" \
    > "../$RESULT_DIR/nohup_new.out" 2>&1 &

TEST_PID=$!
echo "Test started with PID: $TEST_PID"
echo "Logs are being written to: $RESULT_DIR/nohup_new.out"
echo "Results will be saved to: $RESULT_DIR/runtime_update_results_q1000_i140.csv"
echo ""
echo "To check progress: tail -f $RESULT_DIR/nohup_new.out"
echo "To stop the test: kill $TEST_PID"

