#!/bin/bash

# End-to-end test: QPS=1000 search + QPS=20 insert + QPS=2 delete (10:1 ratio) with NGFix baseline (no graph fixing)

set -e

# Default paths (adjust as needed)
BASE_INDEX_PATH="${BASE_INDEX_PATH:-/workspace/OOD-ANNS/Ours/data/comparison_10M/partial_rebuild.index}"
TRAIN_QUERY_PATH="${TRAIN_QUERY_PATH:-/workspace/OOD-ANNS/Ours/data/comparison_10M/query.test.8M.fbin}"
TRAIN_GT_PATH="${TRAIN_GT_PATH:-/workspace/OOD-ANNS/Ours/data/comparison_10M/train.gt.8M.bin}"
ADDITIONAL_VECTOR_PATH="${ADDITIONAL_VECTOR_PATH:-/workspace/RoarGraph/data/t2i-10M/base.additional.10M.fbin}"
RESULT_DIR="${RESULT_DIR:-./data/runtime_update_test}"
METRIC="${METRIC:-ip_float}"
K="${K:-100}"
DURATION_MINUTES="${DURATION_MINUTES:-300}"

# Create result directory
mkdir -p "$RESULT_DIR"

cd /workspace/OOD-ANNS/Ours/build

echo "=== Building Runtime Update End-to-End Test (NGFix Baseline) ==="
make test_runtime_update_end2end_ngfix -j$(nproc)

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
echo "Duration: $DURATION_MINUTES minutes ($(($DURATION_MINUTES / 60)) hours)"
echo "Method: NGFix baseline (NO graph fixing)"
echo "Note: Delete operations use nodes from base.additional.10M.fbin (same as insert)"
echo ""

# Handle absolute vs relative paths
if [[ "$BASE_INDEX_PATH" = /* ]]; then
    # Absolute path, use as is
    INDEX_PATH="$BASE_INDEX_PATH"
else
    # Relative path, prepend ../
    INDEX_PATH="../$BASE_INDEX_PATH"
fi

echo "=== Running Runtime Update End-to-End Test (NGFix Baseline) ==="
cd /workspace/OOD-ANNS/Ours/build
nohup ./test/test_runtime_update_end2end_ngfix \
    --base_index_path "$INDEX_PATH" \
    --train_query_path "$TRAIN_QUERY_PATH" \
    --train_gt_path "$TRAIN_GT_PATH" \
    --additional_vector_path "$ADDITIONAL_VECTOR_PATH" \
    --metric "$METRIC" \
    --result_dir "../$RESULT_DIR" \
    --K "$K" \
    --duration_minutes "$DURATION_MINUTES" \
    --search_qps 1000 \
    --insert_qps 140 \
    > "../$RESULT_DIR/nohup_ngfix.out" 2>&1 &

TEST_PID=$!
echo "Test started with PID: $TEST_PID"
echo "Logs are being written to: $RESULT_DIR/nohup_ngfix.out"
echo "Results will be saved to: $RESULT_DIR/runtime_update_results_ngfix.csv"
echo ""
echo "To check progress: tail -f $RESULT_DIR/nohup_ngfix.out"
echo "To stop the test: kill $TEST_PID"
