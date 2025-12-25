#!/bin/bash

MEX=48
M=16
efC=500
K=100
NUM_TRAIN_QUERIES=10000000  # 10M
NUM_TEST_QUERIES=1000

# Use base graph (not NGFix processed)
INDEX_PATH=/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index
TRAIN_QUERY_PATH=/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin
TRAIN_GT_PATH=/workspace/RoarGraph/data/t2i-10M/train.gt.bin
TEST_QUERY_PATH=/workspace/RoarGraph/data/t2i-10M/query.10k.fbin
TEST_GT_PATH=/workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin
FEATURE_CACHE_PATH=/workspace/OOD-ANNS/Ours/data/t2i-10M/hardness_features_cache.bin
RESULT_PATH=/workspace/OOD-ANNS/Ours/data/t2i-10M/hardness_predictor_tree_results.json
BEST_FEATURES_JSON=/workspace/OOD-ANNS/NGFix/data/t2i-10M/hardness_predictor_results.json  # From linear regression

# Create data directory if it doesn't exist
mkdir -p /workspace/OOD-ANNS/Ours/data/t2i-10M

echo "=== Hardness Predictor Tree Testing (Ours) ==="
echo "Index: $INDEX_PATH"
echo "Train Queries: $TRAIN_QUERY_PATH"
echo "Train GT: $TRAIN_GT_PATH"
echo "Test Queries: $TEST_QUERY_PATH"
echo "Test GT: $TEST_GT_PATH"
echo "Feature Cache: $FEATURE_CACHE_PATH"
echo "Output: $RESULT_PATH"
echo ""

# Check if feature cache exists, if not, collect features
if [ ! -f "$FEATURE_CACHE_PATH" ]; then
    echo "Step 1: Collecting features from ${NUM_TRAIN_QUERIES} queries..."
    cd /workspace/OOD-ANNS/Ours/build
    taskset -c 1 test/test_hardness_predictor_tree \
    --train_query_path $TRAIN_QUERY_PATH \
    --train_gt_path $TRAIN_GT_PATH \
    --test_query_path $TEST_QUERY_PATH \
    --test_gt_path $TEST_GT_PATH \
    --metric ip_float \
    --K $K \
    --num_queries $NUM_TRAIN_QUERIES \
    --index_path $INDEX_PATH \
    --result_path $RESULT_PATH \
    --feature_cache_path $FEATURE_CACHE_PATH \
    --mode collect
    
    if [ $? -ne 0 ]; then
        echo "Error: Feature collection failed!"
        exit 1
    fi
    echo ""
else
    echo "Feature cache exists, skipping collection step..."
    echo ""
fi

echo "Step 2: Training and testing with 100 trees on 1000 queries..."

# Step 2: Train and test (output format like hardness_predictor_results.json)
cd /workspace/OOD-ANNS/Ours/build
taskset -c 1 test/test_hardness_predictor_tree \
--train_query_path $TRAIN_QUERY_PATH \
--train_gt_path $TRAIN_GT_PATH \
--test_query_path $TEST_QUERY_PATH \
--test_gt_path $TEST_GT_PATH \
--metric ip_float \
--K $K \
--num_queries $NUM_TRAIN_QUERIES \
--index_path $INDEX_PATH \
--result_path $RESULT_PATH \
--feature_cache_path $FEATURE_CACHE_PATH \
--mode train_test \
--num_trees 100 \
--best_features_json $BEST_FEATURES_JSON

if [ $? -ne 0 ]; then
    echo "Error: Testing failed!"
    exit 1
fi

echo ""
echo "Step 3: Generating visualization..."
cd /workspace/OOD-ANNS/Ours
python3 scripts/visualize_hardness_predictor.py $RESULT_PATH /workspace/OOD-ANNS/Ours/pictures/hardness_predictor_tree_scatter.png

if [ $? -ne 0 ]; then
    echo "Warning: Visualization failed, but results are saved"
fi

echo ""
echo "Done! Results saved to:"
echo "  Feature cache: $FEATURE_CACHE_PATH"
echo "  Results: $RESULT_PATH"
echo "  Visualization: /workspace/OOD-ANNS/Ours/pictures/hardness_predictor_tree_scatter.png"

