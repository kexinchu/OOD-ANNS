#!/bin/bash
# Complete training script for 10M data with optimized parameters

cd /workspace/OOD-ANNS/Ours

echo "=========================================="
echo "10M Training Data - LightGBM Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Training samples: 10,000,000"
echo "  - Trees: 300 (increased for more learning)"
echo "  - Depth: 20"
echo "  - Learning rate: 0.05 (reduced for better convergence)"
echo "  - Early stopping: 200 rounds patience"
echo ""

# Wait for data collection to complete
echo "Step 1: Checking if data collection is complete..."
while [ ! -f "data/t2i-10M/hardness_features_cache_lgbm_10M.bin" ]; do
    echo "⏳ Waiting for data collection... ($(date +%H:%M:%S))"
    sleep 60
done

size=$(ls -lh data/t2i-10M/hardness_features_cache_lgbm_10M.bin | awk '{print $5}')
echo "✅ Data collection completed! File size: $size"
echo ""

# Train model
echo "Step 2: Training LightGBM model..."
python3 scripts/train_lightgbm_hardness_predictor.py \
    data/t2i-10M/hardness_features_cache_lgbm_10M.bin \
    data/t2i-10M/hardness_predictor_tree_results_lgbm_10M.json.lgbm.model \
    300 20 0.05 2>&1 | tee data/t2i-10M/training_10M.log

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "Step 3: Extracting test features and predicting..."
cd build

# Extract test features
taskset -c 1 ./test/test_hardness_predictor_tree \
    --train_query_path /workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin \
    --train_gt_path /workspace/RoarGraph/data/t2i-10M/train.gt.bin \
    --test_query_path /workspace/RoarGraph/data/t2i-10M/query.10k.fbin \
    --test_gt_path /workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin \
    --metric ip_float \
    --K 100 \
    --num_queries 10000000 \
    --index_path /workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M16_efC500_MEX48.index \
    --result_path /workspace/OOD-ANNS/Ours/data/t2i-10M/hardness_predictor_tree_results_lgbm_10M.json \
    --feature_cache_path /workspace/OOD-ANNS/Ours/data/t2i-10M/hardness_features_cache_lgbm_10M.bin.test \
    --mode collect 2>&1 | grep -E "(Processing|Collection|Saved)" | tail -5

# Predict
python3 ../scripts/predict_lightgbm_hardness.py \
    ../data/t2i-10M/hardness_predictor_tree_results_lgbm_10M.json.lgbm.model \
    ../data/t2i-10M/hardness_features_cache_lgbm_10M.bin.test \
    ../data/t2i-10M/hardness_predictor_tree_results_lgbm_10M.json

echo ""
echo "=========================================="
echo "Training and Testing Complete!"
echo "=========================================="
echo "Results saved to: data/t2i-10M/hardness_predictor_tree_results_lgbm_10M.json"

