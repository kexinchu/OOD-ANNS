#!/usr/bin/env python3
"""
Predict hardness using trained LightGBM model.
"""

import numpy as np
import lightgbm as lgb
import json
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: predict_lightgbm_hardness.py <model_path> <feature_cache_path> <output_json_path> [max_samples]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    feature_cache_path = sys.argv[2]
    output_json_path = sys.argv[3]
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = lgb.Booster(model_file=model_path)
    
    # Load test features
    print(f"Loading test features from {feature_cache_path}...")
    with open(feature_cache_path, 'rb') as f:
        n_samples = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        n_features = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        
        # Read features
        features = np.frombuffer(f.read(n_samples * n_features * 4), dtype=np.float32)
        features = features.reshape(n_samples, n_features)
        
        # Read targets (actual hardness)
        targets = np.frombuffer(f.read(n_samples * 4), dtype=np.float32)
        
        # Read recalls
        recalls = np.frombuffer(f.read(n_samples * 4), dtype=np.float32)
    
    # Limit samples if requested
    if max_samples is not None and max_samples < len(features):
        features = features[:max_samples]
        targets = targets[:max_samples]
        recalls = recalls[:max_samples]
        print(f"  Limited to {max_samples} samples for testing")
    
    print(f"  Testing on {len(features)} samples")
    
    # Predict
    print("Predicting...")
    predictions = model.predict(features, num_iteration=model.best_iteration)
    
    # Clamp to [0, 1]
    predictions = np.clip(predictions, 0.0, 1.0)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    correlation = np.corrcoef(predictions, targets)[0, 1]
    recall_correlation = np.corrcoef(predictions, recalls)[0, 1]
    
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Correlation (pred vs hardness): {correlation:.4f}")
    print(f"  Correlation (pred vs recall): {recall_correlation:.4f}")
    
    # Prepare output
    queries = []
    for i in range(len(predictions)):
        queries.append({
            'recall': float(recalls[i]),
            'hardness': float(targets[i]),
            'predicted_hardness': float(predictions[i])
        })
    
    output = {
        'queries': queries,
        'best_mse': float(mse),
        'best_mae': float(mae),
        'correlation': float(correlation),
        'recall_correlation': float(recall_correlation),
        'num_trees': model.num_trees(),
        'best_feature_combination': []  # LightGBM doesn't have feature selection
    }
    
    # Save results
    with open(output_json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {output_json_path}")

if __name__ == '__main__':
    main()

