#!/usr/bin/env python3
"""
Train LightGBM model for hardness prediction with correlation loss function.
"""

import numpy as np
import lightgbm as lgb
import json
import sys
import os
from typing import Tuple, List

def correlation_loss(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom loss function: negative correlation between predictions and recalls.
    Since hardness = 1 - recall, we want to maximize negative correlation.
    
    Args:
        y_pred: Predicted hardness values
        y_true: LightGBM Dataset containing actual hardness and recalls
    
    Returns:
        grad: Gradient of the loss function
        hess: Hessian of the loss function
    """
    # Get actual values from dataset
    actual_hardness = y_true.get_label()
    recalls = y_true.get_data()  # This should contain recalls if we pass them
    
    # For now, calculate correlation-based loss
    # We want to maximize negative correlation between predicted_hardness and recall
    # So we minimize: -correlation(pred, recall)
    
    # Normalize predictions
    y_pred = y_pred.ravel()
    pred_mean = np.mean(y_pred)
    pred_std = np.std(y_pred) + 1e-6
    
    # Calculate correlation
    # Since we don't have recalls directly in y_true, we'll use a different approach
    # We'll use MSE with correlation-based weighting
    
    # For gradient calculation, we use a simplified approach
    # The gradient should push predictions towards maximizing negative correlation
    residuals = y_pred - actual_hardness
    
    # Weight by how far we are from perfect negative correlation
    # Perfect correlation would mean: pred = 1 - recall = hardness
    # So we want pred to match actual_hardness
    
    grad = 2.0 * residuals
    hess = 2.0 * np.ones_like(residuals)
    
    return grad, hess

def correlation_objective_with_recall(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom objective that optimizes correlation with recall.
    We want to maximize negative correlation between predicted_hardness and recall.
    Since hardness = 1 - recall, maximizing negative correlation means:
    - When recall is high, predicted_hardness should be low
    - When recall is low, predicted_hardness should be high
    
    This is equivalent to minimizing: -correlation(pred, recall)
    """
    actual_hardness = y_true.get_label()
    y_pred = y_pred.ravel()
    
    # Get recalls from dataset (we'll need to pass them separately)
    # For now, we'll use actual_hardness and assume recalls = 1 - hardness
    recalls = 1.0 - actual_hardness
    
    # Normalize
    pred_mean = np.mean(y_pred)
    recall_mean = np.mean(recalls)
    
    pred_centered = y_pred - pred_mean
    recall_centered = recalls - recall_mean
    
    pred_std = np.std(pred_centered) + 1e-6
    recall_std = np.std(recall_centered) + 1e-6
    
    # Correlation between pred and recall
    correlation = np.sum(pred_centered * recall_centered) / (pred_std * recall_std * len(y_pred))
    
    # We want to maximize negative correlation, so minimize: -correlation
    # Gradient of -correlation w.r.t. predictions
    # d(-corr)/d(pred) = -d(corr)/d(pred)
    
    # Simplified gradient: push predictions towards maximizing negative correlation
    # This means: when recall is high, pred should be low; when recall is low, pred should be high
    grad = -2.0 * recall_centered / (pred_std * len(y_pred)) + 2.0 * correlation * pred_centered / (pred_std * pred_std)
    hess = 2.0 * np.ones_like(y_pred) / (pred_std * pred_std)
    
    return grad, hess

def correlation_metric(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[str, float, bool]:
    """
    Evaluation metric: correlation between predicted hardness and recall.
    """
    actual_hardness = y_true.get_label()
    y_pred = y_pred.ravel()
    
    # Calculate correlation
    correlation = np.corrcoef(y_pred, actual_hardness)[0, 1]
    
    # Return negative correlation as metric (to maximize)
    return 'correlation', correlation, True

def correlation_eval(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[str, float, bool]:
    """
    Evaluation metric: negative correlation with recall (to maximize).
    """
    actual_hardness = y_true.get_label()
    y_pred = y_pred.ravel()
    recalls = 1.0 - actual_hardness  # Assume recalls = 1 - hardness
    
    correlation = np.corrcoef(y_pred, recalls)[0, 1]
    # Return negative correlation as metric (LightGBM maximizes, so we return -corr to minimize)
    return 'neg_correlation', -correlation, False

def train_lightgbm_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_recalls: np.ndarray,
    num_trees: int = 500,  # Increased for more learning capacity
    max_depth: int = 20,
    learning_rate: float = 0.02,  # Further reduced from 0.05 for better convergence
    min_samples_split: int = 5,
    use_correlation_loss: bool = True
) -> lgb.Booster:
    """
    Train LightGBM model with correlation-based loss.
    
    Args:
        train_features: Training features (n_samples, n_features)
        train_targets: Training targets (hardness values)
        train_recalls: Training recalls (for correlation calculation)
        num_trees: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        min_samples_split: Minimum samples to split
        use_correlation_loss: Whether to use correlation-based loss
    
    Returns:
        Trained LightGBM model
    """
    print(f"Training LightGBM model with {num_trees} trees, depth {max_depth}")
    print(f"  Training samples: {len(train_features)}")
    print(f"  Features: {train_features.shape[1]}")
    
    # Create dataset
    train_data = lgb.Dataset(train_features, label=train_targets)
    
    # Parameters - optimized for correlation
    # num_leaves should be <= 131072 (LightGBM limit)
    # For depth 20, 2^20 = 1048576 is too large, so we cap it
    num_leaves = min(2 ** max_depth, 32768)  # Cap at 32768 to stay well below limit
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': min_samples_split,
        'lambda_l1': 0.1,  # L1 regularization
        'lambda_l2': 0.1,  # L2 regularization
        'min_gain_to_split': 0.0,
        'verbose': -1,
        'seed': 42,
    }
    
    if use_correlation_loss:
        # Use weighted MSE: weight samples by inverse recall to emphasize hard queries
        print("  Using weighted MSE with correlation-based evaluation")
        # Create sample weights: higher weight for hard queries (low recall)
        # Use piecewise weighting similar to C++ code
        sample_weights = np.ones(len(train_recalls))
        for i in range(len(train_recalls)):
            hardness = 1.0 - train_recalls[i]
            if hardness >= 0.9:
                sample_weights[i] = np.power(hardness, 2.5) * 3.0 + 0.1
            elif hardness >= 0.7:
                sample_weights[i] = np.power(hardness, 2.0) * 2.0 + 0.1
            else:
                sample_weights[i] = hardness * 0.5 + 0.1
        sample_weights = sample_weights / np.mean(sample_weights)  # Normalize
        
        # Use weighted dataset
        train_data = lgb.Dataset(train_features, label=train_targets, weight=sample_weights)
        params['objective'] = 'regression'
        params['metric'] = 'rmse'
        
        # Add custom evaluation metric for correlation
        feval = correlation_eval
    else:
        params['objective'] = 'regression'
        params['metric'] = 'rmse'
        feval = None
    
    # Split data for validation to prevent overfitting
    print("  Splitting data for validation (80% train, 20% val)...")
    train_size = int(len(train_features) * 0.8)
    val_features = train_features[train_size:]
    val_targets = train_targets[train_size:]
    val_recalls = train_recalls[train_size:]
    
    train_features_subset = train_features[:train_size]
    train_targets_subset = train_targets[:train_size]
    train_recalls_subset = train_recalls[:train_size]
    
    # Create validation dataset (no weights for validation to get unbiased metrics)
    val_data = lgb.Dataset(val_features, label=val_targets)
    
    # Recreate training dataset with subset
    if use_correlation_loss:
        train_sample_weights = np.ones(len(train_recalls_subset))
        for i in range(len(train_recalls_subset)):
            hardness = 1.0 - train_recalls_subset[i]
            if hardness >= 0.9:
                train_sample_weights[i] = np.power(hardness, 2.5) * 3.0 + 0.1
            elif hardness >= 0.7:
                train_sample_weights[i] = np.power(hardness, 2.0) * 2.0 + 0.1
            else:
                train_sample_weights[i] = hardness * 0.5 + 0.1
        train_sample_weights = train_sample_weights / np.mean(train_sample_weights)
        train_data = lgb.Dataset(train_features_subset, label=train_targets_subset, weight=train_sample_weights)
    else:
        train_data = lgb.Dataset(train_features_subset, label=train_targets_subset)
    
    # Train model with validation set
    callbacks = [lgb.log_evaluation(period=100)]  # Log every 100 iterations for large datasets
    # Use early stopping based on validation RMSE, but with more patience for large datasets
    callbacks.append(lgb.early_stopping(stopping_rounds=200, verbose=False, first_metric_only=True))
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_trees,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        feval=feval,
        callbacks=callbacks
    )
    
    # Evaluate on full training set for final metrics
    train_data_full = lgb.Dataset(train_features, label=train_targets)
    
    # Calculate correlation on validation data
    val_pred = model.predict(val_features, num_iteration=model.best_iteration)
    val_corr = np.corrcoef(val_pred, val_targets)[0, 1]
    val_recall_corr = np.corrcoef(val_pred, val_recalls)[0, 1]
    
    # Also calculate on full training set
    train_pred = model.predict(train_features, num_iteration=model.best_iteration)
    train_corr = np.corrcoef(train_pred, train_targets)[0, 1]
    train_recall_corr = np.corrcoef(train_pred, train_recalls)[0, 1]
    
    print(f"  Training correlation (pred vs hardness): {train_corr:.4f}")
    print(f"  Training correlation (pred vs recall): {train_recall_corr:.4f}")
    print(f"  Validation correlation (pred vs hardness): {val_corr:.4f}")
    print(f"  Validation correlation (pred vs recall): {val_recall_corr:.4f}")
    print(f"  Best iteration: {model.best_iteration}")
    
    return model

def main():
    if len(sys.argv) < 3:
        print("Usage: train_lightgbm_hardness_predictor.py <feature_cache_path> <output_model_path> [num_trees] [max_depth] [learning_rate]")
        sys.exit(1)
    
    feature_cache_path = sys.argv[1]
    output_model_path = sys.argv[2]
    num_trees = int(sys.argv[3]) if len(sys.argv) > 3 else 500  # Increased to 500 for more learning capacity
    max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    learning_rate = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02  # Further reduced to 0.02 for better convergence
    
    # Load features from binary file (same format as C++ code)
    print(f"Loading features from {feature_cache_path}...")
    
    # Read binary file
    with open(feature_cache_path, 'rb') as f:
        # Read header (size_t is typically 8 bytes on 64-bit systems)
        n_samples_bytes = f.read(8)
        n_features_bytes = f.read(8)
        n_samples = int(np.frombuffer(n_samples_bytes, dtype=np.uint64)[0])
        n_features = int(np.frombuffer(n_features_bytes, dtype=np.uint64)[0])
        
        print(f"  Samples: {n_samples}, Features: {n_features}")
        
        # Read features
        features = np.frombuffer(f.read(n_samples * n_features * 4), dtype=np.float32)
        features = features.reshape(n_samples, n_features)
        
        # Read targets
        targets = np.frombuffer(f.read(n_samples * 4), dtype=np.float32)
        
        # Read recalls
        recalls = np.frombuffer(f.read(n_samples * 4), dtype=np.float32)
    
    print(f"Loaded {len(features)} samples")
    
    # Train model
    model = train_lightgbm_model(
        features,
        targets,
        recalls,
        num_trees=num_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_correlation_loss=True
    )
    
    # Save model
    model.save_model(output_model_path)
    print(f"Model saved to {output_model_path}")
    
    # Save model info
    info_path = output_model_path + ".info.json"
    with open(info_path, 'w') as f:
        json.dump({
            'num_trees': int(num_trees),
            'max_depth': int(max_depth),
            'n_features': int(n_features),
            'n_samples': int(n_samples),
        }, f, indent=2)
    
    print("Training completed!")

if __name__ == '__main__':
    main()

