#!/usr/bin/env python3
"""
Visualize Hardness Predictor results: Recall vs Hardness scatter plot
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def load_results(json_path):
    """Load hardness predictor results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_scatter_plot(data, output_path):
    """Create scatter plot of Recall vs Predicted Hardness"""
    queries = data['queries']
    recalls = [q['recall'] for q in queries]
    hardness = [q['hardness'] for q in queries]
    predicted_hardness = [q['predicted_hardness'] for q in queries]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Actual Hardness vs Predicted Hardness
    ax1 = axes[0]
    ax1.scatter(hardness, predicted_hardness, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(min(hardness), min(predicted_hardness))
    max_val = max(max(hardness), max(predicted_hardness))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (Perfect Prediction)')
    
    ax1.set_xlabel('Actual Hardness', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Hardness', fontsize=12, fontweight='bold')
    ax1.set_title('Hardness Prediction: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add correlation and error metrics
    if np.std(hardness) > 1e-6 and np.std(predicted_hardness) > 1e-6:
        corr = np.corrcoef(hardness, predicted_hardness)[0, 1]
    else:
        corr = 0.0
    mae = np.mean(np.abs(np.array(hardness) - np.array(predicted_hardness)))
    rmse = np.sqrt(np.mean((np.array(hardness) - np.array(predicted_hardness))**2))
    
    stats_text = f'Correlation: {corr:.3f}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}'
    ax1.text(0.05, 0.95, stats_text, 
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Recall vs Predicted Hardness
    ax2 = axes[1]
    ax2.scatter(recalls, predicted_hardness, alpha=0.6, s=50, c='coral', edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Hardness', fontsize=12, fontweight='bold')
    ax2.set_title('Recall vs Predicted Hardness', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation
    if np.std(recalls) > 1e-6 and np.std(predicted_hardness) > 1e-6:
        corr2 = np.corrcoef(recalls, predicted_hardness)[0, 1]
    else:
        corr2 = 0.0
    
    stats_text2 = f'Correlation: {corr2:.3f}'
    ax2.text(0.05, 0.95, stats_text2, 
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add best feature combination info
    best_combo = data.get('best_feature_combination', [])
    best_mse = data.get('best_mse', 0.0)
    num_trees = data.get('num_trees', 'N/A')
    
    title_text = f'Hardness Predictor Results (Random Forest)'
    if num_trees != 'N/A':
        title_text += f'\nTrees: {num_trees}'
    title_text += f'\nBest Features: {", ".join(best_combo[:5])}...' if len(best_combo) > 5 else f'\nBest Features: {", ".join(best_combo)}'
    title_text += f'\nBest MSE: {best_mse:.4f}'
    
    fig.suptitle(title_text, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nScatter plots saved to: {output_path}")
    print(f"Statistics:")
    print(f"  Hardness Prediction - Correlation: {corr:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    print(f"  Recall vs Predicted Hardness - Correlation: {corr2:.3f}")
    print(f"  Best feature combination: {', '.join(best_combo)}")
    print(f"  Best MSE: {best_mse:.4f}")
    if num_trees != 'N/A':
        print(f"  Number of trees: {num_trees}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_hardness_predictor.py <hardness_predictor_results.json> [output_image.png]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else json_path.replace('.json', '_scatter.png')
    
    data = load_results(json_path)
    create_scatter_plot(data, output_path)


