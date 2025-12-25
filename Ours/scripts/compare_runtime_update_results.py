#!/usr/bin/env python3
"""
Compare runtime update results from two CSV files (first 43 minutes)
Creates line plots comparing different metrics between the two datasets
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import sys
import os

try:
    from scipy.interpolate import make_interp_spline, UnivariateSpline
except ImportError:
    make_interp_spline = None
    UnivariateSpline = None

def load_and_filter_data(csv_path, max_minutes=43, add_baseline=True):
    """Load CSV data and filter to first max_minutes minutes"""
    df = pd.read_csv(csv_path)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get start time
    start_time = df['timestamp'].iloc[0]
    
    # Filter to first max_minutes minutes
    end_time = start_time + timedelta(minutes=max_minutes)
    df_filtered = df[df['timestamp'] <= end_time].copy()
    
    # Calculate minutes from start
    df_filtered['minutes_from_start'] = (df_filtered['timestamp'] - start_time).dt.total_seconds() / 60.0
    
    # Add baseline data point at x=0 if requested
    if add_baseline:
        # Baseline values from index baseline performance test
        baseline_values = {
            'avg_recall': 0.989289,
            'avg_latency_ms': 4.2182,
            'avg_ndc': 16761.06,
        }
        
        # Create a baseline row with all columns from df_filtered
        baseline_row = df_filtered.iloc[0].copy()
        
        # Set baseline values for metrics
        for key, value in baseline_values.items():
            if key in baseline_row.index:
                baseline_row[key] = value
        
        # Set timestamp and minutes_from_start
        baseline_row['timestamp'] = start_time
        baseline_row['minutes_from_start'] = 0.0
        
        # Set other columns to baseline state (no operations yet)
        for col in baseline_row.index:
            if col not in baseline_values and col not in ['timestamp', 'minutes_from_start']:
                baseline_row[col] = 0
        
        # Prepend baseline row to dataframe
        df_filtered = pd.concat([pd.DataFrame([baseline_row]), df_filtered], ignore_index=True)
    
    return df_filtered, start_time


def prepare_smooth_curve(x_values, y_values, num_points=300, smooth_strength=2.0):
    """
    Prepare smoothed x/y arrays.
    - Prefer a smoothing spline (UnivariateSpline) to avoid高阶振荡；
    - Fall back to cubic interpolating spline;
    - Fall back to original points when缺少 SciPy或点数不足。
    smooth_strength 越大，曲线越接近直线。
    """
    data = pd.DataFrame({'x': x_values, 'y': y_values}).sort_values('x')
    # Ensure strictly increasing x by dropping duplicate x entries
    data = data.loc[~data['x'].duplicated(keep='first')]
    x_arr = data['x'].values
    y_arr = data['y'].values

    # Require at least 4 points for a cubic spline and SciPy installed
    if len(x_arr) < 4 or (make_interp_spline is None and UnivariateSpline is None):
        return x_arr, y_arr

    x_smooth = np.linspace(x_arr.min(), x_arr.max(), num_points)

    # Prefer smoothing spline for stability and直线趋向
    if UnivariateSpline is not None:
        # s 控制平滑度；与方差和样本量相关，值越大越“直”
        s = smooth_strength * np.var(y_arr) * max(x_arr)
        spline = UnivariateSpline(x_arr, y_arr, k=3, s=s)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth

    # Fallback: cubic interpolating spline (may更贴合数据)
    spline = make_interp_spline(x_arr, y_arr, k=3)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

def create_comparison_plots(df1, df2, label1, label2, output_path):
    """Create comparison line plots for multiple metrics"""
    
    # Determine common columns to plot
    common_cols = set(df1.columns) & set(df2.columns)
    include_cols = {'avg_latency_ms', 'avg_recall'} #,  "avg_ndc", "index_size"}
    plot_cols = [col for col in common_cols if col in include_cols]
    
    # Create subplots - arrange in a grid
    num_plots = len(plot_cols)
    cols_per_row = 1
    rows = (num_plots + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    color_map = {
        'line1': '#3b6ea5',
        'points1': '#7fa6d6',
        'line2': '#d55a3c',
        'points2': '#f2a074',
    }
    
    # Plot each metric
    for idx, col in enumerate(plot_cols):
        ax = axes[idx]
        
        # Plot raw points
        ax.scatter(df1['minutes_from_start'], df1[col],
                   color=color_map['points1'], s=24, alpha=0.85,
                   edgecolor='none', label=f'{label1} points', zorder=3)
        ax.scatter(df2['minutes_from_start'], df2[col],
                   color=color_map['points2'], s=24, alpha=0.85,
                   edgecolor='none', label=f'{label2} points', zorder=3)

        # Plot smoothed curves
        # x1_smooth, y1_smooth = prepare_smooth_curve(df1['minutes_from_start'], df1[col])
        # x2_smooth, y2_smooth = prepare_smooth_curve(df2['minutes_from_start'], df2[col])

        # ax.plot(x1_smooth, y1_smooth,
        #         linewidth=2.4, label=f'{label1} (smooth)',
        #         alpha=0.9, color=color_map['line1'])
        # ax.plot(x2_smooth, y2_smooth,
        #         linewidth=2.4, label=f'{label2} (smooth)',
        #         alpha=0.9, color=color_map['line2'])
        
        ax.set_xlabel('Minutes from Start', fontsize=11, fontweight='bold')
        ax.set_ylabel(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_title(f'{col.replace("_", " ").title()} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format x-axis to show integer minutes
        ax.set_xlim(left=0)
        if df1['minutes_from_start'].max() > 0 or df2['minutes_from_start'].max() > 0:
            max_minutes = max(df1['minutes_from_start'].max(), df2['minutes_from_start'].max())
            ax.set_xticks(range(0, int(max_minutes) + 1, 5))
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Runtime Update Results Comparison: {label1} vs {label2}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plots saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    for col in plot_cols:
        print(f"\n{col.replace('_', ' ').title()}:")
        print(f"  {label1}:")
        if col in df1.columns:
            print(f"    Mean: {df1[col].mean():.4f}")
            print(f"    Std:  {df1[col].std():.4f}")
            print(f"    Min:  {df1[col].min():.4f}")
            print(f"    Max:  {df1[col].max():.4f}")
        print(f"  {label2}:")
        if col in df2.columns:
            print(f"    Mean: {df2[col].mean():.4f}")
            print(f"    Std:  {df2[col].std():.4f}")
            print(f"    Min:  {df2[col].min():.4f}")
            print(f"    Max:  {df2[col].max():.4f}")

if __name__ == '__main__':
    # Default file paths
    default_file1 = '/workspace/OOD-ANNS/Ours/data/runtime_update_test/runtime_update_results_ngfix.csv'
    default_file2 = '/workspace/OOD-ANNS/Ours/data/runtime_update_test/runtime_update_results_q1000_i20_d2.csv'
    
    if len(sys.argv) >= 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        label1 = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(file1).replace('.csv', '')
        label2 = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(file2).replace('.csv', '')
        output_path = sys.argv[5] if len(sys.argv) > 5 else 'runtime_update_comparison.png'
    else:
        file1 = default_file1
        file2 = default_file2
        label1 = 'ngfix'
        label2 = 'ReLANCE'
        output_path = '/workspace/OOD-ANNS/Ours/pictures/runtime_update_comparison.png'
    
    print(f"Loading data from:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    
    # Load and filter data (first 43 minutes)
    print("\nFiltering to first 43 minutes...")
    print("Adding baseline data point at x=0 (Recall: 0.989289, Latency: 4.2182 ms, NDC: 16761.06)...")
    df1, start1 = load_and_filter_data(file1, max_minutes=300, add_baseline=True)
    df2, start2 = load_and_filter_data(file2, max_minutes=300, add_baseline=True)
    
    print(f"  {label1}: {len(df1)} data points (from {start1} to {df1['timestamp'].iloc[-1]})")
    print(f"  {label2}: {len(df2)} data points (from {start2} to {df2['timestamp'].iloc[-1]})")
    
    # Create comparison plots
    create_comparison_plots(df1, df2, label1, label2, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")

