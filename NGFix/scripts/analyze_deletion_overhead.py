#!/usr/bin/env python3
"""
Analyze deletion overhead and generate detailed report.
"""

import pandas as pd
import numpy as np
import sys
import os

def analyze_overhead(csv_file, output_file):
    """Analyze deletion overhead from CSV file and generate markdown report."""
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    if df.empty:
        print("Error: CSV file is empty")
        return
    
    # Group by deletion_type and deletion_percentage
    summary = df.groupby(['deletion_type', 'deletion_percentage']).agg({
        'total_time_us': ['mean', 'std'],
        'distance_time_us': ['mean', 'std'],
        'is_deleted_time_us': ['mean', 'std'],
        'lock_time_us': ['mean', 'std'],
        'queue_time_us': ['mean', 'std'],
        'neighbor_time_us': ['mean', 'std'],
        'visited_time_us': ['mean', 'std'],
        'other_time_us': ['mean', 'std'],
        'num_distance': 'mean',
        'num_is_deleted': 'mean',
        'num_lock': 'mean',
        'num_queue': 'mean',
        'num_neighbor': 'mean',
        'num_visited': 'mean',
        'num_deleted_visited': 'mean',
        'num_valid_visited': 'mean',
        'recall': 'mean',
        'ndc': 'mean',
        'rderr': 'mean'
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in summary.columns.values]
    
    with open(output_file, 'w') as f:
        f.write("# Deletion Overhead Analysis Report\n\n")
        f.write("This report analyzes the detailed overhead breakdown for lazy deletion vs real deletion.\n\n")
        
        for del_pct in sorted(df['deletion_percentage'].unique()):
            f.write(f"## {del_pct:.0f}% Deletion\n\n")
            
            lazy_data = summary[(summary['deletion_percentage'] == del_pct) & 
                               (summary['deletion_type'] == 'lazy_deletion')]
            real_data = summary[(summary['deletion_percentage'] == del_pct) & 
                              (summary['deletion_type'] == 'real_deletion')]
            
            if lazy_data.empty or real_data.empty:
                f.write("Insufficient data for this deletion percentage.\n\n")
                continue
            
            lazy = lazy_data.iloc[0]
            real = real_data.iloc[0]
            
            # Time breakdown
            f.write("### Time Breakdown (microseconds)\n\n")
            f.write("| Component | Lazy Deletion | Real Deletion | Difference | Diff % |\n")
            f.write("|-----------|---------------|---------------|------------|--------|\n")
            
            components = [
                ('Total Time', 'total_time_us'),
                ('Distance Computation', 'distance_time_us'),
                ('is_deleted Check', 'is_deleted_time_us'),
                ('Lock Acquisition', 'lock_time_us'),
                ('Queue Operations', 'queue_time_us'),
                ('Neighbor Access', 'neighbor_time_us'),
                ('Visited Check', 'visited_time_us'),
                ('Other', 'other_time_us')
            ]
            
            for name, col in components:
                lazy_val = lazy[f'{col}_mean']
                real_val = real[f'{col}_mean']
                diff = real_val - lazy_val
                diff_pct = (diff / lazy_val * 100) if lazy_val > 0 else 0
                f.write(f"| {name} | {lazy_val:.2f} | {real_val:.2f} | {diff:+.2f} | {diff_pct:+.2f}% |\n")
            
            f.write("\n### Time Percentage Breakdown\n\n")
            f.write("| Component | Lazy Deletion % | Real Deletion % | Difference |\n")
            f.write("|-----------|------------------|------------------|------------|\n")
            
            lazy_total = lazy['total_time_us_mean']
            real_total = real['total_time_us_mean']
            
            for name, col in components[1:]:  # Skip total
                lazy_pct = (lazy[f'{col}_mean'] / lazy_total * 100) if lazy_total > 0 else 0
                real_pct = (real[f'{col}_mean'] / real_total * 100) if real_total > 0 else 0
                diff_pct = real_pct - lazy_pct
                f.write(f"| {name} | {lazy_pct:.2f}% | {real_pct:.2f}% | {diff_pct:+.2f}% |\n")
            
            # Operation counts
            f.write("\n### Operation Counts\n\n")
            f.write("| Operation | Lazy Deletion | Real Deletion | Difference |\n")
            f.write("|-----------|---------------|---------------|------------|\n")
            
            counts = [
                ('Distance Computations', 'num_distance'),
                ('is_deleted Checks', 'num_is_deleted'),
                ('Lock Acquires', 'num_lock'),
                ('Queue Operations', 'num_queue'),
                ('Neighbor Accesses', 'num_neighbor'),
                ('Visited Checks', 'num_visited'),
                ('Deleted Nodes Visited', 'num_deleted_visited'),
                ('Valid Nodes Visited', 'num_valid_visited')
            ]
            
            for name, col in counts:
                lazy_val = lazy[col]
                real_val = real[col]
                diff = real_val - lazy_val
                f.write(f"| {name} | {lazy_val:.1f} | {real_val:.1f} | {diff:+.1f} |\n")
            
            # Performance metrics
            f.write("\n### Performance Metrics\n\n")
            f.write("| Metric | Lazy Deletion | Real Deletion | Difference |\n")
            f.write("|--------|---------------|---------------|------------|\n")
            f.write(f"| Recall | {lazy['recall']:.6f} | {real['recall']:.6f} | {real['recall'] - lazy['recall']:+.6f} |\n")
            f.write(f"| NDC | {lazy['ndc']:.2f} | {real['ndc']:.2f} | {real['ndc'] - lazy['ndc']:+.2f} |\n")
            f.write(f"| Rderr | {lazy['rderr']:.6f} | {real['rderr']:.6f} | {real['rderr'] - lazy['rderr']:+.6f} |\n")
            
            # Key insights
            f.write("\n### Key Insights\n\n")
            
            total_diff = real['total_time_us_mean'] - lazy['total_time_us_mean']
            total_diff_pct = (total_diff / lazy['total_time_us_mean'] * 100) if lazy['total_time_us_mean'] > 0 else 0
            
            if total_diff > 0:
                f.write(f"- **Real deletion is {total_diff:.2f} us ({total_diff_pct:.2f}%) slower than lazy deletion**\n")
            else:
                f.write(f"- **Real deletion is {abs(total_diff):.2f} us ({abs(total_diff_pct):.2f}%) faster than lazy deletion**\n")
            
            # Find the component with largest difference
            max_diff = 0
            max_diff_component = ""
            for name, col in components[1:]:
                diff = abs(real[f'{col}_mean'] - lazy[f'{col}_mean'])
                if diff > max_diff:
                    max_diff = diff
                    max_diff_component = name
            
            f.write(f"- **Largest overhead difference: {max_diff_component} ({max_diff:.2f} us)**\n")
            
            # Analyze why real deletion might be slower
            if real['total_time_us_mean'] > lazy['total_time_us_mean']:
                f.write("\n#### Possible Reasons for Real Deletion Being Slower:\n\n")
                
                if real['distance_time_us_mean'] > lazy['distance_time_us_mean']:
                    f.write(f"1. **Distance computation overhead**: Real deletion has {real['distance_time_us_mean'] - lazy['distance_time_us_mean']:.2f} us more distance computation time\n")
                    f.write("   - This could be due to cache misses from graph structure changes\n")
                
                if real['neighbor_time_us_mean'] > lazy['neighbor_time_us_mean']:
                    f.write(f"2. **Neighbor access overhead**: Real deletion has {real['neighbor_time_us_mean'] - lazy['neighbor_time_us_mean']:.2f} us more neighbor access time\n")
                    f.write("   - Graph repair may have changed memory layout, causing cache misses\n")
                
                if real['other_time_us_mean'] > lazy['other_time_us_mean']:
                    f.write(f"3. **Other overhead**: Real deletion has {real['other_time_us_mean'] - lazy['other_time_us_mean']:.2f} us more unaccounted time\n")
                    f.write("   - This includes memory allocation, function call overhead, etc.\n")
                
                if real['num_deleted_visited'] > 0:
                    f.write(f"4. **Deleted nodes still visited**: Real deletion visited {real['num_deleted_visited']:.1f} deleted nodes\n")
                    f.write("   - This shouldn't happen with real deletion - may indicate a bug\n")
            
            f.write("\n")
    
    print(f"Analysis report generated: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 analyze_deletion_overhead.py <input_csv> <output_md>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2]
    
    analyze_overhead(csv_file, output_file)
