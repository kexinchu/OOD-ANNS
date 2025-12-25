#!/usr/bin/env python3
"""
Generate a markdown report comparing lazy deletion vs real deletion results.
"""

import argparse
import csv
import sys
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Generate markdown report for lazy vs real deletion comparison')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with test results')
    parser.add_argument('--output', type=str, required=True, help='Output markdown file path')
    return parser.parse_args()

def parse_deletion_percentage(stage_name):
    """Extract deletion percentage from stage name like 'lazy_deletion_10pct' or 'real_deletion_5pct'"""
    if 'pct' in stage_name:
        # Extract number before 'pct'
        parts = stage_name.split('_')
        for part in parts:
            if 'pct' in part:
                pct_str = part.replace('pct', '')
                try:
                    return float(pct_str)
                except ValueError:
                    pass
    return None

def main():
    args = parse_args()
    
    # Read CSV data
    results = []
    try:
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Organize results by deletion type and percentage
    lazy_results = {}
    real_results = {}
    
    for row in results:
        deletion_type = row['deletion_type']
        
        # Parse deletion percentage from stage name
        del_pct = parse_deletion_percentage(deletion_type)
        
        if del_pct is None or del_pct == 0:
            continue
        
        # CSV format: deletion_type, deletion_percentage, efs, recall, ndc, latency_ms, rderr, start_timestamp, end_timestamp
        try:
            # Parse based on actual CSV column names
            recall = float(row.get('recall', '0'))
            ndc = float(row.get('ndc', '0'))
            latency_ms = float(row.get('latency_ms', '0'))
            rderr = float(row.get('rderr', '0'))
            
            result_data = {
                'recall': recall,
                'ndc': ndc,
                'latency_ms': latency_ms,
                'rderr': rderr
            }
        except (ValueError, KeyError) as e:
            print(f"Warning: Skipping row due to parsing error: {e}", file=sys.stderr)
            print(f"Row data: {row}", file=sys.stderr)
            continue
        
        if 'lazy' in deletion_type:
            lazy_results[del_pct] = result_data
        elif 'real' in deletion_type:
            real_results[del_pct] = result_data
    
    # Generate markdown report
    with open(args.output, 'w') as f:
        f.write("# Lazy Deletion vs Real Deletion Comparison Report\n\n")
        f.write("This report compares the impact of lazy deletion (baseline) vs real deletion with NGFix repair on search performance.\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| Deletion % | Method | Recall | NDC | Latency (ms) | Rderr |\n")
        f.write("|------------|--------|--------|-----|--------------|-------|\n")
        
        # Sort percentages
        all_percentages = sorted(set(list(lazy_results.keys()) + list(real_results.keys())))
        
        for pct in all_percentages:
            if pct in lazy_results:
                lazy = lazy_results[pct]
                f.write(f"| {pct:.0f}% | Lazy Deletion | {lazy['recall']:.6f} | {lazy['ndc']:.2f} | {lazy['latency_ms']:.4f} | {lazy['rderr']:.6f} |\n")
            
            if pct in real_results:
                real = real_results[pct]
                f.write(f"| {pct:.0f}% | Real Deletion + NGFix | {real['recall']:.6f} | {real['ndc']:.2f} | {real['latency_ms']:.4f} | {real['rderr']:.6f} |\n")
        
        f.write("\n")
        
        # Detailed comparison
        f.write("## Detailed Comparison\n\n")
        
        for pct in all_percentages:
            f.write(f"### {pct:.0f}% Deletion\n\n")
            
            if pct in lazy_results and pct in real_results:
                lazy = lazy_results[pct]
                real = real_results[pct]
                
                recall_diff = real['recall'] - lazy['recall']
                recall_improvement = (recall_diff / lazy['recall'] * 100) if lazy['recall'] > 0 else 0
                
                ndc_diff = real['ndc'] - lazy['ndc']
                ndc_improvement = (ndc_diff / lazy['ndc'] * 100) if lazy['ndc'] > 0 else 0
                
                latency_diff = real['latency_ms'] - lazy['latency_ms']
                latency_improvement = (latency_diff / lazy['latency_ms'] * 100) if lazy['latency_ms'] > 0 else 0
                
                f.write("| Metric | Lazy Deletion | Real Deletion + NGFix | Difference | Improvement |\n")
                f.write("|--------|---------------|----------------------|------------|------------|\n")
                f.write(f"| Recall | {lazy['recall']:.6f} | {real['recall']:.6f} | {recall_diff:+.6f} | {recall_improvement:+.2f}% |\n")
                f.write(f"| NDC | {lazy['ndc']:.2f} | {real['ndc']:.2f} | {ndc_diff:+.2f} | {ndc_improvement:+.2f}% |\n")
                f.write(f"| Latency (ms) | {lazy['latency_ms']:.4f} | {real['latency_ms']:.4f} | {latency_diff:+.4f} | {latency_improvement:+.2f}% |\n")
                f.write(f"| Rderr | {lazy['rderr']:.6f} | {real['rderr']:.6f} | {real['rderr'] - lazy['rderr']:+.6f} | - |\n")
                f.write("\n")
            elif pct in lazy_results:
                lazy = lazy_results[pct]
                f.write("| Metric | Lazy Deletion |\n")
                f.write("|--------|---------------|\n")
                f.write(f"| Recall | {lazy['recall']:.6f} |\n")
                f.write(f"| NDC | {lazy['ndc']:.2f} |\n")
                f.write(f"| Latency (ms) | {lazy['latency_ms']:.4f} |\n")
                f.write(f"| Rderr | {lazy['rderr']:.6f} |\n")
                f.write("\n")
            elif pct in real_results:
                real = real_results[pct]
                f.write("| Metric | Real Deletion + NGFix |\n")
                f.write("|--------|----------------------|\n")
                f.write(f"| Recall | {real['recall']:.6f} |\n")
                f.write(f"| NDC | {real['ndc']:.2f} |\n")
                f.write(f"| Latency (ms) | {real['latency_ms']:.4f} |\n")
                f.write(f"| Rderr | {real['rderr']:.6f} |\n")
                f.write("\n")
        
        # Analysis
        f.write("## Analysis\n\n")
        f.write("### Key Observations\n\n")
        f.write("1. **Lazy Deletion (Baseline)**: Deleted nodes remain in the index and are still visited during search, but filtered from results.\n")
        f.write("2. **Real Deletion + NGFix**: Deleted nodes are completely removed from the index, and the graph is repaired using NGFix.\n")
        f.write("3. This comparison shows the impact of lazy deleted nodes on search performance.\n\n")
        
        f.write("### Metrics Explained\n\n")
        f.write("- **Recall**: Fraction of true nearest neighbors found in top-k results\n")
        f.write("- **NDC**: Number of distance computations (search cost)\n")
        f.write("- **Latency**: Average search time in milliseconds\n")
        f.write("- **Rderr**: Relative distance error\n\n")
    
    print(f"Markdown report generated: {args.output}")

if __name__ == '__main__':
    main()

