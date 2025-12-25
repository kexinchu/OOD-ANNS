#!/usr/bin/env python3
"""
Analyze deletion percentage test results with multiple efSearch values
"""

import csv
import sys
from collections import defaultdict

def analyze_results(csv_file):
    """Analyze and format the results"""
    
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    # Group by deletion percentage
    by_deletion = defaultdict(list)
    for row in results:
        deletion_pct = row['deletion_percentage']
        by_deletion[deletion_pct].append(row)
    
    # Print summary table
    print("=" * 120)
    print("NGFix Deletion Percentage Test Results Summary")
    print("=" * 120)
    print()
    
    # Table header
    print(f"{'Deletion %':<12} {'efSearch':<10} {'Recall':<10} {'NDC':<12} {'Latency (ms)':<15} {'Rel Dist Error':<15}")
    print("-" * 120)
    
    # Sort deletion percentages
    deletion_order = ['deletion_1pct', 'deletion_2pct', 'deletion_3pct', 'deletion_4pct', 
                      'deletion_5pct', 'deletion_10pct', 'deletion_15pct']
    efs_order = [100, 200, 300, 400, 500, 1000, 1500, 2000]
    
    for del_pct in deletion_order:
        if del_pct not in by_deletion:
            continue
        
        rows = by_deletion[del_pct]
        # Sort by efSearch
        rows_sorted = sorted(rows, key=lambda x: int(x['efs']))
        
        for row in rows_sorted:
            del_pct_display = del_pct.replace('deletion_', '').replace('pct', '%')
            print(f"{del_pct_display:<12} {row['efs']:<10} {float(row['recall']):<10.6f} "
                  f"{float(row['ndc']):<12.2f} {float(row['latency_ms']):<15.4f} "
                  f"{float(row['rderr']):<15.6f}")
        print("-" * 120)
    
    print()
    print("=" * 120)
    print("Analysis by Deletion Percentage")
    print("=" * 120)
    print()
    
    # For each deletion percentage, show the impact
    for del_pct in deletion_order:
        if del_pct not in by_deletion:
            continue
        
        rows = by_deletion[del_pct]
        rows_sorted = sorted(rows, key=lambda x: int(x['efs']))
        
        print(f"\n{del_pct.replace('deletion_', '').replace('pct', '%')} Deletion:")
        print(f"  efSearch | Recall    | NDC       | Latency (ms)")
        print(f"  ---------|-----------|-----------|-------------")
        for row in rows_sorted:
            print(f"  {int(row['efs']):<8} | {float(row['recall']):<9.6f} | "
                  f"{float(row['ndc']):<9.2f} | {float(row['latency_ms']):<11.4f}")
    
    print()
    print("=" * 120)
    print("Analysis by efSearch Value")
    print("=" * 120)
    print()
    
    # Group by efSearch
    by_efs = defaultdict(list)
    for row in results:
        by_efs[int(row['efs'])].append(row)
    
    for efs in sorted(by_efs.keys()):
        rows = by_efs[efs]
        rows_sorted = sorted(rows, key=lambda x: x['deletion_percentage'])
        
        print(f"\nefSearch = {efs}:")
        print(f"  Deletion % | Recall    | NDC       | Latency (ms)")
        print(f"  ----------|-----------|-----------|-------------")
        for row in rows_sorted:
            del_pct_display = row['deletion_percentage'].replace('deletion_', '').replace('pct', '%')
            print(f"  {del_pct_display:<10} | {float(row['recall']):<9.6f} | "
                  f"{float(row['ndc']):<9.2f} | {float(row['latency_ms']):<11.4f}")
    
    # Calculate recall impact
    print()
    print("=" * 120)
    print("Recall Impact Analysis (compared to 1% deletion baseline)")
    print("=" * 120)
    print()
    
    # Get baseline (1% deletion) recalls for each efSearch
    baseline = {}
    for row in by_deletion.get('deletion_1pct', []):
        baseline[int(row['efs'])] = float(row['recall'])
    
    print(f"{'Deletion %':<12} {'efSearch':<10} {'Recall':<10} {'Recall Drop':<15} {'% Drop':<10}")
    print("-" * 120)
    
    for del_pct in deletion_order:
        if del_pct not in by_deletion or del_pct == 'deletion_1pct':
            continue
        
        rows = by_deletion[del_pct]
        rows_sorted = sorted(rows, key=lambda x: int(x['efs']))
        
        for row in rows_sorted:
            efs = int(row['efs'])
            recall = float(row['recall'])
            baseline_recall = baseline.get(efs, recall)
            recall_drop = baseline_recall - recall
            pct_drop = (recall_drop / baseline_recall * 100) if baseline_recall > 0 else 0
            
            del_pct_display = del_pct.replace('deletion_', '').replace('pct', '%')
            print(f"{del_pct_display:<12} {efs:<10} {recall:<10.6f} {recall_drop:<15.6f} {pct_drop:<10.2f}%")
        print("-" * 120)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_deletion_percentage_efs.py <results_csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyze_results(csv_file)



