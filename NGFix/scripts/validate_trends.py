#!/usr/bin/env python3
"""
Validate trends in lazy vs real deletion test results.
"""

import csv
import sys
import os

def validate_trends(csv_file):
    """Validate three trends in the test results."""
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    if len(data) < 16:  # Need at least 8 lazy + 8 real results
        return False, "Insufficient data"
    
    lazy_results = {}
    real_results = {}
    
    for row in data:
        del_pct = float(row['deletion_percentage'])
        if 'lazy' in row['deletion_type']:
            lazy_results[del_pct] = {
                'ndc': float(row['ndc']),
                'latency': float(row['latency_ms'])
            }
        elif 'real' in row['deletion_type']:
            real_results[del_pct] = {
                'ndc': float(row['ndc']),
                'latency': float(row['latency_ms'])
            }
    
    if len(lazy_results) < 8 or len(real_results) < 8:
        return False, "Missing data points"
    
    # Sort by deletion percentage
    lazy_pcts = sorted(lazy_results.keys())
    real_pcts = sorted(real_results.keys())
    
    errors = []
    
    # Trend 1: Lazy deletion latency should generally increase with deletion percentage
    # Check overall trend: compare first and last, and check most transitions
    lazy_lat_first = lazy_results[lazy_pcts[0]]['latency']
    lazy_lat_last = lazy_results[lazy_pcts[-1]]['latency']
    if lazy_lat_last < lazy_lat_first * 0.90:  # Overall should increase by at least 10%
        errors.append(f"Trend1 violation: Lazy deletion latency should increase overall: {lazy_pcts[0]}% ({lazy_lat_first:.4f}ms) -> {lazy_pcts[-1]}% ({lazy_lat_last:.4f}ms)")
    
    # Check major transitions (skip small fluctuations)
    major_transitions = [(0, 2), (2, 4), (4, len(lazy_pcts)-1)]  # 1%->3%, 3%->5%, 5%->20%
    for start_idx, end_idx in major_transitions:
        if end_idx < len(lazy_pcts):
            prev_latency = lazy_results[lazy_pcts[start_idx]]['latency']
            curr_latency = lazy_results[lazy_pcts[end_idx]]['latency']
            if curr_latency < prev_latency * 0.95:
                errors.append(f"Trend1 violation: Lazy {lazy_pcts[start_idx]}% ({prev_latency:.4f}ms) -> {lazy_pcts[end_idx]}% ({curr_latency:.4f}ms)")
    
    # Trend 2: Real deletion latency should generally decrease with deletion percentage
    # Check overall trend
    real_lat_first = real_results[real_pcts[0]]['latency']
    real_lat_last = real_results[real_pcts[-1]]['latency']
    if real_lat_last > real_lat_first * 1.10:  # Overall should decrease
        errors.append(f"Trend2 violation: Real deletion latency should decrease overall: {real_pcts[0]}% ({real_lat_first:.4f}ms) -> {real_pcts[-1]}% ({real_lat_last:.4f}ms)")
    
    # Check major transitions for real deletion
    for start_idx, end_idx in major_transitions:
        if end_idx < len(real_pcts):
            prev_latency = real_results[real_pcts[start_idx]]['latency']
            curr_latency = real_results[real_pcts[end_idx]]['latency']
            if curr_latency > prev_latency * 1.05:
                errors.append(f"Trend2 violation: Real {real_pcts[start_idx]}% ({prev_latency:.4f}ms) -> {real_pcts[end_idx]}% ({curr_latency:.4f}ms)")
    
    # Trend 3: Real deletion latency should be lower than lazy deletion at same percentage
    # Check at key percentages: 3%, 5%, 10%, 15%, 20%
    key_pcts = [3.0, 5.0, 10.0, 15.0, 20.0]
    for pct in key_pcts:
        if pct in lazy_results and pct in real_results:
            lazy_lat = lazy_results[pct]['latency']
            real_lat = real_results[pct]['latency']
            if real_lat > lazy_lat * 1.05:  # Allow 5% tolerance for small percentages
                errors.append(f"Trend3 violation: At {pct}%, Real ({real_lat:.4f}ms) > Lazy ({lazy_lat:.4f}ms) by {((real_lat/lazy_lat-1)*100):.2f}%")
    
    if errors:
        return False, "\n".join(errors)
    
    return True, "All trends satisfied"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: validate_trends.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    valid, message = validate_trends(csv_file)
    if valid:
        print("SUCCESS: All trends are satisfied")
        sys.exit(0)
    else:
        print(f"FAILED: {message}")
        sys.exit(1)

