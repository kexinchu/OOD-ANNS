#!/usr/bin/env python3
"""
分析重建操作对query延迟的影响
"""

import csv
import sys
from collections import defaultdict

def analyze_rebuild_impact(csv_file):
    """分析重建期间的query延迟"""
    
    queries = []
    rebuilds = []
    
    # 读取CSV文件
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] == 'query':
                queries.append({
                    'timestamp': int(row['timestamp_us']),
                    'latency': int(row['latency_us']),
                    'progress': int(row['progress']),
                    'thread_safe': row['thread_safe'] == '1'
                })
            elif row['type'] in ['insert', 'delete'] and row['phase'] == 'rebuild':
                rebuilds.append({
                    'timestamp': int(row['timestamp_us']),
                    'latency': int(row['latency_us']),
                    'type': row['type']
                })
    
    if not rebuilds:
        print("No rebuild operations found!")
        return
    
    print(f"=== Rebuild Impact Analysis ===")
    print(f"Total queries: {len(queries)}")
    print(f"Total rebuilds: {len(rebuilds)}")
    print()
    
    # 分析每个重建期间的query延迟
    rebuild_windows = []
    for rebuild in rebuilds:
        rebuild_start = rebuild['timestamp']
        rebuild_end = rebuild_start + rebuild['latency']
        rebuild_windows.append({
            'start': rebuild_start,
            'end': rebuild_end,
            'latency': rebuild['latency'],
            'type': rebuild['type']
        })
    
    # 分类query：重建期间 vs 非重建期间
    queries_during_rebuild = []
    queries_outside_rebuild = []
    
    for query in queries:
        in_rebuild = False
        for window in rebuild_windows:
            if window['start'] <= query['timestamp'] <= window['end']:
                in_rebuild = True
                break
        if in_rebuild:
            queries_during_rebuild.append(query)
        else:
            queries_outside_rebuild.append(query)
    
    print(f"Queries during rebuild: {len(queries_during_rebuild)}")
    print(f"Queries outside rebuild: {len(queries_outside_rebuild)}")
    print()
    
    # 计算统计信息
    def calc_stats(query_list, label):
        if not query_list:
            print(f"{label}: No queries")
            return
        
        latencies = [q['latency'] for q in query_list if q['thread_safe']]
        if not latencies:
            print(f"{label}: No valid queries")
            return
        
        latencies.sort()
        avg = sum(latencies) / len(latencies)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        print(f"{label}:")
        print(f"  Count: {len(latencies)}")
        print(f"  Average latency: {avg / 1000.0:.3f} ms")
        print(f"  P50 latency: {p50 / 1000.0:.3f} ms")
        print(f"  P95 latency: {p95 / 1000.0:.3f} ms")
        print(f"  P99 latency: {p99 / 1000.0:.3f} ms")
        print()
    
    calc_stats(queries_during_rebuild, "Queries DURING rebuild")
    calc_stats(queries_outside_rebuild, "Queries OUTSIDE rebuild")
    
    # 分析每个重建窗口的query延迟
    print("=== Per-Rebuild Analysis ===")
    for i, window in enumerate(rebuild_windows):
        window_queries = [q for q in queries 
                         if window['start'] <= q['timestamp'] <= window['end']]
        
        if window_queries:
            latencies = [q['latency'] for q in window_queries if q['thread_safe']]
            if latencies:
                avg = sum(latencies) / len(latencies)
                max_lat = max(latencies)
                print(f"Rebuild {i+1} ({window['type']}):")
                print(f"  Rebuild latency: {window['latency'] / 1000.0:.3f} ms")
                print(f"  Queries during rebuild: {len(latencies)}")
                print(f"  Average query latency: {avg / 1000.0:.3f} ms")
                print(f"  Max query latency: {max_lat / 1000.0:.3f} ms")
                print()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analyze_rebuild_impact.py <csv_file>")
        sys.exit(1)
    
    analyze_rebuild_impact(sys.argv[1])

