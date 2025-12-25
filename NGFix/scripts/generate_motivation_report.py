#!/usr/bin/env python3
"""
Generate markdown report from motivation test results.
"""

import argparse
import csv
import os
from datetime import datetime

def parse_timestamp(ts_str):
    """Parse timestamp string to datetime object."""
    try:
        # Format: "2024-01-01 12:00:00.123"
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except:
        try:
            # Format without milliseconds
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except:
            return None

def calculate_duration(start_ts, end_ts):
    """Calculate duration between two timestamps in milliseconds."""
    start = parse_timestamp(start_ts)
    end = parse_timestamp(end_ts)
    if start and end:
        delta = end - start
        return delta.total_seconds() * 1000  # Convert to milliseconds
    return None

def generate_markdown_report(input_csv, output_md):
    """Generate markdown report from CSV results."""
    
    results = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    if not results:
        print("No results found in CSV file!")
        return
    
    # Generate markdown content
    md_content = []
    md_content.append("# NGFix Motivation Test Report\n")
    md_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append("\n## Test Overview\n")
    md_content.append("\nThis test evaluates the overhead of periodic index rebuilding in NGFix:\n")
    md_content.append("1. Delete 20% of addition edges (edges added through NGFix)\n")
    md_content.append("2. Test with 10K test queries\n")
    md_content.append("3. Rebuild index using 10M train queries\n")
    md_content.append("4. Test at every 10% completion of rebuild process\n\n")
    
    # Extract unique stages
    stages = []
    for row in results:
        stage = row['stage']
        if stage not in [s['name'] for s in stages]:
            stages.append({'name': stage, 'rows': []})
    
    for row in results:
        stage_name = row['stage']
        for s in stages:
            if s['name'] == stage_name:
                s['rows'].append(row)
                break
    
    # Summary table
    md_content.append("## Summary\n\n")
    md_content.append("| Stage | efs | Avg Recall | Avg NDC | Avg Latency (ms) | Rel Dist Error | Test Duration (ms) |\n")
    md_content.append("|-------|-----|------------|---------|------------------|----------------|---------------------|\n")
    
    for stage in stages:
        if not stage['rows']:
            continue
        # Use the first row for this stage (assuming same efs for all rows in a stage)
        row = stage['rows'][0]
        duration = calculate_duration(row['start_timestamp'], row['end_timestamp'])
        duration_str = f"{duration:.2f}" if duration else "N/A"
        
        md_content.append(f"| {row['stage']} | {row['efs']} | {float(row['recall']):.6f} | "
                         f"{float(row['ndc']):.2f} | {float(row['latency_ms']):.4f} | "
                         f"{float(row['rderr']):.6f} | {duration_str} |\n")
    
    # Detailed results by stage
    md_content.append("\n## Detailed Results\n\n")
    
    for stage in stages:
        if not stage['rows']:
            continue
        stage_name = stage['name']
        md_content.append(f"### {stage_name}\n\n")
        
        # Get timestamps from first row
        first_row = stage['rows'][0]
        start_ts = first_row['start_timestamp']
        end_ts = first_row['end_timestamp']
        duration = calculate_duration(start_ts, end_ts)
        
        md_content.append(f"- **Start Time**: {start_ts}\n")
        md_content.append(f"- **End Time**: {end_ts}\n")
        if duration:
            md_content.append(f"- **Test Duration**: {duration:.2f} ms\n")
        md_content.append("\n")
        
        md_content.append("| Metric | Value |\n")
        md_content.append("|--------|-------|\n")
        for row in stage['rows']:
            md_content.append(f"| efs | {row['efs']} |\n")
            md_content.append(f"| Avg Recall | {float(row['recall']):.6f} |\n")
            md_content.append(f"| Avg NDC | {float(row['ndc']):.2f} |\n")
            md_content.append(f"| Avg Latency (ms) | {float(row['latency_ms']):.4f} |\n")
            md_content.append(f"| Rel Dist Error | {float(row['rderr']):.6f} |\n")
            md_content.append("\n")
    
    # Rebuild progress analysis
    md_content.append("\n## Rebuild Progress Analysis\n\n")
    md_content.append("The following table shows how metrics change as the rebuild progresses:\n\n")
    
    rebuild_stages = [s for s in stages if s['name'].startswith('rebuild_')]
    if rebuild_stages:
        md_content.append("| Rebuild Progress | Avg Recall | Avg NDC | Avg Latency (ms) | Rel Dist Error |\n")
        md_content.append("|------------------|------------|---------|------------------|----------------|\n")
        
        # Sort by percentage
        def get_percentage(stage_name):
            if 'after_deletion' in stage_name:
                return -1
            try:
                return int(stage_name.split('_')[1].replace('pct', ''))
            except:
                return 999
        
        sorted_stages = sorted(rebuild_stages, key=lambda s: get_percentage(s['name']))
        
        # Add after_deletion stage if exists
        after_deletion = [s for s in stages if 'after_deletion' in s['name']]
        if after_deletion:
            sorted_stages = after_deletion + sorted_stages
        
        for stage in sorted_stages:
            if not stage['rows']:
                continue
            row = stage['rows'][0]
            progress = row['stage'].replace('rebuild_', '').replace('after_deletion_20pct', 'After 20% Deletion')
            
            md_content.append(f"| {progress} | {float(row['recall']):.6f} | "
                             f"{float(row['ndc']):.2f} | {float(row['latency_ms']):.4f} | "
                             f"{float(row['rderr']):.6f} |\n")
    
    # Write to file
    with open(output_md, 'w') as f:
        f.writelines(md_content)
    
    print(f"Markdown report generated: {output_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate markdown report from motivation test results')
    parser.add_argument('--input', required=True, help='Input CSV file with test results')
    parser.add_argument('--output', required=True, help='Output markdown file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    generate_markdown_report(args.input, args.output)

