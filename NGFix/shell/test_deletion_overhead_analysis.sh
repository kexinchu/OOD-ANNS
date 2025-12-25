#!/bin/bash

MEX=48
M=16
efC=500
efC_AKNN=1500
efC_delete=500
efs=100
K=100
NUM_QUERIES=1000  # Use 1000 queries for detailed analysis

# Data paths
BASE_DATA_DIR="/workspace/RoarGraph/data/t2i-10M"
INDEX_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M"
RESULT_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/lazy_vs_real_deletion_results"

# Create result directory if it doesn't exist
mkdir -p ${RESULT_DIR}

# Input index
INPUT_INDEX="${INDEX_DIR}/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}.index"

# Test data (10K queries)
TEST_QUERY_PATH="${BASE_DATA_DIR}/query.10k.fbin"
TEST_GT_PATH="${BASE_DATA_DIR}/groundtruth-computed.10k.ibin"

# Output result file
RESULT_CSV="${RESULT_DIR}/deletion_overhead_analysis.csv"
LOG_FILE="${RESULT_DIR}/overhead_analysis.log"

echo "=== Deletion Overhead Analysis ===" | tee -a ${LOG_FILE}
echo "Input Index: ${INPUT_INDEX}" | tee -a ${LOG_FILE}
echo "Test Query: ${TEST_QUERY_PATH}" | tee -a ${LOG_FILE}
echo "Test GT: ${TEST_GT_PATH}" | tee -a ${LOG_FILE}
echo "Result CSV: ${RESULT_CSV}" | tee -a ${LOG_FILE}
echo "Number of queries: ${NUM_QUERIES}" | tee -a ${LOG_FILE}
echo "Timestamp: $(date)" | tee -a ${LOG_FILE}
echo ""

# Check if files exist
if [ ! -f "${INPUT_INDEX}" ]; then
    echo "Error: Input index not found: ${INPUT_INDEX}" | tee -a ${LOG_FILE}
    exit 1
fi

if [ ! -f "${TEST_QUERY_PATH}" ]; then
    echo "Error: Test query file not found: ${TEST_QUERY_PATH}" | tee -a ${LOG_FILE}
    exit 1
fi

if [ ! -f "${TEST_GT_PATH}" ]; then
    echo "Error: Test GT file not found: ${TEST_GT_PATH}" | tee -a ${LOG_FILE}
    exit 1
fi

# Run the overhead analysis
echo "Starting overhead analysis..." | tee -a ${LOG_FILE}
cd /workspace/OOD-ANNS/NGFix
./build/test/test_deletion_overhead_analysis \
    --test_query_path ${TEST_QUERY_PATH} \
    --test_gt_path ${TEST_GT_PATH} \
    --index_path ${INPUT_INDEX} \
    --metric ip_float \
    --K ${K} \
    --efs ${efs} \
    --efC_AKNN ${efC_AKNN} \
    --efC_delete ${efC_delete} \
    --num_queries ${NUM_QUERIES} \
    --result_path ${RESULT_CSV} 2>&1 | tee -a ${LOG_FILE}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "" | tee -a ${LOG_FILE}
    echo "=== Overhead Analysis completed successfully ===" | tee -a ${LOG_FILE}
    echo "Results saved to: ${RESULT_CSV}" | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
    echo "Generating summary report..." | tee -a ${LOG_FILE}
    
    # Generate summary using Python
    python3 << 'PYTHON_SCRIPT'
import pandas as pd
import sys
import os

result_file = "/workspace/OOD-ANNS/NGFix/data/t2i-10M/lazy_vs_real_deletion_results/deletion_overhead_analysis.csv"
summary_file = "/workspace/OOD-ANNS/NGFix/data/t2i-10M/lazy_vs_real_deletion_results/overhead_summary.md"

if not os.path.exists(result_file):
    print(f"Result file not found: {result_file}")
    sys.exit(1)

df = pd.read_csv(result_file)

# Group by deletion_type and deletion_percentage
summary = df.groupby(['deletion_type', 'deletion_percentage']).agg({
    'total_time_us': 'mean',
    'distance_time_us': 'mean',
    'is_deleted_time_us': 'mean',
    'lock_time_us': 'mean',
    'queue_time_us': 'mean',
    'neighbor_time_us': 'mean',
    'visited_time_us': 'mean',
    'other_time_us': 'mean',
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

with open(summary_file, 'w') as f:
    f.write("# Deletion Overhead Analysis Summary\n\n")
    
    for del_pct in sorted(df['deletion_percentage'].unique()):
        f.write(f"## {del_pct}% Deletion\n\n")
        
        lazy_data = summary[(summary['deletion_percentage'] == del_pct) & 
                           (summary['deletion_type'] == 'lazy_deletion')].iloc[0]
        real_data = summary[(summary['deletion_percentage'] == del_pct) & 
                          (summary['deletion_type'] == 'real_deletion')].iloc[0]
        
        f.write("### Time Breakdown (microseconds)\n\n")
        f.write("| Component | Lazy Deletion | Real Deletion | Difference |\n")
        f.write("|-----------|---------------|---------------|------------|\n")
        
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
            lazy_val = lazy_data[col]
            real_val = real_data[col]
            diff = real_val - lazy_val
            f.write(f"| {name} | {lazy_val:.2f} | {real_val:.2f} | {diff:+.2f} |\n")
        
        f.write("\n### Time Percentage Breakdown\n\n")
        f.write("| Component | Lazy Deletion % | Real Deletion % |\n")
        f.write("|-----------|-----------------|------------------|\n")
        
        lazy_total = lazy_data['total_time_us']
        real_total = real_data['total_time_us']
        
        for name, col in components[1:]:  # Skip total
            lazy_pct = (lazy_data[col] / lazy_total * 100) if lazy_total > 0 else 0
            real_pct = (real_data[col] / real_total * 100) if real_total > 0 else 0
            f.write(f"| {name} | {lazy_pct:.2f}% | {real_pct:.2f}% |\n")
        
        f.write("\n### Operation Counts\n\n")
        f.write("| Operation | Lazy Deletion | Real Deletion |\n")
        f.write("|-----------|---------------|---------------|\n")
        
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
            lazy_val = lazy_data[col]
            real_val = real_data[col]
            f.write(f"| {name} | {lazy_val:.1f} | {real_val:.1f} |\n")
        
        f.write("\n### Performance Metrics\n\n")
        f.write("| Metric | Lazy Deletion | Real Deletion |\n")
        f.write("|--------|---------------|---------------|\n")
        f.write(f"| Recall | {lazy_data['recall']:.6f} | {real_data['recall']:.6f} |\n")
        f.write(f"| NDC | {lazy_data['ndc']:.2f} | {real_data['ndc']:.2f} |\n")
        f.write(f"| Rderr | {lazy_data['rderr']:.6f} | {real_data['rderr']:.6f} |\n")
        f.write("\n")

print(f"Summary report generated: {summary_file}")
PYTHON_SCRIPT

    echo "Summary report generated" | tee -a ${LOG_FILE}
    exit 0
else
    echo "" | tee -a ${LOG_FILE}
    echo "=== Overhead Analysis failed ===" | tee -a ${LOG_FILE}
    exit 1
fi
