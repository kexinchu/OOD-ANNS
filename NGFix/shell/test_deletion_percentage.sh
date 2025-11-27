#!/bin/bash
# Deletion Test Script
# Tests deletion at different percentages: 1%, 2%, 3%, 4%, 5%, 10%, 20%
# Starting from 10M, testing lazy delete and total delete latency

set -e  # Exit on error

MEX=48
M=16
efC=500
efC_AKNN=1500
K=100
efs=100

BASE_SIZE=8000000
ADDITIONAL_SIZE=2000000
TOTAL_SIZE=10000000
BASE_DATA_SIZE=8000000  # Base data size for percentage calculation

# Ensure the 10M index exists (if not, build it first from 8M + 2M insertion)
BASE_10M_INDEX_PATH=/workspace/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_10M.index
BASE_8M_INDEX_PATH=/workspace/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_8M.index

if [ ! -f "$BASE_10M_INDEX_PATH" ]; then
    echo "10M index not found. Building from 8M index..."
    
    if [ ! -f "$BASE_8M_INDEX_PATH" ]; then
        echo "8M base index not found. Building base index first..."
        # Build bottom layer with 8M data
        BASE_8M_INDEX=/workspace/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}_8M.index
        ./test/build_hnsw_bottom --base_data_path /workspace/RoarGraph/data/t2i-10M/base.8M.fbin \
        --metric ip_float \
        --result_hnsw_index_path ${BASE_8M_INDEX} \
        --M ${M} --MEX ${MEX} --efC ${efC}
        
        # Build NGFix index
        ./test/build_hnsw_ngfix_aknn \
        --train_query_path /workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin \
        --train_gt_path /workspace/RoarGraph/data/t2i-10M/train.gt.bin \
        --base_graph_path ${BASE_8M_INDEX} \
        --metric ip_float --efC_AKNN ${efC_AKNN} \
        --result_index_path ${BASE_8M_INDEX_PATH}
    fi
    
    # Extract additional 2M data
    ADDITIONAL_DATA_PATH=/workspace/NGFix/data/t2i-10M/base.additional.2M.fbin
    if [ ! -f "$ADDITIONAL_DATA_PATH" ]; then
        echo "Extracting additional 2M data..."
        python3 << 'PYTHON_SCRIPT'
import struct
SRC = '/workspace/RoarGraph/data/t2i-10M/base.10M.fbin'
DST = '/workspace/NGFix/data/t2i-10M/base.additional.2M.fbin'
START_IDX = 8000000
COUNT = 2000000

with open(SRC, 'rb') as fin:
    header = fin.read(8)
    num, dim = struct.unpack('<ii', header)
    vec_bytes = dim * 4
    fin.seek(8 + START_IDX * vec_bytes)
    with open(DST, 'wb') as fout:
        fout.write(struct.pack('<ii', COUNT, dim))
        remaining = COUNT * vec_bytes
        bufsize = 64 * 1024 * 1024
        while remaining > 0:
            chunk = fin.read(min(bufsize, remaining))
            if not chunk:
                break
            fout.write(chunk)
            remaining -= len(chunk)
print(f'Extracted {COUNT} vectors to {DST}')
PYTHON_SCRIPT
    fi
    
    # Insert 2M to make 10M index
    echo "Inserting 2M data to create 10M index..."
    FULL_DATA_PATH=/workspace/RoarGraph/data/t2i-10M/base.10M.fbin
    ./test/test_hnsw_ngfix_insertion_percentage \
    --base_data_path ${FULL_DATA_PATH} \
    --train_query_path /workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin \
    --train_gt_path /workspace/RoarGraph/data/t2i-10M/train.gt.bin \
    --raw_index_path ${BASE_8M_INDEX_PATH} \
    --metric ip_float \
    --result_index_path ${BASE_10M_INDEX_PATH} \
    --efC ${efC} \
    --insert_st_id ${BASE_SIZE} \
    --insert_count ${ADDITIONAL_SIZE}
fi

# Create results directory
RESULTS_DIR=/workspace/NGFix/data/t2i-10M/deletion_percentage_results
mkdir -p ${RESULTS_DIR}

# Test percentages: 1%, 2%, 3%, 4%, 5%, 10%, 15%, 20%, 25%
# These percentages are based on BASE_DATA_SIZE (8M)
# 1% = 0.08M, 2% = 0.16M, 3% = 0.24M, 4% = 0.32M, 5% = 0.4M
# 10% = 0.8M, 15% = 1.2M, 20% = 1.6M, 25% = 2M
PERCENTAGES=(1 2 3 4 5 10 15 20 25)

echo "Starting deletion percentage tests..."
echo "Base 10M index: ${BASE_10M_INDEX_PATH}"
echo "=========================================="

# Initialize summary file
SUMMARY_FILE=${RESULTS_DIR}/summary.csv
echo "Percentage,DeleteCount,LazyDeleteLatency_ms,TotalDeleteLatency_ms" > ${SUMMARY_FILE}

for PERCENT in "${PERCENTAGES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing deletion of ${PERCENT}% data"
    echo "=========================================="
    
    # Calculate deletion count (percentage of 8M base data)
    DELETE_COUNT=$((BASE_DATA_SIZE * PERCENT / 100))
    # Delete from the end of the 10M index (the inserted 2M part)
    DELETE_START_ID=$((TOTAL_SIZE - DELETE_COUNT))
    DELETE_END_ID=${TOTAL_SIZE}
    
    # Ensure we don't delete beyond available data
    if [ ${DELETE_COUNT} -gt ${ADDITIONAL_SIZE} ]; then
        echo "Warning: Requested ${DELETE_COUNT} deletions but only ${ADDITIONAL_SIZE} inserted data available. Skipping ${PERCENT}% test."
        continue
    fi
    
    echo "Delete count: ${DELETE_COUNT} (${PERCENT}% of ${BASE_DATA_SIZE}M base data)"
    echo "Delete range: ID ${DELETE_START_ID} to ${DELETE_END_ID}"
    
    # Create index path for this percentage (copy from base first)
    TEST_INDEX_PATH=${RESULTS_DIR}/index_before_delete_${PERCENT}percent.index
    LAZY_DELETE_INDEX_PATH=${RESULTS_DIR}/index_lazy_delete_${PERCENT}percent.index
    TOTAL_DELETE_INDEX_PATH=${RESULTS_DIR}/index_total_delete_${PERCENT}percent.index
    DELETION_LOG=${RESULTS_DIR}/deletion_log_${PERCENT}percent.txt
    
    # Copy base index for testing
    echo "Copying base index for test..."
    cp ${BASE_10M_INDEX_PATH} ${TEST_INDEX_PATH}
    
    # Run deletion test program
    echo "Running deletion test..."
    ./test/test_hnsw_ngfix_deletion_percentage \
    --index_path ${TEST_INDEX_PATH} \
    --metric ip_float \
    --delete_start_id ${DELETE_START_ID} \
    --delete_end_id ${DELETE_END_ID} \
    --lazy_delete_index_path ${LAZY_DELETE_INDEX_PATH} \
    --total_delete_index_path ${TOTAL_DELETE_INDEX_PATH} \
    --result_log ${DELETION_LOG} \
    --efC ${efC}
    
    # Extract latencies from log
    LAZY_DELETE_LATENCY=$(grep "Lazy delete latency:" ${DELETION_LOG} | grep -oP '\d+' | head -1)
    TOTAL_DELETE_LATENCY=$(grep "Total delete latency:" ${DELETION_LOG} | grep -oP '\d+' | head -1)
    
    echo "Lazy delete latency: ${LAZY_DELETE_LATENCY} ms"
    echo "Total delete latency: ${TOTAL_DELETE_LATENCY} ms"
    
    # Save summary
    echo "${PERCENT},${DELETE_COUNT},${LAZY_DELETE_LATENCY},${TOTAL_DELETE_LATENCY}" >> ${SUMMARY_FILE}
    
    echo "Completed ${PERCENT}% deletion test"
done

echo ""
echo "=========================================="
echo "All deletion tests completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "Summary: ${SUMMARY_FILE}"
echo "=========================================="
cat ${SUMMARY_FILE}

