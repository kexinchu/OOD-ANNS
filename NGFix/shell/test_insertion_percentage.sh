#!/bin/bash
# Insertion Test Script
# Tests insertion at different percentages: 1%, 2%, 3%, 4%, 5%, 10%, 20%
# Starting from 8M base, inserting from the remaining 2M data

set -e  # Exit on error

MEX=48
M=16
efC=500
efC_AKNN=1500
K=100
efs=1000
NOISE_SCALE=0.01  # Add random bias to inserted data (0.01 = 1% noise)

BASE_SIZE=8000000
ADDITIONAL_SIZE=2000000  # 2M additional data available
BASE_DATA_SIZE=8000000   # Base data size for percentage calculation

# Ensure the base index exists (built on 8M data)
BASE_INDEX_PATH=/workspace/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_8M.index

if [ ! -f "$BASE_INDEX_PATH" ]; then
    echo "Base index not found. Building base index first..."
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
    --result_index_path ${BASE_INDEX_PATH}
fi

# Extract additional 2M data (from 8M to 10M) if not exists
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
    # Skip to start index
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

# Create results directory (with efs and noise suffix)
RESULTS_DIR=/workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs${efs}_noise${NOISE_SCALE}
mkdir -p ${RESULTS_DIR}

# Initialize summary file
SUMMARY_FILE=${RESULTS_DIR}/summary.csv
echo "Percentage,InsertCount,InsertionLatency_ms,Recall,NDC,SearchLatency_ms" > ${SUMMARY_FILE}

# Test percentages: 1%, 2%, 3%, 4%, 5%, 10%, 15%, 20%, 25%
# These percentages are based on BASE_DATA_SIZE (8M)
# 1% = 0.08M, 2% = 0.16M, 3% = 0.24M, 4% = 0.32M, 5% = 0.4M
# 10% = 0.8M, 15% = 1.2M, 20% = 1.6M, 25% = 2M
PERCENTAGES=(1 2 3 4 5 10 15 20 25)

echo "Starting insertion percentage tests..."
echo "Base index: ${BASE_INDEX_PATH}"
echo "Additional data: ${ADDITIONAL_DATA_PATH}"
echo "=========================================="

for PERCENT in "${PERCENTAGES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing insertion of ${PERCENT}% additional data"
    echo "=========================================="
    
    # Calculate insertion count (percentage of 8M base data)
    INSERT_COUNT=$((BASE_DATA_SIZE * PERCENT / 100))
    INSERT_START_ID=$BASE_SIZE
    INSERT_END_ID=$((BASE_SIZE + INSERT_COUNT))
    TOTAL_SIZE=$((BASE_SIZE + INSERT_COUNT))
    
    # Check if we have enough additional data
    if [ ${INSERT_COUNT} -gt ${ADDITIONAL_SIZE} ]; then
        echo "Warning: Requested ${INSERT_COUNT} vectors but only ${ADDITIONAL_SIZE} available. Skipping ${PERCENT}% test."
        continue
    fi
    
    echo "Insert count: ${INSERT_COUNT} (${PERCENT}% of ${BASE_DATA_SIZE}M base data)"
    echo "Insert range: ID ${INSERT_START_ID} to ${INSERT_END_ID}"
    echo "Total size after insertion: ${TOTAL_SIZE}"
    
    # Create index path for this percentage
    INDEX_PATH=${RESULTS_DIR}/index_insert_${PERCENT}percent.index
    SEARCH_RESULT_PATH=${RESULTS_DIR}/search_results_insert_${PERCENT}percent.csv
    INSERTION_LOG=${RESULTS_DIR}/insertion_log_${PERCENT}percent.txt
    
    # Use the new insertion_percentage program which can precisely control insertion count
    # It uses the full 10M data file and filters ground truth to avoid out-of-bounds errors
    FULL_DATA_PATH=/workspace/RoarGraph/data/t2i-10M/base.10M.fbin
    
    # Perform insertion using the new program (no training/rebuild)
    echo "Performing insertion..."
    ./test/test_hnsw_ngfix_insertion_percentage \
    --base_data_path ${FULL_DATA_PATH} \
    --raw_index_path ${BASE_INDEX_PATH} \
    --metric ip_float \
    --result_index_path ${INDEX_PATH} \
    --efC ${efC} \
    --insert_st_id ${INSERT_START_ID} \
    --insert_count ${INSERT_COUNT} \
    --noise_scale ${NOISE_SCALE} > ${INSERTION_LOG} 2>&1
    
    # Extract insertion latency from log
    INSERTION_LATENCY=$(grep "Insertion latency:" ${INSERTION_LOG} | grep -oP '\d+' | head -1)
    echo "Insertion latency: ${INSERTION_LATENCY} ms"
    
    # Perform search to test recall and latency
    # Use filtered search to avoid accessing out-of-bounds IDs
    echo "Performing search test..."
    taskset -c 0,1 ./test/search_hnsw_ngfix_filtered \
    --test_query_path /workspace/RoarGraph/data/t2i-10M/query.10k.fbin \
    --test_gt_path /workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin \
    --metric ip_float --K ${K} --result_path ${SEARCH_RESULT_PATH} \
    --index_path ${INDEX_PATH} \
    --max_valid_id ${TOTAL_SIZE} > ${RESULTS_DIR}/search_log_${PERCENT}percent.txt 2>&1
    
    # Wait a moment to ensure file is written
    sleep 1
    
    # Extract recall, latency, and NDC from search results (efs=100 row)
    if [ -f "${SEARCH_RESULT_PATH}" ] && [ -s "${SEARCH_RESULT_PATH}" ]; then
        RECALL=$(grep "^${efs}," ${SEARCH_RESULT_PATH} | cut -d',' -f2 | tr -d ' ')
        NDC=$(grep "^${efs}," ${SEARCH_RESULT_PATH} | cut -d',' -f3 | tr -d ' ')
        LATENCY=$(grep "^${efs}," ${SEARCH_RESULT_PATH} | cut -d',' -f4 | tr -d ' ')
    else
        echo "Warning: Search result file empty or missing, retrying..."
        sleep 2
        RECALL=$(grep "^${efs}," ${SEARCH_RESULT_PATH} 2>/dev/null | cut -d',' -f2 | tr -d ' ' || echo "0")
        NDC=$(grep "^${efs}," ${SEARCH_RESULT_PATH} 2>/dev/null | cut -d',' -f3 | tr -d ' ' || echo "0")
        LATENCY=$(grep "^${efs}," ${SEARCH_RESULT_PATH} 2>/dev/null | cut -d',' -f4 | tr -d ' ' || echo "0")
    fi
    
    echo "Search results (efs=${efs}):"
    echo "  Recall: ${RECALL}"
    echo "  NDC: ${NDC}"
    echo "  Latency: ${LATENCY} ms"
    
    # Save summary (including NDC)
    echo "${PERCENT},${INSERT_COUNT},${INSERTION_LATENCY},${RECALL},${NDC},${LATENCY}" >> ${SUMMARY_FILE}
    
    echo "Completed ${PERCENT}% insertion test"
done

echo ""
echo "=========================================="
echo "All insertion tests completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "Summary: ${RESULTS_DIR}/summary.csv"
echo "=========================================="
cat ${RESULTS_DIR}/summary.csv

