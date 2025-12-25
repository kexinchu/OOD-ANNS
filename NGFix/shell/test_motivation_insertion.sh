#!/bin/bash
# Motivation Test Script
# Tests insertion of different amounts of data from additional.10M into 10M NGFix index
# Tests: 0.1M, 0.2M, 0.3M, 0.4M, 0.5M, 1M, 1.5M, 2M
# Tests 10k query at start and after each insertion

set -e  # Exit on error

# Configuration
M=16
efC=500
MEX=48
efC_AKNN=1500
K=100
efs=1000  # Use efs=1000 for testing

# Base index (10M NGFix index)
BASE_INDEX_PATH=/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_10M.index

# Additional data source
ADDITIONAL_DATA_PATH=/workspace/RoarGraph/data/t2i-10M/base.additional.10M.fbin

# Query and ground truth paths
QUERY_PATH=/workspace/RoarGraph/data/t2i-10M/query.10k.fbin
GT_PATH=/workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin

# Check if base index exists
if [ ! -f "$BASE_INDEX_PATH" ]; then
    echo "Error: Base index not found at ${BASE_INDEX_PATH}"
    exit 1
fi

# Check if additional data exists
if [ ! -f "$ADDITIONAL_DATA_PATH" ]; then
    echo "Error: Additional data not found at ${ADDITIONAL_DATA_PATH}"
    exit 1
fi

# Create results directory
RESULTS_DIR=/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_insertion_results
mkdir -p ${RESULTS_DIR}

# Initialize summary file
SUMMARY_FILE=${RESULTS_DIR}/summary.csv
echo "InsertCount,InsertionLatency_ms,Recall,NDC,SearchLatency_ms" > ${SUMMARY_FILE}

# Test insertion amounts (in millions)
INSERT_AMOUNTS=(0.1 0.2 0.3 0.4 0.5 1.0 1.5 2.0)

# Base size is 10M
BASE_SIZE=10000000

echo "=========================================="
echo "Motivation Insertion Test"
echo "=========================================="
echo "Base index: ${BASE_INDEX_PATH}"
echo "Additional data: ${ADDITIONAL_DATA_PATH}"
echo "Base size: ${BASE_SIZE}"
echo "=========================================="

# Test initial state (before any insertion)
echo ""
echo "=========================================="
echo "Testing initial state (10M index)"
echo "=========================================="

INITIAL_SEARCH_RESULT=${RESULTS_DIR}/search_initial.csv
echo "Performing initial search test..."
cd /workspace/OOD-ANNS/NGFix/build
taskset -c 0,1 ./test/search_hnsw_ngfix_filtered \
    --test_query_path ${QUERY_PATH} \
    --test_gt_path ${GT_PATH} \
    --metric ip_float --K ${K} --result_path ${INITIAL_SEARCH_RESULT} \
    --index_path ${BASE_INDEX_PATH} \
    --max_valid_id ${BASE_SIZE} > ${RESULTS_DIR}/search_log_initial.txt 2>&1 &

SEARCH_PID=$!

# Wait for search to complete
echo "Waiting for initial search to complete..."
MAX_WAIT=300
WAIT_COUNT=0
while [ ${WAIT_COUNT} -lt ${MAX_WAIT} ]; do
    if ! kill -0 ${SEARCH_PID} 2>/dev/null; then
        # Process finished
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo "  Still waiting... (${WAIT_COUNT}s)"
    fi
done

# Wait a bit more for file to be written
sleep 2

# Extract initial metrics (wait a bit more for file to be fully written)
sleep 2

# Retry extraction with multiple attempts
for attempt in 1 2 3; do
    if [ -f "${INITIAL_SEARCH_RESULT}" ] && [ -s "${INITIAL_SEARCH_RESULT}" ]; then
        if grep -q "^${efs}," ${INITIAL_SEARCH_RESULT}; then
            RECALL=$(grep "^${efs}," ${INITIAL_SEARCH_RESULT} | cut -d',' -f2 | tr -d ' ')
            NDC=$(grep "^${efs}," ${INITIAL_SEARCH_RESULT} | cut -d',' -f3 | tr -d ' ')
            LATENCY=$(grep "^${efs}," ${INITIAL_SEARCH_RESULT} | cut -d',' -f4 | tr -d ' ')
            
            if [ -n "$RECALL" ] && [ -n "$NDC" ] && [ -n "$LATENCY" ]; then
                break
            fi
        fi
    fi
    if [ ${attempt} -lt 3 ]; then
        sleep 2
    fi
done

if [ -z "$RECALL" ] || [ -z "$NDC" ] || [ -z "$LATENCY" ]; then
    echo "Warning: Could not extract all initial metrics, using defaults"
    RECALL=${RECALL:-"0"}
    NDC=${NDC:-"0"}
    LATENCY=${LATENCY:-"0"}
fi

echo "Initial results (efs=${efs}):"
echo "  Recall: ${RECALL}"
echo "  NDC: ${NDC}"
echo "  Latency: ${LATENCY} ms"

# Save initial state
echo "0,0,${RECALL},${NDC},${LATENCY}" >> ${SUMMARY_FILE}

# Current index path (starts as base index, will be updated after each insertion)
CURRENT_INDEX_PATH=${BASE_INDEX_PATH}
CURRENT_SIZE=${BASE_SIZE}
CURRENT_DATA_OFFSET=0  # Track where we are in additional.10M.fbin

# Test each insertion amount
for AMOUNT in "${INSERT_AMOUNTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing insertion of ${AMOUNT}M vectors"
    echo "=========================================="
    
    # Calculate insertion count for this amount
    # Each step should insert incremental amount to reach the target total
    # 0.1M: insert 0.1M (total 10.1M)
    # 0.2M: insert additional 0.1M (total 10.2M, incremental from 0.1M)
    # 0.3M: insert additional 0.1M (total 10.3M, incremental from 0.2M)
    # etc.
    if [ "$AMOUNT" == "0.1" ]; then
        STEP_INSERT_COUNT=100000
    elif [ "$AMOUNT" == "0.2" ]; then
        STEP_INSERT_COUNT=100000  # Additional 0.1M to reach 0.2M total
    elif [ "$AMOUNT" == "0.3" ]; then
        STEP_INSERT_COUNT=100000  # Additional 0.1M to reach 0.3M total
    elif [ "$AMOUNT" == "0.4" ]; then
        STEP_INSERT_COUNT=100000  # Additional 0.1M to reach 0.4M total
    elif [ "$AMOUNT" == "0.5" ]; then
        STEP_INSERT_COUNT=100000  # Additional 0.1M to reach 0.5M total
    elif [ "$AMOUNT" == "1.0" ]; then
        STEP_INSERT_COUNT=500000  # Additional 0.5M to reach 1.0M total (from 0.5M)
    elif [ "$AMOUNT" == "1.5" ]; then
        STEP_INSERT_COUNT=500000  # Additional 0.5M to reach 1.5M total (from 1.0M)
    elif [ "$AMOUNT" == "2.0" ]; then
        STEP_INSERT_COUNT=500000  # Additional 0.5M to reach 2.0M total (from 1.5M)
    else
        echo "Error: Unknown amount ${AMOUNT}"
        continue
    fi
    
    INSERT_COUNT=${STEP_INSERT_COUNT}
    
    INSERT_START_ID=${CURRENT_SIZE}
    INSERT_END_ID=$((CURRENT_SIZE + INSERT_COUNT))
    TOTAL_SIZE=${INSERT_END_ID}
    
    echo "Insert count in this step: ${INSERT_COUNT} (incremental)"
    echo "Total inserted so far: $((TOTAL_SIZE - BASE_SIZE)) (target: ${AMOUNT}M)"
    echo "Insert range: ID ${INSERT_START_ID} to ${INSERT_END_ID}"
    echo "Total size after insertion: ${TOTAL_SIZE}"
    
    # Create index path for this insertion
    INDEX_NAME=$(echo "${AMOUNT}" | sed 's/\./_/')
    INDEX_PATH=${RESULTS_DIR}/index_insert_${INDEX_NAME}M.index
    SEARCH_RESULT_PATH=${RESULTS_DIR}/search_results_insert_${INDEX_NAME}M.csv
    INSERTION_LOG=${RESULTS_DIR}/insertion_log_${INDEX_NAME}M.txt
    
    # Perform insertion (no training/rebuild, just regular insert)
    # data_offset: read from CURRENT_DATA_OFFSET in additional.10M.fbin
    # insert_id_start: insert at index ID starting from CURRENT_SIZE
    echo "Performing insertion..."
    cd /workspace/OOD-ANNS/NGFix/build
    ./test/test_hnsw_ngfix_insertion_with_offset \
        --base_data_path ${ADDITIONAL_DATA_PATH} \
        --raw_index_path ${CURRENT_INDEX_PATH} \
        --metric ip_float \
        --result_index_path ${INDEX_PATH} \
        --efC ${efC} \
        --data_offset ${CURRENT_DATA_OFFSET} \
        --insert_id_start ${INSERT_START_ID} \
        --insert_count ${INSERT_COUNT} \
        --noise_scale 0.0 > ${INSERTION_LOG} 2>&1
    
    # Extract insertion latency from log
    INSERTION_LATENCY=$(grep "Insertion latency:" ${INSERTION_LOG} | grep -oP '\d+' | head -1)
    if [ -z "$INSERTION_LATENCY" ]; then
        INSERTION_LATENCY="0"
    fi
    echo "Insertion latency: ${INSERTION_LATENCY} ms"
    
    # Perform search to test recall and latency
    echo "Performing search test..."
    taskset -c 0,1 ./test/search_hnsw_ngfix_filtered \
        --test_query_path ${QUERY_PATH} \
        --test_gt_path ${GT_PATH} \
        --metric ip_float --K ${K} --result_path ${SEARCH_RESULT_PATH} \
        --index_path ${INDEX_PATH} \
        --max_valid_id ${TOTAL_SIZE} > ${RESULTS_DIR}/search_log_${INDEX_NAME}M.txt 2>&1
    
    # Wait for search to complete (check if file has all efs values)
    echo "Waiting for search to complete..."
    MAX_WAIT=300  # Maximum wait time in seconds
    WAIT_COUNT=0
    while [ ${WAIT_COUNT} -lt ${MAX_WAIT} ]; do
        if [ -f "${SEARCH_RESULT_PATH}" ] && [ -s "${SEARCH_RESULT_PATH}" ]; then
            # Check if file contains the target efs value
            if grep -q "^${efs}," ${SEARCH_RESULT_PATH}; then
                break
            fi
        fi
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 2))
        if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
            echo "  Still waiting... (${WAIT_COUNT}s)"
        fi
    done
    
    # Extract recall, latency, and NDC from search results (efs=1000 row)
    if [ -f "${SEARCH_RESULT_PATH}" ] && [ -s "${SEARCH_RESULT_PATH}" ]; then
        RECALL=$(grep "^${efs}," ${SEARCH_RESULT_PATH} | cut -d',' -f2 | tr -d ' ')
        NDC=$(grep "^${efs}," ${SEARCH_RESULT_PATH} | cut -d',' -f3 | tr -d ' ')
        LATENCY=$(grep "^${efs}," ${SEARCH_RESULT_PATH} | cut -d',' -f4 | tr -d ' ')
        
        if [ -z "$RECALL" ] || [ -z "$NDC" ] || [ -z "$LATENCY" ]; then
            echo "Warning: Could not extract all metrics, using defaults"
            RECALL=${RECALL:-"0"}
            NDC=${NDC:-"0"}
            LATENCY=${LATENCY:-"0"}
        fi
    else
        echo "Warning: Search result file empty or missing"
        RECALL="0"
        NDC="0"
        LATENCY="0"
    fi
    
    echo "Search results (efs=${efs}):"
    echo "  Recall: ${RECALL}"
    echo "  NDC: ${NDC}"
    echo "  Latency: ${LATENCY} ms"
    
    # Save summary
    echo "${INSERT_COUNT},${INSERTION_LATENCY},${RECALL},${NDC},${LATENCY}" >> ${SUMMARY_FILE}
    
    # Update current index path, size, and data offset for next iteration
    CURRENT_INDEX_PATH=${INDEX_PATH}
    CURRENT_SIZE=${TOTAL_SIZE}
    CURRENT_DATA_OFFSET=$((CURRENT_DATA_OFFSET + INSERT_COUNT))
    
    echo "Completed ${AMOUNT}M insertion test (total size: ${CURRENT_SIZE} vectors, data offset: ${CURRENT_DATA_OFFSET})"
done

echo ""
echo "=========================================="
echo "All insertion tests completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "Summary: ${SUMMARY_FILE}"
echo "=========================================="
cat ${SUMMARY_FILE}

