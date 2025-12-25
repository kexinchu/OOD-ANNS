#!/bin/bash

# Comprehensive Deletion Comparison Test
# This script:
# 1. Builds base index with 10M data
# 2. Inserts 3M additional data
# 3. Applies NGFix + RFix with 10M train queries
# 4. Tests initial recall/NDC/latency with efSearch = 100/200/300/400/500/1000/2000
# 5. Stores index to SSD
# 6. Tests lazy deletion
# 7. Tests real deletion
# 8. Compares all three

set -e  # Exit on error

MEX=48
M=16
efC=500
efC_AKNN=1500
efC_delete=500
K=100

# Data paths
BASE_DATA_DIR="/workspace/RoarGraph/data/t2i-10M"
INDEX_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M"
RESULT_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/comprehensive_deletion_results"

# Create result directory if it doesn't exist
mkdir -p ${RESULT_DIR}

# Data files
BASE_DATA_PATH="${BASE_DATA_DIR}/base.10M.fbin"
ADDITIONAL_DATA_PATH="${BASE_DATA_DIR}/base.additional.10M.fbin"  # We'll extract 3M from this
TRAIN_QUERY_PATH="${BASE_DATA_DIR}/query.train.10M.fbin"
TRAIN_GT_PATH="${BASE_DATA_DIR}/train.gt.bin"
TEST_QUERY_PATH="${BASE_DATA_DIR}/query.10k.fbin"
TEST_GT_PATH="${BASE_DATA_DIR}/groundtruth-computed.10k.ibin"

# Index paths
BASE_BOTTOM_INDEX="${INDEX_DIR}/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}_10M.index"
BASE_NGFIX_INDEX="${INDEX_DIR}/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_10M.index"
FINAL_INDEX="${INDEX_DIR}/t2i_10M_3M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN${efC_AKNN}_13M.index"
RESULT_CSV="${RESULT_DIR}/comprehensive_deletion_results.csv"

# Extract 3M additional data if needed
ADDITIONAL_DATA_EXTRACTED="${INDEX_DIR}/base.additional.3M.fbin"
if [ ! -f "${ADDITIONAL_DATA_EXTRACTED}" ]; then
    echo "=== Extracting 3M additional data ==="
    python3 << 'PYTHON_SCRIPT'
import struct
import sys

SRC = '/workspace/RoarGraph/data/t2i-10M/base.additional.10M.fbin'
DST = '/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.additional.3M.fbin'
START_IDX = 0
COUNT = 3000000

with open(SRC, 'rb') as fin:
    n, d = struct.unpack('ii', fin.read(8))
    print(f"Source file: {n} vectors, dimension {d}")
    
    if START_IDX + COUNT > n:
        print(f"Error: Requested {COUNT} vectors starting from {START_IDX}, but only {n} available")
        sys.exit(1)
    
    # Skip to start position
    fin.seek(8 + START_IDX * d * 4)
    
    with open(DST, 'wb') as fout:
        # Write header
        fout.write(struct.pack('ii', COUNT, d))
        # Copy data
        bytes_to_copy = COUNT * d * 4
        while bytes_to_copy > 0:
            chunk = fin.read(min(bytes_to_copy, 1024*1024))  # 1MB chunks
            if not chunk:
                break
            fout.write(chunk)
            bytes_to_copy -= len(chunk)
    
    print(f"Extracted {COUNT} vectors to {DST}")

PYTHON_SCRIPT
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract additional data"
        exit 1
    fi
    echo "Additional data extracted successfully"
fi

ADDITIONAL_DATA_PATH="${ADDITIONAL_DATA_EXTRACTED}"

echo "=== Comprehensive Deletion Comparison Test ==="
echo "Base data: ${BASE_DATA_PATH}"
echo "Additional data: ${ADDITIONAL_DATA_PATH}"
echo "Train query: ${TRAIN_QUERY_PATH}"
echo "Test query: ${TEST_QUERY_PATH}"
echo "Result CSV: ${RESULT_CSV}"
echo ""

# Check if files exist
for file in "${BASE_DATA_PATH}" "${ADDITIONAL_DATA_PATH}" "${TRAIN_QUERY_PATH}" "${TEST_QUERY_PATH}" "${TEST_GT_PATH}"; do
    if [ ! -f "${file}" ]; then
        echo "Error: Required file not found: ${file}"
        exit 1
    fi
done

cd /workspace/OOD-ANNS/NGFix

# Step 1: Build base bottom layer index (10M data)
if [ ! -f "${BASE_BOTTOM_INDEX}" ]; then
    echo "=== Step 1: Building base bottom layer index (10M data) ==="
    ./build/test/build_hnsw_bottom \
        --base_data_path ${BASE_DATA_PATH} \
        --metric ip_float \
        --result_hnsw_index_path ${BASE_BOTTOM_INDEX} \
        --M ${M} --MEX ${MEX} --efC ${efC}
    
    if [ $? -ne 0 ] || [ ! -f "${BASE_BOTTOM_INDEX}" ]; then
        echo "Error: Failed to build base bottom layer index"
        exit 1
    fi
    echo "Base bottom layer index built successfully"
else
    echo "Base bottom layer index already exists: ${BASE_BOTTOM_INDEX}"
fi

# Step 2: Build base NGFix index (10M data with NGFix + RFix)
if [ ! -f "${BASE_NGFIX_INDEX}" ]; then
    echo "=== Step 2: Building base NGFix index (10M data) ==="
    ./build/test/build_hnsw_ngfix_with_gt \
        --train_query_path ${TRAIN_QUERY_PATH} \
        --train_gt_path ${TRAIN_GT_PATH} \
        --base_graph_path ${BASE_BOTTOM_INDEX} \
        --metric ip_float \
        --efC_AKNN ${efC_AKNN} \
        --result_index_path ${BASE_NGFIX_INDEX}
    
    if [ $? -ne 0 ] || [ ! -f "${BASE_NGFIX_INDEX}" ]; then
        echo "Error: Failed to build base NGFix index"
        exit 1
    fi
    echo "Base NGFix index built successfully"
else
    echo "Base NGFix index already exists: ${BASE_NGFIX_INDEX}"
fi

# Step 3: Run comprehensive deletion comparison test
echo "=== Step 3: Running comprehensive deletion comparison test ==="
echo "This will:"
echo "  1. Insert 3M additional data"
echo "  2. Apply NGFix + RFix with 10M train queries"
echo "  3. Test initial recall/NDC/latency"
echo "  4. Store index to SSD"
echo "  5. Test lazy deletion"
echo "  6. Test real deletion"
echo "  7. Compare all three"
echo ""

# Build the test program if needed
if [ ! -f "./build/test/test_comprehensive_deletion_comparison" ]; then
    echo "Building test program..."
    cd build
    make test_comprehensive_deletion_comparison
    cd ..
fi

# Run the comprehensive test
./build/test/test_comprehensive_deletion_comparison \
    --base_data_path ${BASE_DATA_PATH} \
    --additional_data_path ${ADDITIONAL_DATA_PATH} \
    --train_query_path ${TRAIN_QUERY_PATH} \
    --train_gt_path ${TRAIN_GT_PATH} \
    --test_query_path ${TEST_QUERY_PATH} \
    --test_gt_path ${TEST_GT_PATH} \
    --base_index_path ${BASE_NGFIX_INDEX} \
    --result_index_path ${FINAL_INDEX} \
    --result_csv_path ${RESULT_CSV} \
    --metric ip_float \
    --M ${M} --MEX ${MEX} --efC ${efC} \
    --efC_AKNN ${efC_AKNN} --efC_delete ${efC_delete} \
    --K ${K} \
    --base_size 10000000 \
    --additional_size 3000000

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Test completed successfully ==="
    echo "Results saved to: ${RESULT_CSV}"
    echo "Final index saved to: ${FINAL_INDEX}"
    echo ""
    echo "First few lines of results:"
    head -20 ${RESULT_CSV}
    echo ""
    echo "Summary by stage:"
    echo "=================="
    echo "Initial:"
    grep "^initial," ${RESULT_CSV} | tail -7
    echo ""
    echo "Lazy Deletion:"
    grep "^lazy_deletion," ${RESULT_CSV} | tail -7
    echo ""
    echo "Real Deletion:"
    grep "^real_deletion," ${RESULT_CSV} | tail -7
else
    echo "Error: Test failed!"
    exit 1
fi

