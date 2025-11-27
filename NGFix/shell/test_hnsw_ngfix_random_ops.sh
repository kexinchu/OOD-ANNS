#!/bin/bash
# Random Insert/Delete + Search Test
# This script tests search performance while randomly inserting/deleting nodes
# Run from build directory

MEX=48
M=16
efC=500
efC_AKNN=1500
efs=100
K=100
SEARCH_QPS=128
OPS_QPS=128
TEST_DURATION_HOURS=1

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

# Extract 2M additional data (from 8M to 10M)
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

# Run random ops + search test
taskset -c 0,1 ./test/test_hnsw_ngfix_random_ops \
--index_path ${BASE_INDEX_PATH} \
--test_query_path /workspace/RoarGraph/data/t2i-10M/query.10k.fbin \
--test_gt_path /workspace/RoarGraph/data/t2i-10M/groundtruth-computed.10k.ibin \
--additional_data_path ${ADDITIONAL_DATA_PATH} \
--metric ip_float \
--K ${K} \
--efs ${efs} \
--efC ${efC} \
--search_qps ${SEARCH_QPS} \
--ops_qps ${OPS_QPS} \
--test_duration_hours ${TEST_DURATION_HOURS} \
--result_path /workspace/NGFix/data/t2i-10M/random_ops_search_results.json

echo "Random ops + search test completed."
echo "Results saved to: /workspace/NGFix/data/t2i-10M/random_ops_search_results.json"

