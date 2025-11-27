#!/bin/bash

# 测试 Insert (包含真实partial rebuild) + Query 并行执行
# 各自128 QPS下的latency和线程安全性

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${SCRIPT_DIR}/../build

# 配置参数
MEX=48
M=16
efC=500
efs=500
k=100
target_qps=128
partial_rebuild_ratio=0.2  # 20% partial rebuild

# 数据路径（根据实际情况修改）
BASE_DIR=/workspace
DATA_DIR=/workspace/RoarGraph/data/t2i-10M
NGFIX_DATA_DIR=/workspace/NGFix/data/t2i-10M
RESULT_DIR=/workspace/NGFix/data/t2i-10M/concurrent_test_results

# 输入文件
INDEX_PATH=${NGFIX_DATA_DIR}/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN1500.index
BASE_DATA_PATH=${DATA_DIR}/base.10M.fbin
TEST_QUERY_PATH=${DATA_DIR}/query.10k.fbin
TEST_GT_PATH=${DATA_DIR}/groundtruth-computed.10k.ibin

# 创建结果目录
mkdir -p ${RESULT_DIR}

# 测试参数
INSERT_ST_ID=10000000  # 从ID 10000000开始插入
INSERT_COUNT=10000      # 插入10000个向量
REBUILD_INTERVAL=200    # 每200个插入触发一次partial rebuild

# 结果文件
RESULT_PATH=${RESULT_DIR}/insert_query_concurrent_results.csv

echo "========================================"
echo "Insert + Query Concurrent Test"
echo "========================================"
echo "Index path: ${INDEX_PATH}"
echo "Base data path: ${BASE_DATA_PATH}"
echo "Query path: ${TEST_QUERY_PATH}"
echo "GT path: ${TEST_GT_PATH}"
echo "Result path: ${RESULT_PATH}"
echo "Insert QPS: ${target_qps}"
echo "Query QPS: ${target_qps}"
echo "Partial rebuild ratio: ${partial_rebuild_ratio}"
echo "Insert start: ${INSERT_ST_ID}"
echo "Insert count: ${INSERT_COUNT}"
echo "========================================"

# 检查索引文件是否存在
if [ ! -f ${INDEX_PATH} ]; then
    echo "Error: Index file not found: ${INDEX_PATH}"
    exit 1
fi

# 检查测试程序是否存在
if [ ! -f ./test/test_insert_query_concurrent ]; then
    echo "Building test_insert_query_concurrent..."
    make test_insert_query_concurrent || { echo "Error: Failed to build test_insert_query_concurrent"; exit 1; }
fi

# 运行测试
echo "Starting concurrent test..."
./test/test_insert_query_concurrent \
    --index_path ${INDEX_PATH} \
    --base_data_path ${BASE_DATA_PATH} \
    --test_query_path ${TEST_QUERY_PATH} \
    --test_gt_path ${TEST_GT_PATH} \
    --metric ip_float \
    --result_path ${RESULT_PATH} \
    --K ${k} \
    --efs ${efs} \
    --efC ${efC} \
    --insert_st_id ${INSERT_ST_ID} \
    --insert_count ${INSERT_COUNT} \
    --partial_rebuild_ratio ${partial_rebuild_ratio} \
    --rebuild_interval ${REBUILD_INTERVAL} \
    --qps ${target_qps}

EXIT_CODE=$?

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Test failed with exit code ${EXIT_CODE}"
    exit 1
fi

echo ""
echo "========================================"
echo "Test completed successfully!"
echo "Results saved to: ${RESULT_PATH}"
echo "========================================"

