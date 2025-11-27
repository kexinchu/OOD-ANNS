#!/bin/bash

# 测试 Delete (包含真实delete和NGFix重建) + Query 并行执行
# 各自128 QPS下的latency和线程安全性

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${SCRIPT_DIR}/../build

# 配置参数
MEX=48
M=16
efC=500
efC_delete=500
efs=500
k=100
target_qps=128

# 数据路径（根据实际情况修改）
BASE_DIR=/workspace
DATA_DIR=/workspace/RoarGraph/data/t2i-10M
NGFIX_DATA_DIR=/workspace/NGFix/data/t2i-10M
RESULT_DIR=/workspace/NGFix/data/t2i-10M/concurrent_test_results

# 输入文件
INDEX_PATH=${NGFIX_DATA_DIR}/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_AKNN1500.index
TEST_QUERY_PATH=${DATA_DIR}/query.10k.fbin
TEST_GT_PATH=${DATA_DIR}/groundtruth-computed.10k.ibin

# 创建结果目录
mkdir -p ${RESULT_DIR}

# 测试参数
DELETE_START=8000000  # 从ID 8000000开始删除
DELETE_COUNT=10000     # 删除10000个向量
REBUILD_INTERVAL=200   # 每200个删除触发一次NGFix rebuild

# 结果文件
RESULT_PATH=${RESULT_DIR}/delete_query_concurrent_results.csv

echo "========================================"
echo "Delete + Query Concurrent Test"
echo "========================================"
echo "Index path: ${INDEX_PATH}"
echo "Query path: ${TEST_QUERY_PATH}"
echo "GT path: ${TEST_GT_PATH}"
echo "Result path: ${RESULT_PATH}"
echo "Delete QPS: ${target_qps}"
echo "Query QPS: ${target_qps}"
echo "Delete start: ${DELETE_START}"
echo "Delete count: ${DELETE_COUNT}"
echo "========================================"

# 检查索引文件是否存在
if [ ! -f ${INDEX_PATH} ]; then
    echo "Error: Index file not found: ${INDEX_PATH}"
    exit 1
fi

# 检查测试程序是否存在
if [ ! -f ./test/test_delete_query_concurrent ]; then
    echo "Building test_delete_query_concurrent..."
    make test_delete_query_concurrent || { echo "Error: Failed to build test_delete_query_concurrent"; exit 1; }
fi

# 运行测试
echo "Starting concurrent test..."
./test/test_delete_query_concurrent \
    --index_path ${INDEX_PATH} \
    --test_query_path ${TEST_QUERY_PATH} \
    --test_gt_path ${TEST_GT_PATH} \
    --metric ip_float \
    --result_path ${RESULT_PATH} \
    --K ${k} \
    --efs ${efs} \
    --efC_delete ${efC_delete} \
    --delete_start ${DELETE_START} \
    --delete_count ${DELETE_COUNT} \
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

