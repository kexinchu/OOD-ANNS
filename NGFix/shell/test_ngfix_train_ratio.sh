#!/bin/bash

# 测试不同比例的train.gt.bin数据对NGFix增强图的影响
# 测试比例: 10%, 20%, 30%, 40%, 50%, 100%
# 输出: average out-degree 和 recall (efSearch=100)

# Don't use set -e, we want to handle errors manually

# 进入build目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${SCRIPT_DIR}

# 配置参数
MEX=48
M=16
efC=500
efSearch=100

# 数据路径
BASE_DIR=/workspace
DATA_DIR=/workspace/RoarGraph/data/t2i-10M
NGFIX_DATA_DIR=/workspace/NGFix/data/t2i-10M
RESULT_DIR=/workspace/NGFix/data/t2i-10M/ratio_test_results

# 输入文件
TRAIN_QUERY_PATH=${DATA_DIR}/query.train.10M.fbin
TRAIN_GT_FULL_PATH=${DATA_DIR}/train.gt.bin
BASE_GRAPH_PATH=${NGFIX_DATA_DIR}/t2i_10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index
TEST_QUERY_PATH=${DATA_DIR}/query.10k.fbin
TEST_GT_PATH=${DATA_DIR}/groundtruth-computed.10k.ibin

# 创建结果目录
mkdir -p ${RESULT_DIR}

# 测试比例数组
RATIOS=(0.1 0.2 0.3 0.4 0.5 1.0)

# 结果输出文件
RESULTS_FILE=${RESULT_DIR}/ratio_test_results.csv

# 写入CSV头部
echo "Ratio,Average_Out_Degree,Recall_efSearch${efSearch}" > ${RESULTS_FILE}

# 构建必要的工具（如果不存在）
echo "Checking and building necessary tools..."
if [ ! -f ./test/extract_subset_bin ]; then
    echo "Building extract_subset_bin tool..."
    make extract_subset_bin || { echo "Error: Failed to build extract_subset_bin"; exit 1; }
fi

if [ ! -f ./test/search_hnsw_ngfix_single ]; then
    echo "Building search_hnsw_ngfix_single tool..."
    make search_hnsw_ngfix_single || { echo "Error: Failed to build search_hnsw_ngfix_single"; exit 1; }
fi



echo "Starting ratio test..."
echo "========================================"

for ratio in "${RATIOS[@]}"; do
    # 使用awk计算百分比
    ratio_percent=$(awk "BEGIN {printf \"%.0f\", $ratio * 100}")
    echo ""
    echo "========================================"
    echo "Testing ratio: ${ratio_percent}%"
    echo "========================================"
    
    # 生成比例数据的路径
    TRAIN_GT_SUBSET_PATH=${RESULT_DIR}/train.gt.${ratio_percent}percent.bin
    TRAIN_QUERY_SUBSET_PATH=${RESULT_DIR}/query.train.${ratio_percent}percent.fbin
    INDEX_PATH=${RESULT_DIR}/t2i_10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}_ratio${ratio_percent}percent.index
    SEARCH_RESULT_PATH=${RESULT_DIR}/search_result_ratio${ratio_percent}percent.csv
    
    # 步骤1: 提取比例的train.gt.bin和train_query数据
    is_one=$(awk "BEGIN {print ($ratio == 1.0) ? 1 : 0}")
    if [ ! -f ${TRAIN_GT_SUBSET_PATH} ] || [ ! -f ${TRAIN_QUERY_SUBSET_PATH} ] || [ "$is_one" = "1" ]; then
        echo "Step 1: Extracting ${ratio_percent}% of train.gt.bin and train.query..."
        if [ "$is_one" = "1" ]; then
            # 100%直接使用原始文件
            cp ${TRAIN_GT_FULL_PATH} ${TRAIN_GT_SUBSET_PATH}
            cp ${TRAIN_QUERY_PATH} ${TRAIN_QUERY_SUBSET_PATH}
            echo "Using full train.gt.bin and train.query (100%)"
        else
            ./test/extract_subset_bin ${TRAIN_GT_FULL_PATH} ${TRAIN_GT_SUBSET_PATH} ${ratio}
            ./test/extract_subset_bin ${TRAIN_QUERY_PATH} ${TRAIN_QUERY_SUBSET_PATH} ${ratio}
        fi
    else
        echo "Step 1: Using existing subset files"
    fi
    
    # 步骤2: 构建NGFix索引
    echo "Step 2: Building NGFix index with ${ratio_percent}% train.gt.bin..."
    if [ ! -f ${INDEX_PATH} ]; then
        # 使用timeout防止程序无限挂起，设置较长超时时间（2小时）
        # 使用build_hnsw_ngfix_with_gt替代build_hnsw_ngfix（支持train_gt_path且更稳定）
        timeout 7200 ./test/build_hnsw_ngfix_with_gt \
            --train_query_path ${TRAIN_QUERY_SUBSET_PATH} \
            --train_gt_path ${TRAIN_GT_SUBSET_PATH} \
            --base_graph_path ${BASE_GRAPH_PATH} \
            --metric ip_float \
            --result_index_path ${INDEX_PATH} > ${RESULT_DIR}/build_log_${ratio_percent}percent.txt 2>&1
        BUILD_EXIT_CODE=$?
        
        # 检查构建是否成功
        if [ ${BUILD_EXIT_CODE} -ne 0 ] || [ ! -f ${INDEX_PATH} ]; then
            echo "Error: Index building failed for ratio ${ratio_percent}% (exit code: ${BUILD_EXIT_CODE})"
            cat ${RESULT_DIR}/build_log_${ratio_percent}percent.txt | tail -20
            echo "${ratio_percent}%,BUILD_FAILED,BUILD_FAILED" >> ${RESULTS_FILE}
            continue
        fi
        echo "Build completed successfully"
    else
        echo "Index already exists: ${INDEX_PATH}"
    fi
    
    # 步骤3和4: 使用单次搜索工具同时获取average out-degree和recall
    echo "Step 3: Testing search with efSearch=${efSearch} and extracting statistics..."
    
    # 运行搜索并捕获输出
    ./test/search_hnsw_ngfix_single \
        --test_query_path ${TEST_QUERY_PATH} \
        --test_gt_path ${TEST_GT_PATH} \
        --metric ip_float --K 100 --efSearch ${efSearch} \
        --index_path ${INDEX_PATH} > ${RESULT_DIR}/search_log_${ratio_percent}percent.txt 2>&1
    SEARCH_EXIT_CODE=$?
    SEARCH_OUTPUT=$(cat ${RESULT_DIR}/search_log_${ratio_percent}percent.txt)
    
    if [ ${SEARCH_EXIT_CODE} -ne 0 ]; then
        echo "Error: Search failed for ratio ${ratio_percent}% (exit code: ${SEARCH_EXIT_CODE})"
        echo "$SEARCH_OUTPUT" | tail -20
    fi
    
    # 提取average out-degree
    # 优先从构建日志中获取（更准确，因为是在构建后立即打印的）
    BUILD_OUT_DEGREE=$(grep "Average out-degree" ${RESULT_DIR}/build_log_${ratio_percent}percent.txt 2>/dev/null | tail -1 | awk '{print $3}')
    if [ ! -z "$BUILD_OUT_DEGREE" ]; then
        OUT_DEGREE=$BUILD_OUT_DEGREE
    else
        # 如果构建日志中没有，从搜索输出中获取
        OUT_DEGREE=$(echo "$SEARCH_OUTPUT" | grep "Average out-degree" | awk '{print $3}')
    fi
    
    if [ -z "$OUT_DEGREE" ]; then
        echo "Warning: Could not extract average out-degree for ratio ${ratio_percent}%"
        OUT_DEGREE="N/A"
    else
        echo "Average out-degree: ${OUT_DEGREE}"
    fi
    
    # 提取recall
    RECALL=$(echo "$SEARCH_OUTPUT" | grep "Average Recall:" | awk '{print $3}')
    
    if [ -z "$RECALL" ]; then
        echo "Warning: Could not extract recall for ratio ${ratio_percent}%"
        RECALL="N/A"
    else
        echo "Recall (efSearch=${efSearch}): ${RECALL}"
    fi
    
    # 记录结果
    echo "${ratio_percent}%,${OUT_DEGREE},${RECALL}" >> ${RESULTS_FILE}
    
    echo ""
    echo "Completed ratio ${ratio_percent}%: Out-degree=${OUT_DEGREE}, Recall=${RECALL}"
done

echo ""
echo "========================================"
echo "All tests completed!"
echo "Results saved to: ${RESULTS_FILE}"
echo "========================================"
cat ${RESULTS_FILE}

