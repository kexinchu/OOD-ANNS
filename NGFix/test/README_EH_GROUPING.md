# EH矩阵计算分组优化测试

## 概述

这个测试实现了基于节点分组的EH矩阵计算优化方法，用于减少EH计算的复杂度。

## 实现原理

1. **节点分组**: 将2跳内相互可达的节点合并到一个group中
2. **代表节点选择**: 每个group选择距离query最近的节点作为代表
3. **分组EH计算**: 基于groups构建缩小的EH矩阵 (m x m)，而不是原始的 (M x M)
4. **连接处理**: 当group外的node需要连接到group内的node时，使用group的代表节点

## 使用方法

### 编译
```bash
cd /workspace/NGFix/build
cmake ..
make test_eh_grouping
```

### 运行
```bash
cd /workspace/NGFix/build
./test_eh_grouping.sh
```

或者手动运行:
```bash
./test/test_eh_grouping \
  --test_query_path <query_path> \
  --test_gt_path <groundtruth_path> \
  --metric ip_float \
  --K 100 \
  --num_queries 100 \
  --index_path <index_path> \
  --result_path <result_path>
```

## 输出结果

测试会输出JSON格式的结果文件，包含：

- **统计信息**:
  - `avg_M`: 原始节点数量（平均值）
  - `avg_m`: 分组后的group数量（平均值）
  - `avg_reduction_percent`: 节点数量减少百分比
  - `avg_original_latency_us`: 原始EH计算的平均延迟（微秒）
  - `avg_grouped_latency_us`: 分组EH计算的平均延迟（微秒）
  - `speedup`: 加速比
  - `original_matrix_size`: 原始矩阵大小（平均值）
  - `grouped_matrix_size`: 分组后矩阵大小（平均值）
  - `matrix_size_reduction`: 矩阵大小减少百分比

- **每个query的详细信息**:
  - `M`: 原始节点数量
  - `m`: 分组后的group数量
  - `reduction_percent`: 减少百分比
  - `original_latency_us`: 原始EH计算延迟
  - `grouped_latency_us`: 分组EH计算延迟
  - `speedup`: 加速比
  - `original_matrix_size`: 原始矩阵大小
  - `grouped_matrix_size`: 分组后矩阵大小

## 预期效果

- 减少节点数量: M -> m (m < M)
- 减少EH矩阵大小: (M x M) -> (m x m)
- 降低计算延迟: 矩阵大小减少带来计算量减少

