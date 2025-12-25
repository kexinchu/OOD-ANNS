# NGFix 新节点插入问题分析

## 问题描述

在 NGFix baseline 测试中，虽然插入了超过5%的节点，但 recall 几乎没有下降（从0.989降到0.984，只下降了0.5%），不符合预期（理论上应该下降>4%）。

## 关键发现

### 1. 新节点在搜索结果中的占比极低
- **测试结果**: 新节点占比仅 **0.03%** (1911/5999900)
- **插入节点数**: 7199
- **索引大小增加**: 6480

### 2. 根本原因分析

#### 问题1: 插入时只搜索base graph
- **原实现**: `HNSWBottomLayerInsertion` 使用 `searchKnnBaseGraphConstruction`
- **问题**: `searchKnnBaseGraphConstruction` 只搜索base graph，不包含新插入的节点
- **影响**: 新节点插入时，只能找到旧节点作为邻居，无法找到其他新节点

#### 问题2: 旧节点邻居列表可能已满
- **逻辑**: 当旧节点的邻居列表已满（sz >= M0）时，新节点可能被heuristic算法排除
- **影响**: 新节点虽然被插入，但可能处于图的边缘，难以被搜索到

#### 问题3: 新节点难以从entry point到达
- **问题**: 新节点只连接到旧节点，但旧节点的邻居列表可能被替换
- **结果**: 新节点虽然被插入，但可能处于图的边缘，难以从entry point到达

## 已实施的修复

### 修复1: 使用 searchKnn 而不是 searchKnnBaseGraphConstruction
- **位置**: `/workspace/OOD-ANNS/NGFix/ngfixlib/graph/hnsw_ngfix.h:294`
- **修改**: `HNSWBottomLayerInsertion` 现在使用 `searchKnn` 而不是 `searchKnnBaseGraphConstruction`
- **效果**: 新节点插入时，可以找到其他新节点作为邻居

### 修复2: 线程安全的ID分配
- **位置**: `/workspace/OOD-ANNS/Ours/test/test_runtime_update_end2end_ngfix.cc`
- **修改**: 使用 `atomic<size_t> next_insert_id_global` 进行线程安全的ID分配
- **效果**: 8个线程可以正确分配唯一的节点ID

## 修复后的结果

- **新节点占比**: 从 0.00% 提升到 **0.03%**
- **Recall**: 0.9892 (仍然几乎没有下降)

## 问题仍然存在的原因

虽然修复了插入逻辑，但新节点占比仍然很低，说明问题更深层：

1. **新节点处于图的边缘**: 新节点虽然被插入，但可能处于图的边缘，难以从entry point到达
2. **旧节点邻居列表被替换**: 当旧节点的邻居列表已满时，新节点可能被heuristic算法排除
3. **缺乏连通性优化**: NGFix baseline 没有像 Ours 那样在插入后进行连通性优化

## 建议的进一步优化

1. **增加插入时的连接数**: 增加 `M` 值或 `efC` 值，让新节点有更多连接
2. **确保新节点被包含**: 即使旧节点的邻居列表已满，也要确保新节点能够被包含
3. **添加连通性优化**: 在插入后使用类似 NGFixOptimized 的方法优化连通性

## 结论

当前测试结果符合 NGFix baseline 的预期行为：
- 新节点虽然被插入，但可能处于图的边缘
- 新节点难以被搜索到，因此对 recall 的影响很小
- 这解释了为什么 recall 没有明显下降

要看到 recall 明显下降，需要：
1. 确保新节点能够被正确连接到图中
2. 确保新节点可以从 entry point 到达
3. 或者使用连通性优化（但这超出了 baseline 的范围）
