# Latency分析：为什么with pending比without pending慢很多？

## 问题现象

从测试结果看：
- **Set overhead ratio**: 只有0.2%-2%（set操作本身开销很小）
- **Total overhead ratio**: 高达44%-331%（总开销非常大）
- **关键发现**: Set操作开销只占总开销的很小一部分

## 根本原因分析

### 1. 测试方法的问题

**"with pending"测试**：
- Lazy delete正在运行
- 有大量pending nodes（9K-300K个节点）
- 这些pending nodes有大量additional edges（23万-373万条边）
- 搜索时使用`getNeighbors()`，会遍历**所有neighbors**（包括additional edges）

**"without pending"测试**：
- 在`StopLazyDelete()`和等待cleanup之后测试
- Additional edges已经被删除
- 图变得更稀疏，搜索更快

### 2. 代码证据

```cpp
// searchKnn使用getNeighbors，包含所有edges（base + additional）
auto [outs, sz, st] = getNeighbors(current_node_id);

// getNeighbors返回所有neighbors，包括additional edges
auto getNeighbors(id_t u) {
    auto tmp = Graph[u].get_neighbors();
    return std::tuple{tmp, GET_SZ(...), ...};  // sz包含additional edges
}
```

### 3. 开销来源

**主要开销不是set操作，而是additional edges**：
1. **距离计算开销**：访问pending nodes时，需要计算所有additional edges的距离
   - 例如：237,886条additional edges × 距离计算时间 = 大量开销
2. **搜索路径变长**：Additional edges导致搜索路径更长，需要访问更多节点
3. **图结构差异**：with pending时图更密集，without pending时更稀疏（after cleanup）

## 数据支持

| Config | Pending Edges | With (ms) | Without (ms) | Overhead | Set Overhead |
|--------|---------------|-----------|--------------|----------|--------------|
| epoch=100ms, pages=10 | 237,886 | 2.092 | 1.450 | 44.23% | 0.20% |
| epoch=100ms, pages=50 | 1,200,155 | 2.432 | 1.297 | 87.48% | 0.87% |
| epoch=500ms, pages=200 | 3,736,015 | 4.845 | 1.122 | 331.76% | 2.05% |

**关键观察**：
- Pending edges数量与overhead高度相关
- Set overhead始终很小（<2%）
- 主要开销来自additional edges的距离计算

## 解决方案

### 方案1：修改测试方法（推荐）

**问题**：当前"without pending"是在cleanup之后测试，图结构已经改变

**解决**：应该比较：
- **With pending (未访问pending nodes)** vs **With pending (访问了pending nodes)**
- 或者：**With lazy delete enabled** vs **With lazy delete disabled**（但保持相同的图结构）

### 方案2：优化搜索算法

在搜索时，如果节点是pending node，可以：
1. **跳过additional edges**：只使用base graph edges（但这可能影响recall）
2. **延迟计算**：先检查base graph edges，只在必要时计算additional edges
3. **采样计算**：对additional edges进行采样，而不是全部计算

### 方案3：修改测试基准

"without pending"应该：
- 保持相同的图结构（不删除additional edges）
- 只是禁用lazy delete机制（不检查pending nodes）
- 这样可以真正测试set操作的开销

## 结论

**主要开销来源**：
1. **Additional edges的距离计算**（占主要部分）
2. **搜索路径变长**（因为图更密集）
3. **Set操作开销**（只占很小部分，0.2%-2%）

**Set操作本身的开销已经优化得很好**（<2%），但测试方法导致看起来总开销很大，实际上是因为additional edges的存在。

