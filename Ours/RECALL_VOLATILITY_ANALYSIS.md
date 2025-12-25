# Recall波动原因分析报告

## 执行摘要

通过对比 `q1000_i20_d2` 版本和 `ngfix` 版本的测试结果，发现了recall波动的多个根本原因。主要问题包括：

1. **锁竞争**（高影响）：多个全局锁在并发操作时造成阻塞
2. **查询数量统计偏差**（高影响）：低查询数量时单个查询影响巨大
3. **图修复操作干扰**（中影响）：NGFixOptimized和清理操作修改图结构
4. **节点访问时间更新锁竞争**（中影响）：每次搜索访问节点都需要获取全局锁

## 数据分析结果

### 统计对比

- **q1000_i20_d2版本**:
  - Mean recall: 0.979664
  - Std Dev: 0.013787
  - Range: [0.735, 1.0]
  - 极端值：0.735 (最低)，1.0 (最高)

- **ngfix版本**:
  - Mean recall: 0.964637
  - Std Dev: 0.014494
  - Range: [0.939, 0.989]
  - 波动更平滑，无极端值

### 关键发现

1. **低查询数量时波动更大**:
   - Query count < 20: std = 0.034 (波动大)
   - Query count > 100: std = 0.005 (波动小)
   - 说明在查询数量少时，单个低recall查询会显著影响平均值

2. **35个突然下降事件**（>0.02的下降）:
   - 大多数发生在查询数量很少时（1-5个查询）
   - 极端例子：0.987778 -> 0.735000（下降0.25），只有2个查询

3. **低recall期间的特征**:
   - Insert/Delete活动略高于平均水平（但差异不大）
   - Latency略高于平均水平（但差异不大）
   - 主要是查询数量波动导致统计不准确

## 根本原因分析（按影响大小排序）

### 🔴 1. 锁竞争问题（最高影响）

#### 1.1 UpdateNodeAccessTime的全局锁竞争

**位置**: `hnsw_ours.h:988-994`

```cpp
void UpdateNodeAccessTime(id_t node_id) {
    if(node_id >= max_elements || node_id >= n) return;
    
    auto now = std::chrono::steady_clock::now();
    std::unique_lock<std::shared_mutex> lock(lazy_delete_lock);  // 全局锁！
    node_access_times[node_id] = now;
}
```

**问题**:
- 在`searchKnn`中，**每个访问的节点**都会调用`UpdateNodeAccessTime`
- 使用`unique_lock`获取全局的`lazy_delete_lock`
- 在高并发搜索（1000 QPS）时，所有搜索线程都在竞争这一个全局锁
- 锁持有时间虽然短（只是写map），但竞争频率极高（每次节点访问）

**影响**:
- 搜索线程频繁阻塞，导致查询数量下降
- 阻塞时间累积导致查询延迟增加
- 在高并发下成为性能瓶颈

**证据**:
- 低查询数量时期往往对应高锁竞争时期
- ngfix版本没有这个机制，recall更稳定

#### 1.2 CleanupEdgesToDeletedNodes的锁升级

**位置**: `hnsw_ours.h:945-985`

```cpp
void CleanupEdgesToDeletedNodes(id_t node_id) {
    // ...
    std::unique_lock<std::shared_mutex> node_lock(node_locks[node_id]);  // unique_lock
    // 清理边...
}
```

**问题**:
- 在`searchKnn`中，访问每个节点时都可能调用`CleanupEdgesToDeletedNodes`
- 使用`unique_lock`获取节点锁（锁升级）
- 如果多个搜索线程访问同一个节点，其中一个获取unique_lock会阻塞其他线程的shared_lock
- 清理操作可能比较耗时（遍历邻居、重建邻居列表）

**影响**:
- 搜索线程在清理期间被阻塞
- 影响搜索性能，可能导致查询数量下降

#### 1.3 NGFixOptimized的批量锁获取

**位置**: `test_runtime_update_end2end.cc:219`

```cpp
index->NGFixOptimized(vec_data, gt_array, actual_k, actual_k);
```

**位置**: `hnsw_ours.h:1455-1460`

```cpp
for(auto [u, vs] : new_edges) {
    std::unique_lock <std::shared_mutex> lock(node_locks[u]);  // 批量获取unique_lock
    for(auto [v, eh] : vs) {
        Graph[u].add_ngfix_neighbors(v, eh, MEX);
    }
}
```

**问题**:
- 插入后立即调用`NGFixOptimized`优化连通性
- 需要获取多个节点的`unique_lock`来添加边
- 在优化期间，这些节点的搜索操作会被阻塞
- 优化可能涉及多个节点（Nq=100时最多100个节点）

**影响**:
- 插入操作期间，相关节点的搜索被阻塞
- 如果插入频率高（20 QPS），会频繁阻塞搜索
- ngfix版本没有这个优化，所以没有这个问题

#### 1.4 CleanupPendingDeleteEdges的后台清理

**位置**: `hnsw_ours.h:2236-2276`

```cpp
void CleanupPendingDeleteEdges() {
    for(const auto& [node_id, _] : nodes_to_process) {
        std::unique_lock<std::shared_mutex> node_lock(node_locks[node_id]);  // 批量清理
        // 删除不在in-serve集合中的边...
    }
}
```

**问题**:
- 后台线程定期调用（epoch周期）
- 需要获取多个节点的`unique_lock`来清理边
- 与搜索操作竞争节点锁

**影响**:
- 定期阻塞搜索操作
- 可能造成查询性能的周期性下降

### 🟡 2. 查询数量统计偏差（高影响）

**问题**:
- 每分钟统计的查询数量波动很大（1-159个查询）
- 当查询数量少时（<20），单个低recall查询会显著拉低平均值
- 例如：如果只有2个查询，其中一个recall=0.735，平均recall就是0.735

**证据**:
- 极端低recall值（0.735）出现在查询数量=2的时候
- 查询数量>100时，recall标准差只有0.005
- 查询数量<20时，recall标准差达到0.034

**原因**:
- 可能是锁竞争导致查询线程阻塞，查询数量下降
- 也可能是查询处理延迟导致统计周期内完成查询减少

### 🟡 3. 图结构修改期间的搜索（中影响）

**问题**:
- 在插入、删除、图修复操作进行时，图结构正在修改
- 搜索在这些修改过程中进行，可能看到不一致的图状态
- 可能导致搜索结果不稳定

**影响**:
- 轻微的recall波动
- 特别是在插入和NGFixOptimized同时进行时

### 🟢 4. Pending Delete检查的开销（低影响）

**位置**: `hnsw_ours.h:1686-1720`

**问题**:
- 搜索时需要检查节点是否在pending_delete集合中
- 虽然有优化（cache、binary search），但仍有一定开销
- 虽然单个检查很快（~纳秒级），但累积起来可能有影响

**影响**:
- 轻微的性能开销
- 对recall波动影响较小

## 对比ngfix版本

ngfix版本更稳定的原因：

1. **没有UpdateNodeAccessTime机制**：不需要在搜索时更新节点访问时间，避免了全局锁竞争
2. **没有CleanupEdgesToDeletedNodes**：不在搜索时清理边，避免了锁升级
3. **没有NGFixOptimized优化**：插入后不立即优化，避免了批量锁获取
4. **没有PendingDelete机制**：没有后台清理操作，减少了锁竞争

## 建议的优化方案（按优先级）

### 🔴 优先级1：优化UpdateNodeAccessTime的锁竞争

**方案A：使用per-node锁或lock-free数据结构**
```cpp
// 使用per-node lock-free的访问时间更新
// 或者使用thread-local buffer批量更新
```

**方案B：延迟更新或批量更新**
- 使用thread-local buffer收集访问时间
- 定期批量更新，减少锁竞争频率

### 🔴 优先级2：优化CleanupEdgesToDeletedNodes

**方案A：异步清理**
- 不在搜索路径上直接清理
- 标记需要清理的边，由后台线程批量清理

**方案B：使用try_lock避免阻塞**
- 如果无法获取unique_lock，跳过清理（下次再试）

### 🟡 优先级3：优化NGFixOptimized的执行时机

**方案A：延迟优化**
- 不立即在插入后优化
- 收集需要优化的节点，批量优化

**方案B：使用try_lock**
- 如果无法获取锁，跳过该节点的优化

### 🟡 优先级4：改进统计方法

**方案A：使用更长的统计窗口**
- 使用滑动窗口统计，减少单点波动影响

**方案B：记录查询数量和recall的分布**
- 不只是平均值，还要记录中位数、分位数

## 测试验证建议

1. **锁竞争测试**：
   - 使用`perf`或自定义工具测量锁等待时间
   - 比较有/无UpdateNodeAccessTime的性能差异

2. **统计偏差测试**：
   - 在不同查询数量下统计recall
   - 验证查询数量与recall波动的关系

3. **移除优化测试**：
   - 临时禁用UpdateNodeAccessTime和CleanupEdgesToDeletedNodes
   - 对比recall稳定性

## 结论

recall波动的主要原因按影响大小排序：

1. **UpdateNodeAccessTime的全局锁竞争** - 最高影响，每次节点访问都竞争全局锁
2. **查询数量统计偏差** - 高影响，低查询数量时统计不准确
3. **CleanupEdgesToDeletedNodes的锁升级** - 中影响，搜索时清理边造成阻塞
4. **NGFixOptimized的批量锁获取** - 中影响，插入后优化阻塞搜索
5. **图结构修改期间的搜索** - 中影响，可能看到不一致状态
6. **Pending Delete检查开销** - 低影响，优化较好但仍有开销

建议优先解决锁竞争问题，特别是UpdateNodeAccessTime的全局锁竞争。

