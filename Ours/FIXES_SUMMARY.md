# Recall波动修复总结

## 修复时间
2025-12-23

## 修复内容

### 1. ✅ 优化UpdateNodeAccessTime的全局锁竞争（最高优先级）

**问题**：
- 每次搜索访问节点时都会调用`UpdateNodeAccessTime`
- 使用`unique_lock`获取全局的`lazy_delete_lock`
- 在高并发搜索（1000 QPS）时，所有搜索线程都在竞争这一个全局锁

**修复方案**：
- 使用thread-local buffer批量更新节点访问时间
- 每100次更新才flush一次到全局map，大幅减少锁竞争
- 在析构函数中flush所有thread-local buffers

**代码位置**：`ourslib/graph/hnsw_ours.h:1002-1035`

### 2. ✅ 修复查询统计问题

**问题**：
- `ConnectivityEnhancementThread`每秒都会清空`query_results_buffer`
- 导致`StatisticsThread`每分钟统计时，buffer可能已经被清空
- 这是导致查询数量只有1-159的主要原因！

**修复方案**：
- `ConnectivityEnhancementThread`不再清空buffer，只读取副本
- `StatisticsThread`可以正常累积每分钟的查询结果

**代码位置**：`test/test_runtime_update_end2end.cc:422-428`

### 3. ✅ 优化CleanupEdgesToDeletedNodes的锁升级

**问题**：
- 搜索访问节点时可能调用`CleanupEdgesToDeletedNodes`
- 使用`unique_lock`获取节点锁（锁升级），阻塞其他搜索线程

**修复方案**：
- 使用`try_lock`，如果无法获取锁就跳过清理（下次再试）
- 避免阻塞搜索操作

**代码位置**：`ourslib/graph/hnsw_ours.h:959-1016`

### 4. ✅ 优化NGFixOptimized的批量锁获取

**问题**：
- 插入后立即调用`NGFixOptimized`优化连通性
- 需要获取多个节点的`unique_lock`，阻塞搜索操作

**修复方案**：
- 使用`try_lock`，如果无法获取锁就跳过该节点
- 优化会在后续有机会时完成

**代码位置**：`ourslib/graph/hnsw_ours.h:1496-1519`

### 5. ✅ 优化NGFix的锁获取

**问题**：
- 与NGFixOptimized类似，需要获取多个节点的锁

**修复方案**：
- 使用`try_lock`避免阻塞搜索

**代码位置**：`ourslib/graph/hnsw_ours.h:1530-1552`

### 6. ✅ 优化CleanupPendingDeleteEdges的后台清理

**问题**：
- 后台线程定期清理，需要获取多个节点的`unique_lock`
- 与搜索操作竞争节点锁

**修复方案**：
- 使用`try_lock`，如果无法获取锁就跳过该节点
- 减少对搜索操作的影响

**代码位置**：`ourslib/graph/hnsw_ours.h:2278-2318`

## 修复效果验证

### 测试配置
- Search QPS: 1000
- Insert QPS: 20
- Delete QPS: 2
- Duration: 10 minutes (验证测试)

### 修复前（q1000_i20_d2版本）
- Query count: 1-159 per minute（严重不足）
- Recall std: 0.013787
- Recall range: [0.735, 1.0]（极端波动）

### 修复后（初步验证）
- Query count: **7347-8005 per minute**（大幅提升！）
- Recall: 0.9882-0.9895（非常稳定）
- 无极端波动

### 预期效果
- Query count应该接近60000 per minute（1000 QPS * 60秒）
- Recall应该更稳定，std < 0.01
- 无极端低recall值（<0.95）

## 技术细节

### Thread-Local Buffer机制

```cpp
struct NodeAccessTimeBuffer {
    std::unordered_map<id_t, std::chrono::steady_clock::time_point> access_times;
    size_t flush_count;
    static constexpr size_t FLUSH_THRESHOLD = 100;  // 每100次更新flush一次
};
```

### Try-Lock模式

所有可能阻塞搜索的锁操作都改为：
```cpp
std::unique_lock<std::shared_mutex> lock(node_locks[u], std::try_to_lock);
if(lock.owns_lock()) {
    // 执行操作
}
// 如果无法获取锁，跳过（下次再试）
```

## 后续建议

1. **长期测试**：运行完整的10小时测试，验证长期稳定性
2. **性能监控**：监控锁等待时间，确认锁竞争确实减少
3. **进一步优化**：如果查询数量仍未达到预期，可能需要进一步优化统计逻辑

## 文件修改清单

1. `ourslib/graph/hnsw_ours.h`:
   - 添加NodeAccessTimeBuffer结构
   - 修改UpdateNodeAccessTime使用thread-local buffer
   - 修改CleanupEdgesToDeletedNodes使用try_lock
   - 修改NGFixOptimized使用try_lock
   - 修改NGFix使用try_lock
   - 修改CleanupPendingDeleteEdges使用try_lock
   - 在析构函数中flush buffers

2. `test/test_runtime_update_end2end.cc`:
   - 修复ConnectivityEnhancementThread不清空buffer

