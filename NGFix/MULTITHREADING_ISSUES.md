# 多线程并发场景下的潜在问题分析

## 场景描述
在 `test_hnsw_ngfix_random_ops.sh` 中，当以下操作同时发生时：
- Insert 操作（通过 `InsertPoint`）
- Delete 操作（通过 `DeletePointByFlag`）
- Query 操作（通过 `searchKnn`）
- Partial Rebuild 操作（通过 `PartialRemoveEdges`，如果允许触发）

## 主要问题分类

### 1. PartialRemoveEdges 的全局写锁阻塞问题 ⚠️ 严重

**问题位置**: `ngfixlib/graph/hnsw_ngfix.h:355-372`

```cpp
void PartialRemoveEdges(float r) {
    for(int i = 0; i < n; ++i) {
        std::unique_lock <std::shared_mutex> lock(node_locks[i]);  // 写锁！
        // ... 修改节点的 NGFix 边
    }
}
```

**问题分析**:
- `PartialRemoveEdges` 会**顺序遍历所有节点**，对每个节点都加**写锁 (unique_lock)**
- 这会**完全阻塞**所有对该节点的并发操作：
  - ✅ Query 操作使用 `shared_lock` (读锁)，会被写锁阻塞
  - ✅ Insert 操作中的 `HNSWBottomLayerInsertion` 需要写锁修改节点，会被阻塞
  - ✅ Delete 操作中的 `DeleteAllFlagPointsByNGFix` 需要写锁，会被阻塞

**影响**:
- 所有 Query 操作在 Partial Rebuild 期间被阻塞
- 插入和删除操作也被阻塞
- 如果图很大（如 10M 节点），遍历所有节点需要很长时间（几秒钟）
- 导致系统完全停止响应

**证据**: 
- 从测试文档 `docs/concurrent_test_README.md` 可以看到：
  > Partial Rebuild会阻塞Query: `PartialRemoveEdges`会遍历所有节点并加写锁，会阻塞所有query
  > Partial Rebuild延迟: 3951 ms

### 2. DeleteAllFlagPointsByNGFix 的并行处理缺乏锁保护 ⚠️ 严重

**问题位置**: `ngfixlib/graph/hnsw_ngfix.h:382-436`

```cpp
void DeleteAllFlagPointsByNGFix(...) {
    // 获取删除节点列表
    {
        std::unique_lock <std::shared_mutex> lock(delete_lock);
        ids = delete_ids;
        delete_ids.clear();
    }
    
    // 并行处理删除操作 - 但没有对节点加锁！
    #pragma omp parallel for schedule(dynamic) num_threads(Threads)
    for(int i = 0; i < n; ++i) {
        if(ids.find(i) != ids.end()) {
            Graph[i].delete_node();  // 没有锁保护！
        } else {
            // 读取和修改邻居列表 - 没有锁保护！
            auto [outs, sz, st] = getNeighbors(i);
            // ... 修改 Graph[i] 的邻居
            Graph[i].replace_ngfix_neighbors(new_neighbors);
            Graph[i].replace_base_graph_neighbors(new_neighbors2);
        }
    }
    
    // 并行重建删除节点的 NGFix 边
    #pragma omp parallel for schedule(dynamic) num_threads(Threads)
    for(int i = 0; i < v_ids.size(); ++i) {
        // 调用 NGFix 重建边，内部会加锁
        NGFix(getData(v_ids[i]), gt, 100, 100);
    }
}
```

**问题分析**:
1. **严重的数据竞争**: 在并行循环中直接修改 `Graph[i]` 的邻居列表，**完全没有使用 `node_locks[i]` 保护**
   ```cpp
   #pragma omp parallel for schedule(dynamic) num_threads(Threads)
   for(int i = 0; i < n; ++i) {
       // ...
       auto [outs, sz, st] = getNeighbors(i);  // 读取邻居，没有锁！
       // ...
       Graph[i].replace_ngfix_neighbors(new_neighbors);  // 修改邻居，没有锁！
       Graph[i].replace_base_graph_neighbors(new_neighbors2);  // 修改邻居，没有锁！
   }
   ```
   - Query 线程可能正在使用 `shared_lock` 读取该节点的邻居，与这里的无锁修改发生数据竞争
   - Insert 线程可能正在使用 `unique_lock` 修改该节点，也会发生竞争
   - 这会导致**未定义行为**：可能读取到损坏的数据、段错误、或返回错误的结果

2. **部分更新问题**: 由于并行处理，不同节点的更新可能在不同时间完成
   - Query 可能看到部分更新的图状态（某些节点已更新，某些未更新）
   - 导致查询结果不一致、不正确或包含已删除的节点
   - 图结构可能处于不一致的中间状态

3. **与 Insert 的竞争**: 
   - Insert 操作可能在 `DeleteAllFlagPointsByNGFix` 执行过程中修改相同的节点
   - 没有协调机制，可能导致：
     - Insert 添加的边被立即删除
     - Insert 添加的边覆盖删除操作的结果
     - 节点数据结构损坏

4. **getNeighbors 的无锁访问**:
   - `getNeighbors(i)` 只是返回 `Graph[i]` 的内部指针，没有锁保护
   - 在读取过程中，如果其他线程修改了 `Graph[i]`，可能读取到部分更新的数据
   - 读取到的 `outs` 指针可能指向已被释放的内存（如果节点被删除）

### 3. InsertPoint 与 Query 的锁竞争 ⚠️ 中等

**问题位置**: `ngfixlib/graph/hnsw_ngfix.h:338-352, 290-335`

```cpp
void InsertPoint(id_t id, size_t efC, T* vec) {
    // ...
    HNSWBottomLayerInsertion(data, id, efC);  // 内部会加写锁
}

void HNSWBottomLayerInsertion(T* data, id_t cur_id, size_t efC) {
    // 搜索邻居节点（需要读锁）
    auto res = searchKnnBaseGraphConstruction(data, efC, efC, NDC);
    
    // 为新节点加写锁
    {
        std::unique_lock <std::shared_mutex> lock(node_locks[cur_id]);
        Graph[cur_id].replace_base_graph_neighbors(neighbors);
    }
    
    // 为每个邻居节点加写锁并修改
    for(auto [_, neighbor_id] : neighbors) {
        std::unique_lock <std::shared_mutex> lock(node_locks[neighbor_id]);
        // 修改邻居列表
    }
}
```

**问题分析**:
1. **死锁风险**: Insert 需要对多个节点加写锁（当前节点 + 多个邻居节点）
   - 如果多个 Insert 操作同时修改有重叠邻居的节点，可能出现死锁
   - 例如：Insert A 锁住节点 1，等待节点 2；Insert B 锁住节点 2，等待节点 1

2. **与 Query 的竞争**:
   - Insert 对节点加写锁会阻塞所有对该节点的 Query 操作
   - 如果插入频率很高（如 128 QPS），会显著影响查询性能

3. **与 PartialRemoveEdges 的竞争**:
   - Insert 需要锁住节点，PartialRemoveEdges 也需要锁住所有节点
   - 如果 PartialRemoveEdges 开始执行，所有 Insert 操作都会被阻塞

### 4. DeletePointByFlag 的锁使用错误 ⚠️ 严重

**问题位置**: `ngfixlib/graph/hnsw_ngfix.h:440-449`

```cpp
void DeletePointByFlag(id_t id) {
    set_deleted(id);  // 设置删除标志，没有锁保护
    {
        std::shared_lock <std::shared_mutex> lock(delete_lock);  // 读锁！
        if(id != entry_point) {
            delete_ids.insert(id);  // 修改共享数据结构，但只持有读锁！
            n = n - 1;  // n 是 atomic，但 delete_ids.insert 需要写锁
        }
    }
}
```

**问题分析**:
1. **错误的锁使用**: `delete_ids.insert(id)` 是**写操作**，但使用的是 `shared_lock` (读锁)
   - `delete_ids` 是一个 `std::unordered_set<id_t>`，不是线程安全的
   - 多个 Delete 操作并发执行时，会同时持有读锁并修改 `delete_ids`，导致数据竞争
   - 可能导致 `delete_ids` 数据结构损坏、元素丢失或重复

2. **与 DeleteAllFlagPointsByNGFix 的竞争**:
   - `DeleteAllFlagPointsByNGFix` 使用 `unique_lock` (写锁) 保护 `delete_ids`
   - 如果多个 `DeletePointByFlag` 同时使用 `shared_lock`，理论上可以并发执行
   - 但 `DeletePointByFlag` 内部又有写操作 (`delete_ids.insert`)，这是矛盾的
   - 当 `DeleteAllFlagPointsByNGFix` 持有写锁时，所有 `DeletePointByFlag` 会被阻塞
   - 当多个 `DeletePointByFlag` 并发执行时，会同时修改 `delete_ids`，导致未定义行为

3. **set_deleted 缺乏保护**:
   - `set_deleted(id)` 直接修改 `vecdata`，没有锁保护
   - 可能与 Query 操作中的 `is_deleted()` 检查产生竞争

### 5. Query 操作中的不一致视图 ⚠️ 中等

**问题位置**: `ngfixlib/graph/hnsw_ngfix.h:786-851`

```cpp
std::vector<std::pair<float, id_t> > searchKnn(T* query_data, size_t k, size_t ef, size_t& ndc) {
    // ...
    while (!q->is_empty()) {
        std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
        auto [outs, sz, st] = getNeighbors(current_node_id);  // 获取邻居列表
        // 遍历邻居并加入队列
    }
}
```

**问题分析**:
1. **部分更新视图**: Query 在遍历图时，图可能正在被修改
   - 可能读取到部分更新的节点（某些节点已更新，某些未更新）
   - 可能导致查询路径不连续或错误

2. **已删除节点的可见性**:
   - `is_deleted()` 检查可能在节点被标记删除但还未从图中移除时返回 true
   - Query 可能访问到已删除的节点，导致结果不一致

3. **entry_point 的变化**:
   - `entry_point` 可能在 Query 执行过程中被修改（每 100K 插入更新一次）
   - 可能导致查询从错误的入口点开始

### 6. n 的使用在并发场景下的问题 ⚠️ 低

**问题位置**: 多个位置

- `InsertPoint`: `++n;` (n 是 `std::atomic<size_t>`，操作是原子的)
- `DeletePointByFlag`: `n = n - 1;` (n 是原子的)
- `PartialRemoveEdges`: `for(int i = 0; i < n; ++i)` (读取 n 的值)

**问题分析**:
- `n` 是 `std::atomic<size_t>`，所以单个操作是原子的
- 但存在 **Time-of-check Time-of-use (TOCTOU)** 问题：
  - `PartialRemoveEdges` 在循环开始时读取 `n` 的值
  - 在循环执行过程中，`n` 可能被其他线程修改（Insert/Delete）
  - 可能导致循环处理不完整的节点集合，或者访问已删除/未创建的节点
- 类似地，`DeleteAllFlagPointsByNGFix` 在循环开始时使用 `n`，在循环过程中 `n` 可能变化

## 潜在的死锁场景

### 场景 1: Insert + PartialRemoveEdges
```
Thread 1 (Insert): 锁住节点 A，等待锁住节点 B
Thread 2 (PartialRemoveEdges): 锁住节点 B，等待锁住节点 A
→ 死锁（取决于锁的获取顺序）
```

### 场景 2: Insert + Insert
```
Thread 1 (Insert A): 锁住节点 1，等待节点 2
Thread 2 (Insert B): 锁住节点 2，等待节点 1
→ 死锁（如果锁获取顺序不一致）
```

## 性能影响

1. **查询延迟激增**: 
   - Partial Rebuild 期间，所有查询被阻塞
   - 测试数据显示 P99 延迟增加 54%

2. **吞吐量下降**:
   - Insert/Delete/Query 互相阻塞
   - 系统整体吞吐量下降

3. **锁竞争开销**:
   - 大量线程等待锁释放
   - CPU 资源浪费在锁等待上

## 建议的解决方案

1. **PartialRemoveEdges**:
   - 使用细粒度锁，只锁住需要修改的节点
   - 或者使用无锁数据结构
   - 或者分批处理，避免长时间持有锁

2. **DeleteAllFlagPointsByNGFix**:
   - 在并行循环中为每个节点加适当的锁
   - 或者使用更细粒度的锁策略

3. **DeletePointByFlag**:
   - 修改为使用 `unique_lock` 而不是 `shared_lock`（或者在 insert/erase 时临时升级为写锁）
   - 考虑使用线程安全的数据结构（如 `std::unordered_set` 配合读写锁，或使用 `tbb::concurrent_unordered_set`）
   - 保护 `set_deleted()` 操作

4. **InsertPoint**:
   - 使用锁顺序化策略避免死锁
   - 或者使用更细粒度的锁

5. **Query**:
   - 考虑使用快照/版本机制
   - 或者使用无锁读取结构

