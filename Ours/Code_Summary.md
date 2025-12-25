# Ours 项目代码与设计 Presentation 指南

## 总体架构概览

**Ours** 是一个针对 **OOD (Out-of-Distribution) ANNS** 的优化实现，核心目标是：
1. **真正的runtime update**：支持在线动态更新，无需partial rebuild
2. **智能query选择**：选择hard query来指导图更新
3. **分散式更新**：将insert/delete/update操作分散在时间线上
4. **并发一致性**：保证读QPS > 写QPS时的并发安全性

---

## 1. Query（查询）模块

### 1.1 核心查询流程

**主要函数**：`searchKnn()` 

#### 代码结构

```cpp
std::vector<std::pair<float, id_t> > searchKnn(T* query_data, size_t k, size_t ef, size_t& ndc) {
    // 1. 初始化搜索队列（QuadHeap）
    ngfixlib::Search_QuadHeap q0(ef, visited_list_pool_);
    
    // 2. 从entry_point开始
    float dist = getQueryDist(entry_point, query_data);
    q->push(entry_point, dist, is_deleted(entry_point));
    
    // 3. 主搜索循环
    while (!q->is_empty()) {
        // 3.1 获取当前节点
        std::pair<float, id_t> current_node_pair = q->get_next_id();
        id_t current_node_id = current_node_pair.second;
        
        // 3.2 获取节点锁（读锁）
        std::shared_lock<std::shared_mutex> lock(node_locks[current_node_id]);
        auto [outs, sz, st] = getNeighbors(current_node_id);
        
        // 3.3 检查是否为pending node（用于过期边删除）
        // OPTIMIZED: 完全lock-free检查，直接读取atomic数组
        bool is_pending = false;
        if(pending_delete_enabled.load(std::memory_order_acquire)) {
            // Lock-free read: 直接读取atomic bool数组，无需任何锁
            is_pending = is_pending_node(current_node_id);
        }
        
        // 3.4 遍历邻居节点
        for (int i = st; i < st + sz; ++i) {
            id_t candidate_id = outs[i];
            if (!q->is_visited(candidate_id)) {
                float dist = getQueryDist(candidate_id, query_data);
                q->push(candidate_id, dist, is_deleted(candidate_id));
                
                // 3.5 如果是pending node，记录in-serve edge
                if(is_pending) {
                    RecordInServeEdge(current_node_id, candidate_id);
                }
            }
        }
    }
    
    // 4. 返回top-k结果
    return q->get_result(k);
}
```

#### 关键设计点

1. **细粒度读锁**：每个节点使用独立的`shared_mutex`，允许并发读
2. **Pending Delete集成**：在查询过程中检查pending nodes并记录in-serve edges
3. **Lock-free检查**：使用`marked_pending_nodes`原子数组，完全无锁的O(1)检查，开销极小（~1ns）

### 1.2 带指标的查询（Hard Query检测）

**主要函数**：`searchKnnWithLightweightMetrics()`

#### 代码结构

```cpp
std::tuple<std::vector<std::pair<float, id_t> >, size_t, LightweightMetrics> 
searchKnnWithLightweightMetrics(T* query_data, size_t k, size_t ef, size_t& ndc, float alpha = 0.2f) {
    LightweightMetrics metrics;
    
    // 跟踪关键指标
    metrics.t_last_improve = 0;  // 最后一次改进的步数
    metrics.d_worst_early = ...;  // 早期阶段的最差距离
    metrics.d_worst_final = ...;  // 最终的最差距离
    
    while (!q->is_empty()) {
        // 检查是否在早期阶段（alpha * ef）
        if(!early_stage_captured && ndc >= S_0) {
            metrics.d_worst_early = d_worst_current;
        }
        
        // 检查是否有改进
        if(d_worst_current < d_worst_prev - eps) {
            metrics.t_last_improve = ndc;
        }
    }
    
    // 计算最终指标
    metrics.r_visit = metrics.S / ef;  // 访问预算使用率
    metrics.r_early = metrics.t_last_improve / metrics.S;  // 早期收敛率
    metrics.top1_last1_diff = last1_dist - top1_dist;  // 距离差异
    
    return std::make_tuple(res, ndc, metrics);
}
```

#### 指标说明

- **r_visit**: 访问节点比例（S/ef），值越大说明搜索越困难
- **r_early**: 早期终止比例（t_last_improve/S），值越小说明收敛越慢
- **top1_last1_diff**: 最佳候选与最终结果的差异，值越大说明搜索路径不理想

### 1.3 Hard Query检测

**主要函数**：`DetectHardQuery()` 

#### 代码结构

```cpp
template<typename T>
HardnessMetrics DetectHardQuery(HNSW_Ours<T>* searcher, T* query_data, size_t k, size_t ef, size_t dim) {
    // Stage 0: 轻量级指标
    auto [results, ndc_result, lw_metrics] = searcher->searchKnnWithLightweightMetrics(
        query_data, k, ef, ndc, 0.2f);
    
    // 使用ML预测器或轻量级指标计算hardness score
    if(searcher->hasHardnessPredictor()) {
        std::vector<float> features = extractHardnessFeatures<T>(...);
        metrics.hardness_score = searcher->hardness_predictor_->predict(features);
    } else {
        // Fallback: 基于轻量级指标
        hardness_score = lw_metrics.r_visit * (1.0f - lw_metrics.r_early) + 
                         lw_metrics.top1_last1_diff * 0.1f;
    }
    
    // Stage 1: Jitter（扰动稳定性）- 可选
    if(metrics.hardness_score > 0.8f) {
        // 高置信度，跳过jitter计算
        metrics.is_hard = (metrics.hardness_score > 0.5f);
    } 
    
    return metrics;
}
```

---

## 2. Insert（插入）模块

### 2.1 插入流程

**主要函数**：`InsertPoint()`  和 `HNSWBottomLayerInsertion()` 

#### 代码结构

```cpp
std::vector<std::pair<float, id_t> > InsertPoint(id_t id, size_t efC, T* vec) {
    // 1. 设置节点数据
    SetData(id, vec);
    auto data = getData(id);
    
    // 2. 如果图非空，执行插入
    std::vector<std::pair<float, id_t> > search_results;
    if(n != 0) {
        search_results = HNSWBottomLayerInsertion(data, id, efC);
    }
    
    // 3. 更新节点计数
    ++n;
    
    // 4. 定期更新entry point
    if(n % 100000 == 0) {
        SetEntryPoint();
    }
    
    return search_results;  // 返回搜索结果，可用于后续优化
}
```

#### HNSWBottomLayerInsertion详细流程

```cpp
std::vector<std::pair<float, id_t> > HNSWBottomLayerInsertion(T* data, id_t cur_id, size_t efC) {
    // 1. 搜索找到候选邻居（使用base graph）
    size_t NDC = 0;
    auto res = searchKnnBaseGraphConstruction(data, efC, efC, NDC);
    
    // 2. 使用启发式方法选择M个邻居
    auto neighbors = getNeighborsByHeuristic(res, M);
    
    // 3. 为新节点添加出边（需要写锁）
    {
        std::unique_lock<std::shared_mutex> lock(node_locks[cur_id]);
        Graph[cur_id].replace_base_graph_neighbors(neighbors);
    }
    
    // 4. 为邻居节点添加入边（双向连接）
    for(auto [_, neighbor_id] : neighbors) {
        std::unique_lock<std::shared_mutex> lock(node_locks[neighbor_id]);
        auto [ids, sz, st] = getBaseGraphNeighbors(neighbor_id);
        
        if(sz < M0) {
            // 邻居节点未满，直接添加
            Graph[neighbor_id].add_base_graph_neighbors(cur_id);
        } else {
            // 邻居节点已满，需要替换
            // 构建候选列表（包括新节点和现有邻居）
            std::vector<std::pair<float, id_t> > candidates;
            candidates.push_back({getDist(neighbor_id, cur_id), cur_id});
            for (int j = st; j < st + sz; j++) {
                candidates.push_back({getDist(ids[j], neighbor_id), ids[j]});
            }
            
            // 排序并选择最佳M0个邻居
            std::sort(candidates.begin(), candidates.end());
            auto new_neighbors = getNeighborsByHeuristic(candidates, M0);
            Graph[neighbor_id].replace_base_graph_neighbors(new_neighbors);
        }
    }
    
    return res;  // 返回搜索结果
}
```

#### 关键设计点

1. **双向连接**：插入节点时同时更新新节点的出边和邻居节点的入边
2. **启发式选择**：使用`getNeighborsByHeuristic()`选择最优邻居，保证图质量
3. **写锁保护**：所有图更新操作使用`unique_lock`保护
4. **返回搜索结果**：返回的搜索结果可用于后续的NGFix优化，避免重复搜索

### 2.2 与NGFix优化的集成

插入后可以立即使用hard query进行优化：

```cpp
// 插入节点
auto search_results = index->InsertPoint(new_id, efC, vec_data);

// 可选：使用hard query优化（如果检测到hard query）
if(is_hard_query) {
    int gt[k];
    // 使用search_results作为ground truth
    for(int i = 0; i < k; ++i) {
        gt[i] = search_results[i].second;
    }
    index->NGFixOptimized(query_data, gt, Nq, Kh);
}
```

---

## 3. Delete（删除）模块

### 3.1 删除流程

**主要函数**：`DeletePoint()` 

#### 代码结构

```cpp
void DeletePoint(id_t id) {
    // 1. 标记为已删除（使用delete_lock保护）
    {
        std::unique_lock<std::shared_mutex> lock(delete_lock);
        delete_ids.insert(id);
    }
    set_deleted(id);  // 设置删除标志位
    
    // 2. 收集所有邻居节点（在读锁保护下）
    std::unordered_set<id_t> neighbors;
    {
        std::shared_lock<std::shared_mutex> lock(node_locks[id]);
        auto [outs, sz, st] = getNeighbors(id);
        for(int j = st; j < st + sz; ++j) {
            neighbors.insert(outs[j]);
        }
    }
    
    // 3. 从所有邻居节点中移除指向该节点的边
    for(id_t neighbor_id : neighbors) {
        std::unique_lock<std::shared_mutex> lock(node_locks[neighbor_id]);
        auto [outs, sz, st] = getNeighbors(neighbor_id);
        
        // 分别处理ngfix边和base边
        std::vector<id_t> new_ngfix_neighbors;
        for(int j = ngfix_capacity - ngfix_sz; j < ngfix_capacity; ++j) {
            if(outs[j + 1] != id) {
                new_ngfix_neighbors.push_back(outs[j + 1]);
            }
        }
        
        std::vector<std::pair<float, id_t> > new_base_neighbors;
        for(int j = ngfix_capacity; j < ngfix_capacity + base_sz; ++j) {
            if(outs[j + 1] != id) {
                new_base_neighbors.push_back({0, outs[j + 1]});
            }
        }
        
        Graph[neighbor_id].replace_ngfix_neighbors(new_ngfix_neighbors);
        Graph[neighbor_id].replace_base_graph_neighbors(new_base_neighbors);
    }
    
    // 4. 删除节点本身
    {
        std::unique_lock<std::shared_mutex> lock(node_locks[id]);
        Graph[id].delete_node();
    }
}
```

### 3.2 获取入边（Incoming Edges）

**主要函数**：`GetIncomingEdges()` 

```cpp
std::vector<id_t> GetIncomingEdges(id_t node_id, size_t max_hops = 2) {
    // 使用两跳BFS扫描
    // 候选集合：C = Γ(A) ∪ ⋃_{u∈Γ(A)} Γ(u)
    // （A的邻居 + 邻居的邻居）
    
    std::queue<std::pair<id_t, int> > bfs_queue;
    bfs_queue.push({node_id, 0});
    
    while(!bfs_queue.empty()) {
        auto [current, hops] = bfs_queue.front();
        bfs_queue.pop();
        
        if(hops >= max_hops) continue;
        
        // 检查邻居节点是否指向node_id
        auto [neighbors, sz, st] = getNeighbors(current);
        for(int i = st; i < st + sz; ++i) {
            id_t neighbor = neighbors[i];
            
            // 检查neighbor是否有边指向node_id
            std::shared_lock<std::shared_mutex> lock(node_locks[neighbor]);
            auto [neighbor_neighbors, ...] = getNeighbors(neighbor);
            for(int j = ...) {
                if(neighbor_neighbors[j] == node_id) {
                    incoming.push_back(neighbor);
                    break;
                }
            }
        }
    }
    
    return incoming;
}
```

---

## 4. 过期边清理（Pending Delete）

### 4.1 核心机制

随着时间推移，以前新增的连通性（additional edges）可能"过期"，这些边会拖慢查询过程。Pending Delete采用"保护in-serve edge"的策略：
- **In-serve edge**：在查询过程中被实际使用的边
- **过期edge**：不再被查询使用的边，应该被删除

### 4.2 Epoch机制

```cpp
// id [48|16 int32]x64
void PendingDeleteWorker() {
    while(!should_stop_pending_delete.load()) {
        size_t epoch_ms = epoch_duration_ms.load();  // 例如：1000ms
        size_t pages = page_num.load();  // 例如：5页 (5 * 4KB = 20KB per epoch)
        // 假设10M节点，每页1024节点，共约9766页
        // 每epoch处理5页，约1953个epoch处理完所有节点
        // 如果每个epoch 1秒，约32分钟处理完一轮
        
        // Step 1: 随机选择pages，标记其中的nodes为"待删除"
        MarkNodesForDeletionByPages(pages);
        
        // Step 2: 清空上一轮的in-serve edges
        clear_inserve_edges();
        
        // Step 3: 等待一个epoch，让查询访问这些nodes并记录in-serve edges
        std::this_thread::sleep_for(std::chrono::milliseconds(epoch_ms));
        
        // Step 4: 清理不在in-serve集合中的additional edges
        CleanupPendingDeleteEdges();
        
        // Step 5: 清空pending_delete_nodes，准备下一轮
        clear_pending_node();
        
        current_epoch.fetch_add(1);
    }
}
```

### 4.3 标记节点

```cpp
void MarkNodesForDeletionByPages(size_t num_pages) {
    // 1. 计算总页数
    size_t total_pages = (num_nodes + nodes_per_page - 1) / nodes_per_page;
    
    // 2. 随机选择pages
    std::uniform_int_distribution<size_t> page_dist(0, total_pages - 1);
    std::unordered_set<size_t> selected_pages;
    while(selected_pages.size() < num_pages) {
        selected_pages.insert(page_dist(gen));
    }
    
    // 3. 只标记有additional edges的nodes
    std::vector<id_t> nodes_to_mark;
    for(size_t page_id : selected_pages) {
        for(id_t node_id = page_start; node_id < page_end; ++node_id) {
            auto [neighbors, sz, st] = getNeighbors(node_id);
            uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
            
            if(ngfix_sz > 0) {  // 有additional edges
                nodes_to_mark.push_back(node_id);
            }
        }
    }
    
    // 4. 批量设置pending nodes（Lock-free批量写入）
    if(!nodes_to_mark.empty()) {
        set_pending_nodes_batch(nodes_to_mark);  // 使用批量函数，更高效
    }
    
    // 注意：不再需要缓存排序向量，因为is_pending_node()现在是完全lock-free的
    // 直接读取atomic数组即可，无需任何缓存或查找
}
```

### 4.4 记录In-Serve Edge

**主要函数**：`RecordInServeEdge()` 

```cpp
void RecordInServeEdge(id_t node_id, id_t neighbor_id) {
    if(!pending_delete_enabled.load(std::memory_order_acquire)) {
        return;
    }
    
    // 使用线程本地缓冲区减少锁竞争
    ThreadLocalBuffer* buffer = GetThreadBuffer();
    size_t current_ep = current_epoch.load(std::memory_order_acquire);
    
    // 检查epoch是否变化（需要刷新缓冲区）
    if(buffer->epoch != current_ep) {
        FlushThreadBuffer(buffer);
        buffer->epoch = current_ep;
    }
    
    // 添加到线程本地缓冲区（非常快，只是vector push_back）
    buffer->edges.push_back({node_id, neighbor_id});
    total_set_insert_count.fetch_add(1, std::memory_order_relaxed);
    
    // 如果缓冲区满了，刷新到全局集合（批量插入）
    size_t threshold = buffer_flush_threshold.load(std::memory_order_acquire);
    if(buffer->edges.size() >= threshold) {
        FlushThreadBuffer(buffer);
    }
}

// 刷新线程本地缓冲区到全局集合
void FlushThreadBuffer(ThreadLocalBuffer* buffer) {
    if(buffer == nullptr || buffer->edges.empty()) {
        return;
    }
    
    // 批量插入，减少锁竞争
    set_inserve_edges_batch(buffer->edges);
    buffer->edges.clear();
    buffer->edges.shrink_to_fit();  // 释放内存
}
```

### 4.5 辅助函数

#### 节点管理函数（Lock-free优化）

```cpp
// 设置节点为pending状态（Lock-free写入 + Set tracking）
void set_pending_node(id_t node_id) {
    if(!pending_delete_enabled.load() || node_id >= max_elements) return;
    
    // Lock-free write: 直接设置atomic bool（非常快，~1ns）
    marked_pending_nodes[node_id].store(true, std::memory_order_release);
    
    // 同时更新set用于tracking（需要锁，但只在标记时调用，频率低）
    std::unique_lock<std::shared_mutex> lock(pending_delete_lock);
    pending_delete_nodes.insert(node_id);
}

// 批量设置pending nodes（Lock-free批量写入）
void set_pending_nodes_batch(const std::vector<id_t>& node_ids) {
    if(!pending_delete_enabled.load() || node_ids.empty()) return;
    
    // Lock-free batch update: 直接设置atomic bools
    for(id_t node_id : node_ids) {
        if(node_id < max_elements) {
            marked_pending_nodes[node_id].store(true, std::memory_order_release);
        }
    }
    
    // 同时更新set用于tracking
    std::unique_lock<std::shared_mutex> lock(pending_delete_lock);
    pending_delete_nodes.insert(node_ids.begin(), node_ids.end());
}

// 检查节点是否为pending（完全Lock-free，O(1)）
bool is_pending_node(id_t node_id) {
    if(node_id >= max_elements) return false;
    // Lock-free read: 直接读取atomic bool，无需任何锁（~1ns）
    return marked_pending_nodes[node_id].load(std::memory_order_acquire);
}

// 清空pending nodes（Lock-free批量清除）
void clear_pending_node() {
    // 先获取pending nodes的副本
    std::vector<id_t> nodes_to_clear;
    {
        std::shared_lock<std::shared_mutex> lock(pending_delete_lock);
        nodes_to_clear.assign(pending_delete_nodes.begin(), pending_delete_nodes.end());
    }
    
    // Lock-free批量清除atomic数组
    for(id_t node_id : nodes_to_clear) {
        if(node_id < max_elements) {
            marked_pending_nodes[node_id].store(false, std::memory_order_release);
        }
    }
    
    // 清空set
    std::unique_lock<std::shared_mutex> lock(pending_delete_lock);
    pending_delete_nodes.clear();
}
```

#### In-Serve Edge管理函数（独立锁优化）

```cpp
// 设置单个in-serve edge（使用独立的in_serve_edges_lock）
void set_inserve_edge(id_t start_node, id_t end_node) {
    if(!pending_delete_enabled.load()) return;
    std::unique_lock<std::shared_mutex> lock(in_serve_edges_lock);  // 独立锁
    in_serve_edges[start_node][end_node] = true;
}

// 批量设置in-serve edges（减少锁竞争）
void set_inserve_edges_batch(const std::vector<std::pair<id_t, id_t>>& edges) {
    if(!pending_delete_enabled.load()) return;
    std::unique_lock<std::shared_mutex> lock(in_serve_edges_lock);  // 独立锁
    for(const auto& [node_id, neighbor_id] : edges) {
        in_serve_edges[node_id][neighbor_id] = true;
    }
}

// 检查是否为in-serve edge（使用独立的in_serve_edges_lock）
bool is_inserve_edge(id_t start_node, id_t end_node) {
    if(!pending_delete_enabled.load()) return false;
    std::shared_lock<std::shared_mutex> lock(in_serve_edges_lock);  // 独立锁
    auto it = in_serve_edges.find(start_node);
    if(it == in_serve_edges.end()) return false;
    auto edge_it = it->second.find(end_node);
    return edge_it != it->second.end() && edge_it->second;
}

// 清空in-serve edges（使用独立的in_serve_edges_lock）
void clear_inserve_edges() {
    std::unique_lock<std::shared_mutex> lock(in_serve_edges_lock);  // 独立锁
    in_serve_edges.clear();
}
```

### 4.6 清理过期边

```cpp
void CleanupPendingDeleteEdges() {
    // 1. 获取pending nodes的副本（减少锁持有时间）
    std::unordered_set<id_t> nodes_to_process;
    {
        std::shared_lock<std::shared_mutex> lock(pending_delete_lock);
        nodes_to_process = pending_delete_nodes;  // 复制set
    }
    
    // 2. 处理每个pending node
    for(id_t node_id : nodes_to_process) {
        if(node_id >= max_elements) continue;
        
        std::unique_lock<std::shared_mutex> node_lock(node_locks[node_id]);
        auto [neighbors, sz, st] = getNeighbors(node_id);
        
        uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
        uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
        
        if(ngfix_sz > 0) {
            int ngfix_start = ngfix_capacity - ngfix_sz;
            std::vector<id_t> new_ngfix_neighbors;
            
            for(int j = ngfix_start; j < ngfix_capacity; ++j) {
                id_t neighbor_id = neighbors[j + 1];
                // 保留in-serve edges（这些边在查询中被实际使用）
                if(is_inserve_edge(node_id, neighbor_id)) {
                    new_ngfix_neighbors.push_back(neighbor_id);
                }
                // 随机保留20%的非in-serve edges（避免过度删除，保持图的连通性）
                // 这确保了即使某些边在当前epoch未被使用，也有机会保留
                else if(rand() % 5 != 0) {
                    new_ngfix_neighbors.push_back(neighbor_id);
                }
                // 其余80%的非in-serve edges将被删除（这些边在当前epoch未被使用）
            }
            
            // 替换ngfix neighbors
            Graph[node_id].replace_ngfix_neighbors(new_ngfix_neighbors);
        }
    }
}
```

### 4.7 性能优化说明

#### 清理策略的权衡

- **完全删除非in-serve edges**：可能导致图连通性下降，影响查询质量
- **随机保留20%**：在保持图质量的同时，逐步清理过期边
- **Epoch机制**：分散清理操作，避免一次性大量删除造成的性能波动

#### Lock-free优化

- **原子数组**：`marked_pending_nodes` 是一个`std::vector<std::atomic<bool>>`，支持完全lock-free的读写
- **O(1)检查**：`is_pending_node()`直接读取atomic数组，无需任何锁，开销极小（~1ns）
- **批量操作**：`set_pending_nodes_batch()`支持批量lock-free写入，提高标记效率
- **独立锁**：`in_serve_edges`使用独立的`in_serve_edges_lock`，与`pending_delete_lock`分离，减少锁竞争

#### 线程本地缓冲区优化

- **批量写入**：使用`set_inserve_edges_batch()`批量插入，减少锁获取次数
- **动态阈值**：`buffer_flush_threshold`可配置，平衡内存使用和锁竞争
- **内存管理**：使用`shrink_to_fit()`及时释放内存

---

## 5. Group-EH Matrix（分组EH计算） 

### 5.1 主流程

```cpp
void NGFixOptimized(T* query, int* gt, size_t Nq = 100, size_t Kh = 100) {
    // 1. 使用分组策略计算EH矩阵（大幅减少计算量）
    auto hardness_result = CalculateHardnessGroupedWithMapping(gt, Nq, Kh, S, query);
    auto& H = hardness_result.H;
    auto& node_idx_to_group = hardness_result.node_idx_to_group;
    
    // 2. 构建连通性矩阵f
    std::bitset<MAX_Nq> f[Nq];
    for(int i = 0; i < Nq; ++i) {
        for(int j = 0; j < Nq; ++j) {
            f[i][j] = (H[i][j] <= Kh) ? 1 : 0;  // EH值小于等于Kh表示可达
        }
    }
    
    // 3. 使用优化的边选择算法
    auto new_edges = getDefectsFixingEdgesOptimized(f, H, query, gt, Nq, Kh, 
                                                    &node_idx_to_group, &dist_cache);
    
    // 4. 添加边到图中（使用写锁保护）
    size_t ts = current_timestamp.fetch_add(1);
    for(auto [u, vs] : new_edges) {
        std::unique_lock<std::shared_mutex> lock(node_locks[u]);
        for(auto [v, eh] : vs) {
            Graph[u].add_ngfix_neighbors(v, eh, MEX);
            added_edges[u].push_back({u, v, eh, ts});  // 记录边，用于pending delete
        }
    }
}
```

### 5.2 分组EH计算

**主要函数**：`CalculateHardnessGroupedWithMapping()` 

#### 核心思想

将2跳内可达的节点视为同一组，在分组图上计算EH，然后映射回原始节点空间。

```cpp
HardnessResult CalculateHardnessGroupedWithMapping(int* gt, size_t Nq, size_t Kh, size_t S, T* query_data) {
    // Step 1: 节点分组（2跳BFS）
    auto groups = GroupNodesByReachability(this, gt, std::min(Nq, S), 2);
    size_t m = groups.size();  // m << Nq（通常减少60-85%）
    
    // Step 2: 构建分组图Gq_grouped
    std::unordered_map<int, std::vector<int> > Gq_grouped;
    for(size_t group_id = 0; group_id < groups.size(); ++group_id) {
        int u = groups[group_id][0];  // 使用组内第一个节点作为代表
        auto [neighbors, sz, st] = getNeighbors(u);
        // 检查邻居属于哪个组
        for(int v : neighbors) {
            int v_group = node_to_group_map[v];
            if(v_group != group_id) {
                Gq_grouped[group_id].push_back(v_group);
            }
        }
    }
    
    // Step 3: 在分组图上计算EH矩阵（m x m，而不是Nq x Nq）
    std::vector<std::vector<uint16_t> > H_grouped;
    // ... 计算H_grouped（使用标准EH算法） ...
    
    // Step 4: 映射回原始节点空间
    std::vector<std::vector<uint16_t> > H;
    H.resize(Nq);
    for(int i = 0; i < Nq; ++i) {
        H[i].resize(Nq, EH_INF);
        int g_i = node_idx_to_group[i];
        for(int j = 0; j < Nq; ++j) {
            int g_j = node_idx_to_group[j];
            if(g_i >= 0 && g_j >= 0) {
                H[i][j] = H_grouped[g_i][g_j];
            }
        }
    }
    
    return {H, node_idx_to_group};
}
```

### 5.3 优化的边选择

**主要函数**：`getDefectsFixingEdgesOptimized()` 

```cpp
auto getDefectsFixingEdgesOptimized(...) {
    // 1. 距离缓存：避免重复计算节点对距离
    std::vector<std::vector<float>> dist_cache;
    dist_cache.resize(Nq);
    for(int i = 0; i < Nq; ++i) {
        dist_cache[i].resize(Nq, -1.0f);
    }
    
    // 2. 收集候选对（跳过同组节点对）
    std::vector<std::pair<uint16_t, std::pair<int, int>>> candidates;
    for(int i = 0; i < Nq; ++i) {
        for(int j = i + 1; j < Nq; ++j) {
            if(f[i][j] == 1) continue;  // 已连通
            
            // 跳过同组节点对
            if(use_grouping && group_vec[i] == group_vec[j]) {
                continue;
            }
            
            // 计算距离（只计算一次）
            if(dist_cache[i][j] < 0) {
                dist_cache[i][j] = getDist(gt[i], gt[j]);
                dist_cache[j][i] = dist_cache[i][j];  // 对称
            }
            
            candidates.push_back({H[i][j], {i, j}});  // 使用EH值，不是距离
        }
    }
    
    // 3. 使用max-heap选择EH值最大的边（避免O(N²logN)排序）
    std::make_heap(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // 4. 迭代选择边
    while(!candidates.empty()) {
        // 获取EH值最大的对
        std::pop_heap(candidates.begin(), candidates.end(), ...);
        auto [eh_val, ij] = candidates.back();
        candidates.pop_back();
        
        auto [i, j] = ij;
        if(f[i][j] == 1) continue;  // 已连通（由于传递闭包）
        
        // 添加边
        new_edges[gt[i]].push_back({gt[j], eh_val});
        f[i][j] = 1;
        
        // 更新传递闭包
        for(int k = 0; k < Nq; ++k) {
            if(f[k][i]) {
                f[k] |= f[j];
            }
        }
        
        // 定期清理堆（每100次迭代）
        if(iterations % 100 == 0) {
            // 移除已连通的候选对
        }
    }
    
    return new_edges;
}
```

---

## 6. 锁机制（并发一致性）

### 6.1 锁架构

#### 数据结构

```cpp
class HNSW_Ours<T> {
    // 每个节点的独立锁（细粒度）
    std::vector<std::shared_mutex> node_locks;
    
    // 删除操作的全局锁
    std::shared_mutex delete_lock;
    std::unordered_set<id_t> delete_ids;
    
    // 过期边 delete 的机制
    std::shared_mutex pending_delete_lock;  // 仅用于pending_delete_nodes set的tracking
    std::unordered_set<id_t> pending_delete_nodes;  // 用于tracking/cleanup（带锁）
    std::vector<std::atomic<bool>> marked_pending_nodes;  // Lock-free数组，用于快速检查（无锁）
    std::shared_mutex in_serve_edges_lock;  // 独立的锁，用于in_serve_edges
    std::unordered_map<id_t, std::unordered_map<id_t, bool>> in_serve_edges;  // 嵌套map
};
```

### 6.2 读操作（Query）

```cpp
// 在searchKnn()中
std::shared_lock<std::shared_mutex> lock(node_locks[current_node_id]);
auto [outs, sz, st] = getNeighbors(current_node_id);
// 读锁允许并发读，不阻塞其他读操作
```

**特点**：
- 使用`shared_lock`（读锁），允许多个线程并发读
- 每个节点独立锁，减少锁竞争

### 6.3 写操作（Insert/Delete/Update）

```cpp
// 在InsertPoint()中
{
    std::unique_lock<std::shared_mutex> lock(node_locks[cur_id]);
    Graph[cur_id].replace_base_graph_neighbors(neighbors);
}

// 在DeletePoint()中
{
    std::unique_lock<std::shared_mutex> lock(delete_lock);
    delete_ids.insert(id);
}

// 在NGFixOptimized()中
{
    std::unique_lock<std::shared_mutex> lock(node_locks[u]);
    Graph[u].add_ngfix_neighbors(v, eh, MEX);
}
```

**特点**：
- 使用`unique_lock`（写锁），独占访问
- 写操作会阻塞读操作，但只影响被锁定的节点

### 6.4 删除节点保护

```cpp
// 在searchKnn()中
q->push(candidate_id, dist, is_deleted(candidate_id));

// is_deleted()检查
bool is_deleted(id_t id) {
    return (vecdata + id*size_per_element)[0];
}

// 在返回结果前过滤
auto res = q->get_result(k);
// 结果中不会包含已删除节点（因为is_deleted检查）
```

### 6.5 线程本地缓冲区

```cpp
// 线程本地缓冲区结构
struct ThreadLocalBuffer {
    size_t epoch;
    std::vector<std::pair<id_t, id_t>> edges;
};

// 获取线程本地缓冲区
ThreadLocalBuffer* GetThreadBuffer() {
    thread_local ThreadLocalBuffer buffer;
    return &buffer;
}

// 刷新缓冲区到全局集合（使用批量插入函数）
void FlushThreadBuffer(ThreadLocalBuffer* buffer) {
    if(buffer == nullptr || buffer->edges.empty()) {
        return;
    }
    
    // 批量插入，减少锁竞争
    set_inserve_edges_batch(buffer->edges);
    buffer->edges.clear();
    buffer->edges.shrink_to_fit();  // 释放内存
}
```

---

## 7. End-to-End流程示例

```
时间线：
t0: 启动pending delete (StartPendingDelete)
t1: Epoch 1开始 - 标记nodes为pending (MarkNodesForDeletionByPages)
t2: 查询访问pending nodes，记录in-serve edges (RecordInServeEdge)
t3: Epoch 1结束 - 清理过期edges (CleanupPendingDeleteEdges)
t4: Epoch 2开始 - 标记新的nodes为pending
...
```

### 7.1 使用示例

```cpp
// 1. 初始化索引
HNSW_Ours<float>* index = new HNSW_Ours<float>(L2_float, index_path);

// 2. 启动pending delete后台线程
index->StartPendingDelete(1000, 10);  // epoch=1s, pages=10

// 3. 正常查询（会自动记录in-serve edges）
auto results = index->searchKnn(query_data, k, ef, ndc);

// 4. 停止pending delete
index->StopPendingDelete(true);  // 等待清理完成

// 5. 获取统计信息
auto stats = index->GetPendingDeleteStats();
std::cout << "Total checks: " << stats.total_set_checks << std::endl;
std::cout << "Total inserts: " << stats.total_set_inserts << std::endl;
```

---
