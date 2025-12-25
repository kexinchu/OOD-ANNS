# Ours 项目代码与功能分析报告

## 项目概述

**Ours** 是一个针对 **OOD (Out-of-Distribution) ANNS (Approximate Nearest Neighbor Search)** 的优化实现，主要解决图索引在OOD查询时面临的精度下降和搜索开销增加的问题。该项目基于NGFix进行改进，实现了真正的runtime update和OOD查询优化。

## 核心问题与目标

### 要解决的问题
1. **OOD查询精度下降**：图索引在OOD查询时，recall下降，search开销增加
2. **多模态场景**：跨模态检索（如文本检索图像）使得OOD-ANNS场景更广泛
3. **现有方法的局限性**：
   - RoarGraph等方法需要大量历史查询离线构建，难以适配查询分布漂移
   - NGFix支持在线更新，但query选择逻辑是随机的，且需要partial rebuild

### 项目目标
- **真正的runtime update**：支持在线动态更新，无需partial rebuild
- **智能query选择**：选择合适数量和类型的hard query来指导图更新
- **分散式更新**：将insert/delete/update操作分散在时间线上
- **并发一致性**：保证读QPS > 写QPS时的并发安全性

## 核心功能模块

### 1. 低成本Hard Query检测

#### 1.1 两阶段检测机制

**Stage 0: 轻量级指标（Lightweight Metrics）**
- 从搜索轨迹中提取指标：
  - `r_visit`: 访问节点比例
  - `r_early`: 早期终止比例
  - `top1_last1_diff`: 最佳候选与最终结果的差异
- 计算hardness score：
  ```cpp
  hardness_score = r_visit * (1.0 - r_early) + top1_last1_diff * 0.1
  ```

**Stage 1: 扰动稳定性（Jitter）**
- 对查询向量添加小扰动：`q' = normalize(q + ε·noise)`, `ε ∈ [0.01, 0.05]`
- 计算结果稳定性：`J = 1 - |R_b(q) ∩ R_b(q')| / k`
- 阈值判断：`is_hard = (hardness_score > 0.5) || (jitter > 0.1)`

#### 1.2 ML预测器（可选）
- 使用LightGBM训练hardness预测器
- 特征包括：r_visit, r_early, d_worst_final, d_best_cand_final, progress_rate等
- 支持加权采样，增加hard query的训练权重

**实现位置**：
- `ourslib/graph/hnsw_ours.h`: `DetectHardQuery()` 函数
- `scripts/train_lightgbm_hardness_predictor.py`: ML模型训练
- `scripts/predict_lightgbm_hardness.py`: ML模型推理

### 2. 优化的EH（Escape Hardness）计算

#### 2.1 分组策略（Grouping Strategy）

**核心思想**：将2跳内可达的节点视为同一组，在分组图上计算EH，然后映射回原始节点空间。

**实现步骤**：
1. **节点分组**：使用BFS在2跳内找到连通组件
   ```cpp
   GroupNodesByReachability(searcher, node_ids, M, max_hops=2)
   ```

2. **构建分组图**：在分组级别构建邻接关系

3. **计算分组EH矩阵**：在分组图上计算EH（矩阵大小从M²降到m²，m << M）

4. **映射回原始空间**：将分组EH值映射到原始节点对

**性能提升**：
- 矩阵大小减少：60-85%（从~10000降到~1576）
- 计算时间减少：约1.58倍加速
- EH计算开销：从34.11%降到0.2-2.1%

**实现位置**：
- `ourslib/graph/hnsw_ours.h`: 
  - `GroupNodesByReachability()`: 节点分组
  - `CalculateHardnessGroupedWithMapping()`: 分组EH计算

#### 2.2 优化的边选择

**优化点**：
1. **距离缓存**：避免重复计算节点对距离
2. **跳过同组节点对**：分组后只计算跨组节点对的距离
3. **堆排序替代全排序**：使用max-heap选择EH值最大的边，避免O(N²logN)排序
4. **连通性检查**：只添加真正需要的边（未连通的节点对）

**实现位置**：
- `ourslib/graph/hnsw_ours.h`: `getDefectsFixingEdgesOptimized()`

### 3. 动态更新机制

#### 3.1 Lazy Delete（延迟删除）

**问题**：随着时间推移，以前新增的连通性可能"过期"，这些边会拖慢查询过程。

**解决方案**：采用"保护in-serve edge"的策略

**实现机制**：
1. **后台线程**：独立的后台线程管理lazy delete过程
2. **Epoch机制**：设定全局epoch（时间周期）和page_num（每批处理的页数）
3. **标记节点**：每个epoch开始时随机选择pages，标记其中的nodes为"待删除"
4. **记录in-serve edge**：在search过程中，当访问到标记的node时，记录被选中的edge到"in-serve"集合
5. **清理过期edge**：epoch结束时，删除不在in-serve集合中的additional edge

**优化**：
- **Bloom Filter**：快速负向查找（消除~90%的二分查找）
- **排序向量+二分查找**：cache-friendly的快速查找
- **线程本地缓冲区**：减少锁竞争
- **范围检查**：缓存min/max node ID，快速排除大部分检查

**数据结构**：
```cpp
std::unordered_set<id_t> pending_delete_nodes;  // 待删除节点集合
std::unordered_map<id_t, std::unordered_set<id_t>> in_serve_edges;  // in-serve边集合
std::atomic<std::vector<id_t>*> cached_pending_nodes_ptr;  // 缓存的排序向量
std::atomic<uint8_t> cached_bloom_filter[BLOOM_FILTER_SIZE];  // Bloom过滤器
```

**实现位置**：
- `ourslib/graph/hnsw_ours.h`: 
  - `StartLazyDelete()`: 启动lazy delete
  - `LazyDeleteWorker()`: 后台工作线程
  - `RecordInServeEdge()`: 记录in-serve边
  - `CleanupPendingDeleteEdges()`: 清理过期边

#### 3.2 Runtime Insert/Delete

**插入节点**：
- 支持在线插入新节点
- 使用NGFix方法优化新节点的连接

**删除节点**：
- 使用lazy delete机制
- 两跳局部扫描获取incoming edges：
  - 候选集合：`C = Γ(A) ∪ ⋃_{u∈Γ(A)} Γ(u)`（A的邻居 + 邻居的邻居）
  - 对C中每个节点检查其出邻接表是否包含A

**实现位置**：
- `ourslib/graph/hnsw_ours.h`: `addPoint()`, `deletePoint()`

### 4. 并发一致性

#### 4.1 删除节点保护

**问题**：runtime delete的node不能出现在ANNS返回结果中

**解决方案**：
- 使用`delete_ids`集合标记已删除节点
- 在search过程中检查节点是否被删除
- 使用`shared_mutex`保护删除操作

#### 4.2 读写冲突处理

**问题**：新增node/edge时的读写冲突

**解决方案**：
- 使用`std::shared_mutex`实现读写锁
- 每个节点有独立的锁：`std::vector<std::shared_mutex> node_locks`
- 读操作使用`shared_lock`，写操作使用`unique_lock`

**实现位置**：
- `ourslib/graph/hnsw_ours.h`: 所有图更新操作都使用适当的锁

### 5. SSD存储支持（可选）

**功能**：支持将索引和向量数据存储在SSD上，减少内存占用

**实现**：
- `SSDStorage`类：管理SSD上的索引和向量数据
- 页式管理：按页加载和刷新数据
- 脏页跟踪：只刷新修改过的页

**实现位置**：
- `ourslib/utils/ssd_storage.h` / `ssd_storage.cc`

## 代码结构

### 目录结构

```
Ours/
├── ourslib/                    # 核心库
│   ├── graph/
│   │   ├── hnsw_ours.h        # 主图索引实现（2423行）
│   │   └── node.h             # 节点定义
│   ├── metric/                # 距离度量
│   │   ├── l2.h               # L2距离
│   │   ├── ip.h               # 内积距离
│   │   └── rabitq.h           # RabitQ距离
│   └── utils/                 # 工具类
│       ├── search_list.h      # 搜索列表
│       ├── visited_list.h     # 访问列表
│       └── ssd_storage.h/cc   # SSD存储
├── test/                      # 测试程序
│   ├── test_ours.cc           # 主测试程序
│   ├── test_lhp_comparison.cc # LHP方法对比
│   ├── test_lazy_delete.cc    # Lazy delete测试
│   ├── test_hardness_predictor_tree.cc  # Hardness预测器测试
│   └── tools/                 # 测试工具
│       ├── data_loader.h      # 数据加载
│       └── result_evaluation.h # 结果评估
├── scripts/                   # Python脚本
│   ├── train_lightgbm_hardness_predictor.py  # 训练ML模型
│   ├── predict_lightgbm_hardness.py          # ML模型推理
│   └── visualize_hardness_predictor.py       # 可视化
└── data/                      # 数据和结果
    ├── t2i-10M/              # 10M数据集
    └── comparison/            # 对比结果
```

### 核心类：HNSW_Ours

**主要成员变量**：
```cpp
class HNSW_Ours<T> {
    // 图结构
    std::vector<ngfixlib::node> Graph;
    char* vecdata;  // 向量数据
    
    // 距离度量
    ngfixlib::Space<T>* space;
    ngfixlib::Space<T>* query_space;
    
    // 并发控制
    std::vector<std::shared_mutex> node_locks;  // 每个节点的锁
    std::shared_mutex delete_lock;
    std::unordered_set<id_t> delete_ids;
    
    // 动态更新
    std::unordered_map<id_t, std::vector<EdgeInfo>> added_edges;  // 添加的边
    std::atomic<size_t> current_timestamp;
    
    // Lazy delete
    std::unordered_set<id_t> pending_delete_nodes;
    std::unordered_map<id_t, std::unordered_set<id_t>> in_serve_edges;
    std::thread lazy_delete_thread;
    
    // SSD存储（可选）
    std::unique_ptr<SSDStorage> ssd_storage_;
};
```

**主要方法**：
- `searchKnn()`: 标准kNN搜索
- `searchKnnWithLightweightMetrics()`: 带轻量级指标的搜索
- `DetectHardQuery()`: 检测hard query
- `NGFixOptimized()`: 优化的NGFix方法
- `CalculateHardnessGroupedWithMapping()`: 分组EH计算
- `LHPOptimize()`: LHP方法优化
- `addPoint()`: 插入节点
- `deletePoint()`: 删除节点
- `StartLazyDelete()` / `StopLazyDelete()`: 启动/停止lazy delete

## 关键优化技术

### 1. 性能优化

| 优化点 | 方法 | 效果 |
|--------|------|------|
| EH计算 | 分组策略（2跳可达性） | 矩阵大小减少60-85%，计算时间减少1.58倍 |
| 边选择 | 距离缓存 + 堆排序 | 避免重复计算，减少排序开销 |
| Lazy delete检查 | Bloom Filter + 排序向量 | 消除~90%的查找开销 |
| 并发控制 | 细粒度锁 + 线程本地缓冲区 | 减少锁竞争 |

### 2. 内存优化

- **SSD存储**：可选地将索引和向量数据存储在SSD上
- **页式管理**：按需加载数据页
- **缓存策略**：缓存pending nodes的排序向量和Bloom filter

### 3. 算法优化

- **智能query选择**：只使用top 10% hardest queries进行优化
- **连通性检查**：只添加真正需要的边（未连通的节点对）
- **分组计算**：在分组图上计算EH，减少计算量

## 测试与评估

### 测试程序

1. **test_ours.cc**: 主测试程序
   - Hard query检测
   - 图优化
   - Recall和Latency评估

2. **test_lhp_comparison.cc**: LHP方法对比
   - LHP vs EH-Grouped vs NGFix-EH

3. **test_lazy_delete.cc**: Lazy delete测试
   - 不同epoch和batch_size组合的性能

4. **test_hardness_predictor_tree.cc**: Hardness预测器测试
   - ML模型效果评估

### 评估指标

- **Recall**: 召回率
- **Latency**: 查询延迟（P50, P95, P99）
- **QPS**: 每秒查询数
- **NDC**: 距离计算次数
- **边数**: 添加的边数量

## 与NGFix的主要区别

| 特性 | NGFix | Ours |
|------|-------|------|
| Query选择 | 随机选择（从最新query中） | 智能选择（hard query检测） |
| EH计算 | 完整计算O(M²) | 分组计算O(m²)，m << M |
| 边选择 | 全排序 | 堆排序 + 距离缓存 |
| 动态更新 | 需要partial rebuild | 真正的runtime update（lazy delete） |
| 并发安全 | 基本支持 | 优化的细粒度锁 + 线程本地缓冲区 |

## 使用示例

### 基本使用

```cpp
#include "ourslib/graph/hnsw_ours.h"

using namespace ours;

// 创建索引
HNSW_Ours<float>* index = new HNSW_Ours<float>(L2_float, index_path);

// 检测hard query
auto metrics = DetectHardQuery(index, query_data, k, ef, dim);
if(metrics.is_hard) {
    // 使用hard query优化图
    index->NGFixOptimized(query_data, gt, Nq, Kh);
}

// 启动lazy delete
index->StartLazyDelete(1000, 10);  // epoch=1s, pages=10

// 正常查询
auto results = index->searchKnn(query_data, k, ef);

// 停止lazy delete
index->StopLazyDelete();
```

### 运行测试

```bash
cd /workspace/OOD-ANNS/Ours
mkdir -p build && cd build
cmake ..
make -j$(nproc)

./test/test_ours \
    --base_index_path ../NGFix/data/t2i-10M/base.index \
    --train_query_path ../NGFix/data/t2i-10M/train_query.fbin \
    --test_query_path ../NGFix/data/t2i-10M/test_query.fbin \
    --metric ip_float \
    --K 100 \
    --num_test_queries 1000
```

## 性能数据

### EH计算优化效果

- **原始方法**：矩阵大小10000，计算时间371.54μs
- **分组方法**：矩阵大小1576.88，计算时间235.72μs
- **加速比**：1.58倍
- **矩阵大小减少**：84.23%

### LHP方法对比

| 方法 | Recall | Insert Latency | EH开销 |
|------|--------|----------------|--------|
| LHP | 0.9936 | 0.0065ms | 2.1% |
| EH-Grouped | 0.9936 | 0.0013ms | 0.2% |
| NGFix-EH | 0.9936 | 0.001ms | 7.7% |

## 未来工作方向

1. **Hardness预测器优化**：
   - 增加极端hard query的训练样本
   - 改进特征工程
   - 使用更强大的模型（XGBoost, 神经网络）

2. **动态更新优化**：
   - 更智能的edge过期判断
   - 自适应epoch和batch_size调整

3. **多模态支持**：
   - 模态条件距离函数
   - 跨模态锚点路由

4. **SSD存储优化**：
   - 更智能的预取策略
   - 压缩存储

## 总结

Ours项目是一个针对OOD-ANNS场景的全面优化实现，主要贡献包括：

1. **智能hard query检测**：两阶段检测机制（轻量级指标 + Jitter）
2. **优化的EH计算**：分组策略大幅减少计算开销
3. **真正的runtime update**：lazy delete机制支持动态更新
4. **并发安全**：细粒度锁和线程本地缓冲区保证高并发性能
5. **可选SSD存储**：支持大规模数据的内存友好存储

该项目在保持高recall的同时，显著降低了计算开销和更新延迟，为OOD-ANNS场景提供了实用的解决方案。

