# Lazy Delete Test Results

## 测试文件说明

### 1. set_update_overhead.csv
不同大小的set在更新时的开销（无锁情况）
- `set_size`: Set的大小
- `avg_update_time_us`: 平均更新时间（微秒）

### 2. set_update_overhead_with_lock.csv
不同大小的set在更新时的开销（有锁情况，模拟实际场景）
- `set_size`: Set的大小
- `avg_update_time_us`: 平均更新时间（微秒）

### 3. lazy_delete_results.csv
不同epoch + batch_size组合的lazy delete性能测试结果
- `epoch_duration_ms`: Epoch持续时间（毫秒）
- `batch_size`: 批次大小（节点数）
- `num_queries`: 查询数量
- `avg_query_time_with_pending_ms`: 有待删除nodes时的平均查询时间（毫秒）
- `avg_query_time_without_pending_ms`: 没有待删除nodes时的平均查询时间（毫秒）
- `overhead_ratio`: 开销比例
- `total_queries_hitting_pending_nodes`: 访问到pending nodes的查询数量
- `total_queries_not_hitting_pending_nodes`: 未访问到pending nodes的查询数量
- `avg_pending_nodes_count`: 平均pending nodes数量
- `max_pending_nodes_count`: 最大pending nodes数量
- `min_pending_nodes_count`: 最小pending nodes数量
- `avg_set_update_time_us`: 平均set更新时间（微秒）
- `avg_set_update_time_when_accessed_us`: 访问到pending nodes时的平均set更新时间（微秒）
- `total_set_updates`: 总set更新次数
- `total_set_updates_when_accessed`: 访问到pending nodes时的set更新次数
- `total_set_updates_when_not_accessed`: 未访问到pending nodes时的set更新次数
- `total_nodes_marked`: 总标记的nodes数量
- `total_in_serve_edges`: 总in-serve edges数量
- `total_edges_removed`: 总删除的edges数量
- `total_edges_accessed`: 总访问的edges数量

### 4. analysis_report.md
详细的分析报告，包括：
- Set更新开销分析
- Lazy delete性能分析
- 不同参数组合的影响
- 性能优化建议

## 关键发现

1. **Set更新开销很小**：即使在有锁的情况下，set更新开销也只有0.1-0.2微秒，相对于查询时间（1-2毫秒）来说可以忽略不计。

2. **访问pending nodes的查询比例很低**：在1000个查询中，只有5-18个查询真正访问了pending nodes，说明lazy delete的开销很小。

3. **性能提升**：在某些情况下，清理不必要的edges反而提高了性能（负开销），这是因为清理了未使用的additional edges，减少了搜索过程中的遍历开销。

4. **推荐参数**：
   - `epoch_duration_ms`: 1000-2000ms
   - `batch_size`: 100-200 nodes

## 使用方法

查看结果：
```bash
cat /workspace/OOD-ANNS/Ours/data/lazy_delete_test/lazy_delete_results.csv
cat /workspace/OOD-ANNS/Ours/data/lazy_delete_test/analysis_report.md
```

重新运行测试：
```bash
cd /workspace/OOD-ANNS/Ours
./build/test/test_lazy_delete \
    --base_index_path /workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M16_efC500_MEX48_AKNN1500_8M.index \
    --query_data_path /workspace/RoarGraph/data/t2i-10M/query.10k.fbin \
    --num_queries 1000 \
    --k 100 \
    --ef_search 200 \
    --output_dir /workspace/OOD-ANNS/Ours/data/lazy_delete_test
```

