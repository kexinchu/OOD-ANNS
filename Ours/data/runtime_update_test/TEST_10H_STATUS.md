# 10小时NGFix测试状态

## 测试配置
- **测试开始时间**: 2025-12-16 00:22
- **预计运行时长**: 600分钟（10小时）
- **Search QPS**: 400
- **Insert QPS**: 100
- **ef_search**: 1000
- **K**: 100

## 数据配置
- **Base Index**: ./data/comparison_10M/base.index (10M elements)
- **Train Queries**: /workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin
- **Train GT**: /workspace/RoarGraph/data/t2i-10M/train.gt.bin
- **Additional Vectors**: /workspace/RoarGraph/data/t2i-10M/base.additional.10M.fbin

## 索引Resize
- **原始大小**: 10M elements
- **Resize后**: 13.6M elements (允许3.6M inserts)

## 监控命令
```bash
# 查看实时日志
tail -f /workspace/OOD-ANNS/Ours/data/runtime_update_test/nohup_ngfix.out

# 查看统计结果
cat /workspace/OOD-ANNS/Ours/data/runtime_update_test/runtime_update_results_ngfix.csv

# 检查进程状态
ps aux | grep test_runtime_update_end2end_ngfix | grep -v grep

# 检查进度
tail -n +2 /workspace/OOD-ANNS/Ours/data/runtime_update_test/runtime_update_results_ngfix.csv | wc -l

# 查看监控日志
tail -f /workspace/OOD-ANNS/Ours/data/runtime_update_test/monitor_10h.log
```

## 预期结果
- 总查询数: ~14,400,000 (400 QPS × 600分钟 × 60秒)
- 总插入数: ~3,600,000 (100 QPS × 600分钟 × 60秒)
- 统计数据: 600行（每分钟一行）
