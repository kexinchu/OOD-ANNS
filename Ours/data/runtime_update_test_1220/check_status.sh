#!/bin/bash
echo "=== NGFix测试状态 ==="
echo "时间: $(date)"
echo ""
ps aux | grep test_runtime_update_end2end_ngfix | grep -v grep
echo ""
MINUTES=$(($(wc -l < data/runtime_update_test/runtime_update_results_ngfix.csv 2>/dev/null || echo 1) - 1))
echo "已完成分钟数: $MINUTES/60"
echo ""
echo "最新统计数据:"
tail -3 data/runtime_update_test/runtime_update_results_ngfix.csv 2>/dev/null
echo ""
echo "最新日志:"
tail -5 data/runtime_update_test/nohup_ngfix.out 2>/dev/null | tail -3
