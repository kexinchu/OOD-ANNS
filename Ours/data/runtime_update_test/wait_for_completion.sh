#!/bin/bash
RESULT_FILE="/workspace/OOD-ANNS/Ours/data/runtime_update_test/runtime_update_results_ngfix.csv"
TARGET_MINUTES=60

echo "=== 等待NGFix测试完成 ==="
echo "目标: $TARGET_MINUTES 分钟"
echo ""

while true; do
    if [ ! -f "$RESULT_FILE" ]; then
        echo "[$(date '+%H:%M:%S')] 等待结果文件..."
        sleep 60
        continue
    fi
    
    CURRENT_MINUTES=$(($(wc -l < "$RESULT_FILE" 2>/dev/null || echo 1) - 1))
    
    if [ $CURRENT_MINUTES -ge $TARGET_MINUTES ]; then
        echo ""
        echo "✅ 测试已完成！"
        echo "总分钟数: $CURRENT_MINUTES"
        echo ""
        echo "最终统计:"
        tail -5 "$RESULT_FILE"
        break
    fi
    
    echo "[$(date '+%H:%M:%S')] 进度: $CURRENT_MINUTES/$TARGET_MINUTES 分钟"
    
    # Check if process is still running
    if ! ps aux | grep test_runtime_update_end2end_ngfix | grep -v grep > /dev/null; then
        echo "WARNING: 测试进程已停止！"
        tail -20 /workspace/OOD-ANNS/Ours/data/runtime_update_test/nohup_ngfix.out
        exit 1
    fi
    
    sleep 60
done

echo ""
echo "测试监控完成！"
