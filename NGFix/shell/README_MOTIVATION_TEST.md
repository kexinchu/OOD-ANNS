# NGFix Motivation Test - 统一脚本使用说明

## 概述

已将所有motivation test相关的脚本合并为两个统一脚本：
1. `test_ngfix_motivation_unified.sh` - 测试执行脚本
2. `monitor_tests_unified.sh` - 监控脚本（整合了所有monitor功能）

## 测试脚本使用

### 基本用法

```bash
cd /workspace/OOD-ANNS/NGFix

# 运行测试（前台）
./shell/test_ngfix_motivation_unified.sh run --efs 100

# 后台运行测试
./shell/test_ngfix_motivation_unified.sh nohup --efs 100

# 检查测试状态
./shell/test_ngfix_motivation_unified.sh check --efs 100

# 生成最终报告（测试完成后）
./shell/test_ngfix_motivation_unified.sh report --efs 100
```

### 参数说明

- `run` - 在前台运行测试
- `check` - 检查测试状态和进度
- `report` - 生成最终Markdown报告（测试完成后）
- `nohup` - 在后台运行测试
- `--efs VALUE` - 设置efSearch值（默认100）
- `--test-size small|full` - 测试模式（small用于快速测试，full用于生产）

### 示例

```bash
# 使用小数据集快速测试脚本功能
./shell/test_ngfix_motivation_unified.sh run --efs 100 --test-size small

# 使用完整数据集运行（生产环境）
./shell/test_ngfix_motivation_unified.sh run --efs 100 --test-size full

# 后台运行并监控
./shell/test_ngfix_motivation_unified.sh nohup --efs 100
./shell/test_ngfix_motivation_unified.sh check --efs 100
```

## 监控脚本使用

### 基本用法

```bash
cd /workspace/OOD-ANNS/NGFix

# 监控所有测试
./shell/monitor_tests_unified.sh all

# 只监控motivation test
./shell/monitor_tests_unified.sh motivation --efs 100

# 监控insertion test
./shell/monitor_tests_unified.sh insertion --efs 1000

# 监控deletion test
./shell/monitor_tests_unified.sh deletion

# 监控noise test
./shell/monitor_tests_unified.sh noise
```

### 监控类型说明

- `all` - 监控所有测试类型
- `motivation` - 监控motivation test（需要--efs参数）
- `insertion` - 监控insertion test（可选--efs参数，默认1000）
- `deletion` - 监控deletion test
- `noise` - 监控noise test

## 输出文件

测试结果保存在：
- CSV: `/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_results_efs{efs}.csv`
- Markdown报告: `/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_report_efs{efs}.md`
- 日志: `/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_nohup.log`

## 注意事项

1. **不影响后台测试**：新脚本不会影响当前正在运行的测试进程
2. **路径兼容**：脚本自动处理不同的路径格式（/workspace/NGFix 和 /workspace/OOD-ANNS/NGFix）
3. **测试模式**：使用`--test-size small`可以快速测试脚本功能，但记得改回`full`用于实际测试

## 已废弃的脚本

以下脚本已被统一脚本替代，可以删除：
- `test_ngfix_motivation.sh` → 使用 `test_ngfix_motivation_unified.sh`
- `run_motivation_test_nohup.sh` → 使用 `test_ngfix_motivation_unified.sh nohup`
- `check_motivation_test.sh` → 使用 `test_ngfix_motivation_unified.sh check`
- `final_check_and_report.sh` → 使用 `test_ngfix_motivation_unified.sh report`
- `wait_and_generate_report.sh` → 功能已整合
- `continuous_monitor.sh` → 功能已整合
- `monitor_efs1000.sh` → 使用 `monitor_tests_unified.sh insertion --efs 1000`
- `monitor_tests.sh` → 使用 `monitor_tests_unified.sh all`
- `monitor_noise_test.sh` → 使用 `monitor_tests_unified.sh noise`

