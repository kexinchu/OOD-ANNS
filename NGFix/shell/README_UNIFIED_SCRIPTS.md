# Unified Test Scripts Documentation

## Overview

所有测试和监控脚本已合并为两个统一的脚本：
- `test_ngfix_motivation.sh` - 统一的测试执行脚本
- `monitor.sh` - 统一的监控脚本

## Test Script: test_ngfix_motivation.sh

### Usage

```bash
./shell/test_ngfix_motivation.sh [options]
```

### Options

- `--nohup`: 在后台运行测试（使用nohup）
- `--efs EFS_VALUE`: 指定efSearch值（默认: 100）
- `--test-size small`: 使用小数据集进行快速测试（需要修改代码限制训练查询数量）

### Examples

```bash
# 前台运行测试（efSearch=100）
./shell/test_ngfix_motivation.sh

# 后台运行测试（efSearch=100）
./shell/test_ngfix_motivation.sh --nohup

# 使用不同的efSearch值
./shell/test_ngfix_motivation.sh --nohup --efs 200

# 使用小数据集测试（需要先修改代码）
./shell/test_ngfix_motivation.sh --test-size small
```

### Output

- CSV结果文件: `data/t2i-10M/motivation_test_results/motivation_test_results_efs{efs}.csv`
- Markdown报告: `data/t2i-10M/motivation_test_results/motivation_test_report_efs{efs}.md`
- 日志文件: `data/t2i-10M/motivation_test_results/motivation_test_nohup.log` (如果使用--nohup)

## Monitor Script: monitor.sh

### Usage

```bash
./shell/monitor.sh [test_type] [options]
```

### Test Types

- `motivation` - 监控NGFix motivation测试
- `insertion` - 监控insertion percentage测试（efSearch=1000）
- `deletion` - 监控deletion percentage测试
- `noise` - 监控noise测试（noise_scale=0.01）
- `all` - 监控所有测试

### Options

- `--efs EFS_VALUE`: 对于motivation测试，指定efSearch值（默认: 100）
- `--status`: 仅显示当前状态（不进行持续监控）
- `--wait`: 等待测试完成（仅适用于motivation测试）
- `--interval SECONDS`: 持续监控的检查间隔（默认: 60秒）

### Examples

```bash
# 显示帮助信息
./shell/monitor.sh

# 检查motivation测试状态
./shell/monitor.sh motivation --efs 100 --status

# 持续监控motivation测试直到完成
./shell/monitor.sh motivation --efs 100 --wait

# 检查insertion测试状态
./shell/monitor.sh insertion

# 监控所有测试
./shell/monitor.sh all
```

## Removed Redundant Scripts

以下冗余脚本已被删除，功能已合并到统一脚本中：

### Motivation Test Scripts (已删除)
- `check_motivation_test.sh` → 使用 `monitor.sh motivation --status`
- `monitor_motivation_test.sh` → 使用 `monitor.sh motivation --wait`
- `run_motivation_test_nohup.sh` → 使用 `test_ngfix_motivation.sh --nohup`
- `wait_and_generate_report.sh` → 使用 `monitor.sh motivation --wait`
- `final_check_and_report.sh` → 使用 `monitor.sh motivation --status`
- `continuous_monitor.sh` → 使用 `monitor.sh motivation --wait`

### Monitor Scripts (已删除)
- `build/monitor_efs1000.sh` → 使用 `monitor.sh insertion`
- `build/monitor_tests.sh` → 使用 `monitor.sh all`
- `build/monitor_noise_test.sh` → 使用 `monitor.sh noise`

## Migration Guide

### 旧脚本 → 新脚本

| 旧脚本 | 新脚本 |
|--------|--------|
| `./shell/run_motivation_test_nohup.sh` | `./shell/test_ngfix_motivation.sh --nohup` |
| `./shell/check_motivation_test.sh` | `./shell/monitor.sh motivation --efs 100 --status` |
| `./shell/monitor_motivation_test.sh` | `./shell/monitor.sh motivation --efs 100 --wait` |
| `./build/monitor_efs1000.sh` | `./shell/monitor.sh insertion` |
| `./build/monitor_tests.sh` | `./shell/monitor.sh all` |
| `./build/monitor_noise_test.sh` | `./shell/monitor.sh noise` |

## Notes

- 所有脚本都支持通过参数自定义配置
- 后台运行的测试不会因为脚本删除而中断
- 使用 `--wait` 选项可以持续监控直到测试完成
- 测试完成后会自动生成Markdown报告

