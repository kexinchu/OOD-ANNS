# Unified Test and Monitor Scripts

## Overview

This directory contains unified scripts for running and monitoring NGFix tests.

## Main Scripts

### 1. `test_motivation.sh` - Unified Motivation Test Script

Run the NGFix motivation test (deletes 20% addition edges, tests, and rebuilds incrementally).

**Usage:**
```bash
# Run test with default settings (efSearch=100, foreground)
./shell/test_motivation.sh

# Run with custom efSearch value
./shell/test_motivation.sh --efs 200

# Run in background with nohup
./shell/test_motivation.sh --nohup

# Run with custom efSearch in background
./shell/test_motivation.sh --efs 200 --nohup
```

**Options:**
- `--efs VALUE`: Set efSearch value (default: 100)
- `--nohup`: Run test in background with nohup
- `--test-mode`: Use small dataset for testing (requires code modification)

**Output:**
- CSV results: `data/t2i-10M/motivation_test_results/motivation_test_results_efs{VALUE}.csv`
- Markdown report: `data/t2i-10M/motivation_test_results/motivation_test_report_efs{VALUE}.md`
- Log file (if --nohup): `data/t2i-10M/motivation_test_results/motivation_test_nohup_efs{VALUE}.log`

### 2. `monitor_all.sh` - Unified Monitor Script

Monitor all NGFix tests (motivation, insertion, deletion, noise, efs1000).

**Usage:**
```bash
# Monitor all tests (default)
./shell/monitor_all.sh

# Monitor specific test
./shell/monitor_all.sh --motivation
./shell/monitor_all.sh --insertion
./shell/monitor_all.sh --deletion
./shell/monitor_all.sh --noise
./shell/monitor_all.sh --efs1000

# Quick status check (no detailed output)
./shell/monitor_all.sh --status
./shell/monitor_all.sh --motivation --status

# Monitor motivation test with specific efSearch value
./shell/monitor_all.sh --motivation --efs 100
```

**Options:**
- `--motivation`: Monitor motivation test
- `--insertion`: Monitor insertion test
- `--deletion`: Monitor deletion test
- `--noise`: Monitor noise test
- `--efs1000`: Monitor efs1000 test
- `--all`: Monitor all tests (default if no specific test given)
- `--status`: Show only status (quick check)
- `--efs VALUE`: Specify efSearch value for motivation test (default: 100)

## Examples

### Run Motivation Test
```bash
# Run test in foreground
cd /workspace/OOD-ANNS/NGFix
./shell/test_motivation.sh --efs 100

# Run test in background
./shell/test_motivation.sh --efs 100 --nohup

# Monitor progress
./shell/monitor_all.sh --motivation --efs 100
```

### Monitor All Tests
```bash
# Check status of all tests
./shell/monitor_all.sh --status

# Detailed monitoring
./shell/monitor_all.sh --all
```

## Cleanup

After all tests complete, you can remove redundant scripts:

```bash
# This will backup and remove redundant scripts
./shell/REMOVE_AFTER_TEST_COMPLETE.sh
```

**Warning:** Only run this after all tests have completed! It will check for running tests and abort if any are found.

## Migration from Old Scripts

Old scripts have been replaced:

| Old Script | New Script |
|------------|------------|
| `test_ngfix_motivation.sh` | `test_motivation.sh` |
| `run_motivation_test_nohup.sh` | `test_motivation.sh --nohup` |
| `check_motivation_test.sh` | `monitor_all.sh --motivation` |
| `final_check_and_report.sh` | `monitor_all.sh --motivation` (then generate report) |
| `wait_and_generate_report.sh` | Use `monitor_all.sh` to check, then generate report |
| `continuous_monitor.sh` | `monitor_all.sh` (can be run in loop) |
| `monitor_efs1000.sh` | `monitor_all.sh --efs1000` |
| `monitor_tests.sh` | `monitor_all.sh --insertion --deletion` |
| `monitor_noise_test.sh` | `monitor_all.sh --noise` |

## Notes

- The unified scripts support both `/workspace/OOD-ANNS/NGFix` and `/workspace/NGFix` paths
- Test results are saved with efSearch value in filename for easy identification
- Monitor script automatically detects running processes and shows progress
- All scripts include error checking and helpful error messages

