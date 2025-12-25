#!/bin/bash
# Script to remove redundant scripts AFTER current tests complete
# DO NOT RUN THIS WHILE TESTS ARE RUNNING!
# 
# This script will:
# 1. Backup all redundant scripts
# 2. Remove redundant scripts from shell/ and build/
# 3. Keep only: test_motivation.sh and monitor_all.sh

echo "=== WARNING ==="
echo "This script will remove redundant test and monitor scripts."
echo "Make sure all tests have completed before running this!"
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

cd /workspace/OOD-ANNS/NGFix

# Check if any tests are running
RUNNING_TESTS=$(ps aux | grep -E "(test_ngfix_motivation|test_insertion_percentage|test_deletion_percentage)" | grep -v grep | wc -l)
if [ $RUNNING_TESTS -gt 0 ]; then
    echo "⚠ WARNING: There are $RUNNING_TESTS test processes still running!"
    echo "Please wait for tests to complete before removing scripts."
    exit 1
fi

# Backup directory
BACKUP_DIR="./backup_redundant_scripts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo ""
echo "=== Backing up redundant scripts ==="

# Redundant scripts in shell/
SHELL_REDUNDANT=(
    "shell/test_ngfix_motivation.sh"
    "shell/run_motivation_test_nohup.sh"
    "shell/check_motivation_test.sh"
    "shell/final_check_and_report.sh"
    "shell/wait_and_generate_report.sh"
    "shell/continuous_monitor.sh"
    "shell/monitor_efs1000.sh"
    "shell/monitor_motivation_test.sh"
    "shell/monitor_noise_test.sh"
    "shell/monitor_tests.sh"
    "shell/monitor_tests_unified.sh"
    "shell/final_monitor.sh"
)

# Redundant scripts in build/
BUILD_REDUNDANT=(
    "build/monitor_efs1000.sh"
    "build/monitor_noise_test.sh"
    "build/monitor_tests.sh"
    "build/final_monitor.sh"
)

# Backup shell scripts
for script in "${SHELL_REDUNDANT[@]}"; do
    if [ -f "$script" ]; then
        mkdir -p "$BACKUP_DIR/shell"
        cp "$script" "$BACKUP_DIR/shell/"
        echo "  Backed up: $script"
    fi
done

# Backup build scripts
for script in "${BUILD_REDUNDANT[@]}"; do
    if [ -f "$script" ]; then
        mkdir -p "$BACKUP_DIR/build"
        cp "$script" "$BACKUP_DIR/build/"
        echo "  Backed up: $script"
    fi
done

echo ""
echo "=== Removing redundant scripts ==="

# Remove shell scripts
for script in "${SHELL_REDUNDANT[@]}"; do
    if [ -f "$script" ]; then
        rm "$script"
        echo "  Removed: $script"
    fi
done

# Remove build scripts
for script in "${BUILD_REDUNDANT[@]}"; do
    if [ -f "$script" ]; then
        rm "$script"
        echo "  Removed: $script"
    fi
done

echo ""
echo "=== Cleanup complete ==="
echo "Backups saved in: $BACKUP_DIR"
echo ""
echo "Remaining scripts:"
echo "  - shell/test_motivation.sh (unified test script)"
echo "  - shell/monitor_all.sh (unified monitor script)"
echo ""
echo "To restore backups, copy files from: $BACKUP_DIR"

