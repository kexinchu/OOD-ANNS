#!/bin/bash
# Unified Monitor Script for All Tests
# Usage:
#   ./monitor_all.sh [--motivation] [--insertion] [--deletion] [--noise] [--efs1000] [--status] [--all]
#   --motivation: Monitor motivation test
#   --insertion: Monitor insertion test
#   --deletion: Monitor deletion test
#   --noise: Monitor noise test
#   --efs1000: Monitor efs1000 test
#   --status: Show only status (quick check)
#   --all: Monitor all tests (default if no specific test given)
#   --efs VALUE: Specify efSearch value for motivation test (default: 100)

SHOW_MOTIVATION=false
SHOW_INSERTION=false
SHOW_DELETION=false
SHOW_NOISE=false
SHOW_EFS1000=false
QUICK_STATUS=false
MOTIVATION_EFS=100

# Parse arguments
if [ $# -eq 0 ]; then
    # Default: show all
    SHOW_MOTIVATION=true
    SHOW_INSERTION=true
    SHOW_DELETION=true
    SHOW_NOISE=true
    SHOW_EFS1000=true
else
    while [[ $# -gt 0 ]]; do
        case $1 in
            --motivation)
                SHOW_MOTIVATION=true
                shift
                ;;
            --insertion)
                SHOW_INSERTION=true
                shift
                ;;
            --deletion)
                SHOW_DELETION=true
                shift
                ;;
            --noise)
                SHOW_NOISE=true
                shift
                ;;
            --efs1000)
                SHOW_EFS1000=true
                shift
                ;;
            --all)
                SHOW_MOTIVATION=true
                SHOW_INSERTION=true
                SHOW_DELETION=true
                SHOW_NOISE=true
                SHOW_EFS1000=true
                shift
                ;;
            --status)
                QUICK_STATUS=true
                shift
                ;;
            --efs)
                MOTIVATION_EFS="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: $0 [--motivation] [--insertion] [--deletion] [--noise] [--efs1000] [--status] [--all] [--efs VALUE]"
                exit 1
                ;;
        esac
    done
fi

echo "=========================================="
echo "Unified Test Monitor"
echo "=========================================="
echo ""

# Motivation Test Monitor
if [ "$SHOW_MOTIVATION" = true ]; then
    echo "Motivation Test (efSearch=${MOTIVATION_EFS}):"
    echo "=========================================="
    
    RESULT_CSV="/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_results_efs${MOTIVATION_EFS}.csv"
    LOG_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_nohup_efs${MOTIVATION_EFS}.log"
    EXPECTED_STAGES=12
    
    # Check if process is running
    if ps aux | grep -v grep | grep -q "test_ngfix_motivation"; then
        PID=$(ps aux | grep -v grep | grep "test_ngfix_motivation" | awk '{print $2}' | head -1)
        echo "✓ Status: RUNNING (PID: $PID)"
    else
        echo "✗ Status: NOT running"
    fi
    
    if [ "$QUICK_STATUS" = false ]; then
        # Check result file
        if [ -f "$RESULT_CSV" ]; then
            COMPLETED=$(tail -n +2 "$RESULT_CSV" 2>/dev/null | wc -l)
            PERCENTAGE=$((COMPLETED * 100 / EXPECTED_STAGES))
            echo "Progress: $COMPLETED/$EXPECTED_STAGES stages ($PERCENTAGE%)"
            
            if [ $COMPLETED -gt 0 ]; then
                LAST_STAGE=$(tail -1 "$RESULT_CSV" | cut -d',' -f1)
                echo "Last completed: $LAST_STAGE"
                echo ""
                echo "Recent results:"
                tail -3 "$RESULT_CSV" | column -t -s','
            fi
        else
            echo "Result file not found yet"
        fi
        
        # Show log if available
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Latest log (last 5 lines):"
            tail -5 "$LOG_FILE" 2>/dev/null || echo "No log content"
        fi
    fi
    echo ""
fi

# Insertion Test Monitor
if [ "$SHOW_INSERTION" = true ]; then
    echo "Insertion Test:"
    echo "==============="
    
    SUMMARY_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results/summary.csv"
    if [ ! -f "$SUMMARY_FILE" ]; then
        # Try alternative path
        SUMMARY_FILE="/workspace/NGFix/data/t2i-10M/insertion_percentage_results/summary.csv"
    fi
    
    if [ -f "$SUMMARY_FILE" ]; then
        if [ "$QUICK_STATUS" = false ]; then
            echo "Current results:"
            cat "$SUMMARY_FILE"
        fi
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo "Progress: $COMPLETED / 9 tests completed"
        
        # Check running process
        if ps aux | grep -v grep | grep -q "test_insertion_percentage"; then
            echo "✓ Status: RUNNING"
        else
            echo "✗ Status: NOT running"
        fi
    else
        echo "Summary file not found yet"
        if ps aux | grep -v grep | grep -q "test_insertion_percentage"; then
            echo "✓ Status: RUNNING (no results yet)"
        else
            echo "✗ Status: NOT running"
        fi
    fi
    echo ""
fi

# Deletion Test Monitor
if [ "$SHOW_DELETION" = true ]; then
    echo "Deletion Test:"
    echo "================"
    
    SUMMARY_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/deletion_percentage_results/summary.csv"
    if [ ! -f "$SUMMARY_FILE" ]; then
        # Try alternative path
        SUMMARY_FILE="/workspace/NGFix/data/t2i-10M/deletion_percentage_results/summary.csv"
    fi
    
    if [ -f "$SUMMARY_FILE" ]; then
        if [ "$QUICK_STATUS" = false ]; then
            echo "Current results:"
            cat "$SUMMARY_FILE"
        fi
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo "Progress: $COMPLETED / 9 tests completed"
        
        # Check running process
        if ps aux | grep -v grep | grep -q "test_deletion_percentage"; then
            echo "✓ Status: RUNNING"
        else
            echo "✗ Status: NOT running"
        fi
    else
        echo "Summary file not found yet"
        if ps aux | grep -v grep | grep -q "test_deletion_percentage"; then
            echo "✓ Status: RUNNING (no results yet)"
        else
            echo "✗ Status: NOT running"
        fi
    fi
    echo ""
fi

# Noise Test Monitor
if [ "$SHOW_NOISE" = true ]; then
    echo "Noise Test (noise_scale=0.01):"
    echo "=============================="
    
    RESULTS_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs1000_noise0.01"
    if [ ! -d "$RESULTS_DIR" ]; then
        RESULTS_DIR="/workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000_noise0.01"
    fi
    SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
    
    if [ -f "$SUMMARY_FILE" ]; then
        if [ "$QUICK_STATUS" = false ]; then
            echo "Current results:"
            cat "$SUMMARY_FILE"
        fi
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo "Progress: $COMPLETED / 9 tests completed"
        
        # Comparison with no noise
        if [ "$QUICK_STATUS" = false ]; then
            NO_NOISE_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv"
            if [ ! -f "$NO_NOISE_FILE" ]; then
                NO_NOISE_FILE="/workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv"
            fi
            if [ -f "$NO_NOISE_FILE" ]; then
                echo ""
                echo "Comparison with no noise (efSearch=1000):"
                head -4 "$NO_NOISE_FILE"
            fi
        fi
        
        # Check running process
        if ps aux | grep -v grep | grep -q "test_insertion_percentage"; then
            echo "✓ Status: RUNNING"
        else
            echo "✗ Status: NOT running"
        fi
    else
        echo "Summary file not found yet"
        if ps aux | grep -v grep | grep -q "test_insertion_percentage"; then
            echo "✓ Status: RUNNING (no results yet)"
        else
            echo "✗ Status: NOT running"
        fi
    fi
    echo ""
fi

# EFS1000 Test Monitor
if [ "$SHOW_EFS1000" = true ]; then
    echo "Insertion Test (efSearch=1000):"
    echo "==============================="
    
    SUMMARY_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv"
    if [ ! -f "$SUMMARY_FILE" ]; then
        # Try alternative path
        SUMMARY_FILE="/workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv"
    fi
    
    if [ -f "$SUMMARY_FILE" ]; then
        if [ "$QUICK_STATUS" = false ]; then
            echo "Current results:"
            cat "$SUMMARY_FILE"
        fi
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo "Progress: $COMPLETED / 9 tests completed"
        
        # Check running process
        if ps aux | grep -v grep | grep -q "test_insertion_percentage"; then
            echo "✓ Status: RUNNING"
        else
            echo "✗ Status: NOT running"
        fi
    else
        echo "Summary file not found yet"
        if ps aux | grep -v grep | grep -q "test_insertion_percentage"; then
            echo "✓ Status: RUNNING (no results yet)"
        else
            echo "✗ Status: NOT running"
        fi
    fi
    echo ""
fi

# Summary of all running processes
if [ "$QUICK_STATUS" = false ]; then
    echo "All Running Test Processes:"
    echo "============================"
    RUNNING=$(ps aux | grep -E "(test_ngfix_motivation|test_insertion_percentage|test_deletion_percentage)" | grep -v grep)
    if [ -z "$RUNNING" ]; then
        echo "No test processes running"
    else
        echo "$RUNNING" | head -5
    fi
    echo ""
    
    # Show log files if available
    echo "Recent Log Activity:"
    echo "===================="
    if [ -f "/tmp/insertion_final.log" ]; then
        echo "Insertion (last 2 lines):"
        tail -2 /tmp/insertion_final.log 2>/dev/null || echo "No log content"
    fi
    if [ -f "/tmp/deletion_final.log" ]; then
        echo "Deletion (last 2 lines):"
        tail -2 /tmp/deletion_final.log 2>/dev/null || echo "No log content"
    fi
    if [ -f "/tmp/insertion_efs1000.log" ]; then
        echo "Insertion efs1000 (last 2 lines):"
        tail -2 /tmp/insertion_efs1000.log 2>/dev/null || echo "No log content"
    fi
    if [ -f "/tmp/insertion_with_noise.log" ]; then
        echo "Insertion with noise (last 2 lines):"
        tail -2 /tmp/insertion_with_noise.log 2>/dev/null || echo "No log content"
    fi
fi

