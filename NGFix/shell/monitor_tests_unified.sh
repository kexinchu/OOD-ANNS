#!/bin/bash
# Unified Test Monitor Script
# Usage: ./monitor_tests_unified.sh [motivation|insertion|deletion|noise|all] [--efs VALUE]

MONITOR_TYPE="all"  # motivation, insertion, deletion, noise, all
efs=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        motivation|insertion|deletion|noise|all)
            MONITOR_TYPE="$1"
            shift
            ;;
        --efs)
            efs="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [motivation|insertion|deletion|noise|all] [--efs VALUE]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Unified Test Progress Monitor"
echo "=========================================="
echo ""

# Function: Monitor motivation test
monitor_motivation() {
    local efs_val=${efs:-100}
    local RESULT_CSV="/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_results_efs${efs_val}.csv"
    local LOG_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/motivation_test_results/motivation_test_nohup.log"
    local EXPECTED_STAGES=12

    echo "Motivation Test (efSearch=${efs_val}):"
    echo "========================================"
    
    # Check if process is running
    if ps aux | grep -v grep | grep -q "test_ngfix_motivation"; then
        echo "Status: ✓ RUNNING"
        PID=$(ps aux | grep -v grep | grep "test_ngfix_motivation" | awk '{print $2}' | head -1)
        echo "  PID: $PID"
    else
        echo "Status: ✗ NOT running"
    fi
    
    echo ""
    
    # Check results
    if [ -f "$RESULT_CSV" ]; then
        COMPLETED=$(tail -n +2 "$RESULT_CSV" 2>/dev/null | wc -l)
        PERCENTAGE=$((COMPLETED * 100 / EXPECTED_STAGES))
        echo "Progress: $COMPLETED / $EXPECTED_STAGES stages ($PERCENTAGE%)"
        
        if [ $COMPLETED -gt 0 ]; then
            echo ""
            echo "Last completed stage:"
            tail -1 "$RESULT_CSV" | cut -d',' -f1
            echo ""
            echo "Recent results (last 3):"
            tail -3 "$RESULT_CSV" | column -t -s',' 2>/dev/null || tail -3 "$RESULT_CSV"
        fi
    else
        echo "Results file not found yet"
    fi
    
    echo ""
    
    # Show log
    if [ -f "$LOG_FILE" ]; then
        echo "Latest log (last 5 lines):"
        tail -5 "$LOG_FILE"
    fi
    
    echo ""
}

# Function: Monitor insertion test
monitor_insertion() {
    local efs_val=${efs:-1000}
    # Try both possible paths
    local RESULTS_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs${efs_val}"
    if [ ! -d "$RESULTS_DIR" ]; then
        RESULTS_DIR="/workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs${efs_val}"
    fi
    local SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
    
    echo "Insertion Test (efSearch=${efs_val}):"
    echo "====================================="
    
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Current results:"
        cat "$SUMMARY_FILE"
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo ""
        echo "Progress: $COMPLETED / 9 tests completed"
    else
        echo "Summary file not found yet"
    fi
    
    echo ""
    echo "Index files created:"
    ls -1 "${RESULTS_DIR}/index_insert_"*.index 2>/dev/null | wc -l | xargs echo
    
    echo ""
    echo "Running processes:"
    ps aux | grep "test_insertion_percentage" | grep -v grep | head -1 || echo "None"
    
    echo ""
    echo "Latest log:"
    tail -5 /tmp/insertion_efs${efs_val}.log 2>/dev/null || echo "No log yet"
    
    echo ""
}

# Function: Monitor deletion test
monitor_deletion() {
    echo "Deletion Test:"
    echo "=============="
    
    # Try both possible paths
    local RESULTS_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/deletion_percentage_results"
    if [ ! -d "$RESULTS_DIR" ]; then
        RESULTS_DIR="/workspace/NGFix/data/t2i-10M/deletion_percentage_results"
    fi
    local SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
    
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Current results:"
        cat "$SUMMARY_FILE"
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo ""
        echo "Progress: $COMPLETED / 9 tests completed"
    else
        echo "Summary file not found yet"
    fi
    
    echo ""
    echo "Index files created:"
    ls -1 "${RESULTS_DIR}/index_lazy_delete_"*.index 2>/dev/null | wc -l | xargs echo
    
    echo ""
    echo "Running processes:"
    ps aux | grep "test_deletion_percentage" | grep -v grep | head -1 || echo "None"
    
    echo ""
    echo "Latest log:"
    tail -5 /tmp/deletion_final.log 2>/dev/null || echo "No log yet"
    
    echo ""
}

# Function: Monitor noise test
monitor_noise() {
    local noise_scale="0.01"
    # Try both possible paths
    local RESULTS_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs1000_noise${noise_scale}"
    if [ ! -d "$RESULTS_DIR" ]; then
        RESULTS_DIR="/workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000_noise${noise_scale}"
    fi
    local SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
    
    echo "Insertion Test with Noise (noise_scale=${noise_scale}, efSearch=1000):"
    echo "======================================================================"
    
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Current results:"
        cat "$SUMMARY_FILE"
        COMPLETED=$(tail -n +2 "$SUMMARY_FILE" 2>/dev/null | wc -l)
        echo ""
        echo "Progress: $COMPLETED / 9 tests completed"
    else
        echo "Summary file not found yet"
    fi
    
    echo ""
    echo "Comparison with no noise (efSearch=1000):"
    # Try both possible paths
    local NO_NOISE_FILE="/workspace/OOD-ANNS/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv"
    if [ ! -f "$NO_NOISE_FILE" ]; then
        NO_NOISE_FILE="/workspace/NGFix/data/t2i-10M/insertion_percentage_results_efs1000/summary.csv"
    fi
    if [ -f "$NO_NOISE_FILE" ]; then
        echo "No noise results (first 4 lines):"
        head -4 "$NO_NOISE_FILE"
    else
        echo "No noise results file not found"
    fi
    
    echo ""
    echo "Running processes:"
    ps aux | grep "test_insertion_percentage" | grep -v grep | head -1 || echo "None"
    
    echo ""
    echo "Latest log:"
    tail -5 /tmp/insertion_with_noise.log 2>/dev/null || echo "No log yet"
    
    echo ""
}

# Main execution
case $MONITOR_TYPE in
    motivation)
        monitor_motivation
        ;;
    insertion)
        monitor_insertion
        ;;
    deletion)
        monitor_deletion
        ;;
    noise)
        monitor_noise
        ;;
    all)
        monitor_motivation
        echo ""
        echo "----------------------------------------"
        echo ""
        monitor_insertion
        echo ""
        echo "----------------------------------------"
        echo ""
        monitor_deletion
        echo ""
        echo "----------------------------------------"
        echo ""
        monitor_noise
        ;;
    *)
        echo "Unknown monitor type: $MONITOR_TYPE"
        echo "Usage: $0 [motivation|insertion|deletion|noise|all] [--efs VALUE]"
        exit 1
        ;;
esac

echo "=========================================="
echo "Monitor completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

