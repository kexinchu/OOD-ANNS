#!/bin/bash

# Monitor comprehensive deletion test progress

RESULT_DIR="/workspace/OOD-ANNS/NGFix/data/t2i-10M/comprehensive_deletion_results"
RESULT_CSV="${RESULT_DIR}/comprehensive_deletion_results.csv"
LOG_FILE="/workspace/OOD-ANNS/NGFix/comprehensive_deletion_test.log"

echo "=== Monitoring Comprehensive Deletion Test ==="
echo "Result CSV: ${RESULT_CSV}"
echo "Log file: ${LOG_FILE}"
echo ""

# Check if test is running
if pgrep -f "test_comprehensive_deletion_comparison" > /dev/null; then
    echo "✅ Test is running"
    ps aux | grep test_comprehensive_deletion_comparison | grep -v grep | head -2
else
    echo "❌ Test is not running"
fi

echo ""
echo "=== Current Results ==="
if [ -f "${RESULT_CSV}" ]; then
    echo "Total lines in CSV: $(wc -l < ${RESULT_CSV})"
    echo ""
    echo "Latest results:"
    tail -10 ${RESULT_CSV} | column -t -s','
    echo ""
    echo "Progress by stage:"
    echo "Initial: $(grep -c '^initial,' ${RESULT_CSV} 2>/dev/null || echo 0) / 7"
    echo "Lazy deletion: $(grep -c '^lazy_deletion,' ${RESULT_CSV} 2>/dev/null || echo 0) / 7"
    echo "Real deletion: $(grep -c '^real_deletion,' ${RESULT_CSV} 2>/dev/null || echo 0) / 7"
else
    echo "CSV file not created yet"
fi

echo ""
echo "=== Recent Log Activity ==="
if [ -f "${LOG_FILE}" ]; then
    tail -20 ${LOG_FILE}
else
    echo "Log file not found"
fi

