#!/bin/bash

# Test filter performance: hash map vs Bloom filter

cd /workspace/OOD-ANNS/Ours/build

echo "=== Building Filter Performance Test ==="
make test_filter_performance -j$(nproc)

echo ""
echo "=== Running Filter Performance Test ==="
./test/test_filter_performance

