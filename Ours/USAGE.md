# Ours Implementation Usage Guide

## Overview

This is an optimized version of NGFix with the following improvements:

1. **Low-cost hard query detection**: Uses lightweight metrics (hardness predictor) + Jitter (perturbation stability)
2. **Optimized EH calculation**: Uses grouping strategy to reduce computation (2-hop reachability grouping)
3. **Optimized edge fixing**: Avoids full sort, uses heap-based approach
4. **Dynamic update**: Supports removing expired edges over time
5. **Concurrency safety**: Ensures deleted nodes don't appear in search results

## Building

```bash
cd /workspace/OOD-ANNS/Ours
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Running Tests

### Quick Test Script

```bash
cd /workspace/OOD-ANNS/Ours
./test_ours.sh
```

### Manual Test

```bash
cd /workspace/OOD-ANNS/Ours/build

./test/test_ours \
    --base_index_path ../NGFix/data/t2i-10M/base.index \
    --train_query_path ../NGFix/data/t2i-10M/train_query.fbin \
    --train_gt_path ../NGFix/data/t2i-10M/train_gt.ibin \
    --test_query_path ../NGFix/data/t2i-10M/test_query.fbin \
    --test_gt_path ../NGFix/data/t2i-10M/test_gt.ibin \
    --metric ip_float \
    --result_dir ./results \
    --K 100 \
    --num_test_queries 1000 \
    --num_train_queries 1000
```

## Parameters

- `--base_index_path`: Path to the base HNSW index
- `--train_query_path`: Path to training query vectors
- `--train_gt_path`: Path to training ground truth
- `--test_query_path`: Path to test query vectors
- `--test_gt_path`: Path to test ground truth
- `--metric`: Metric type (`ip_float` or `l2_float`)
- `--result_dir`: Directory to save results
- `--K`: Number of nearest neighbors to retrieve
- `--num_test_queries`: Number of test queries to evaluate
- `--num_train_queries`: Number of training queries to use for optimization

## Output

The test will generate:
- `results/ours_results.csv`: Summary of results (recall, latency, etc.)
- `results/ours_optimized.index`: Optimized index file

## Key Differences from NGFix

1. **Hard Query Detection**: Automatically detects hard queries using lightweight metrics and Jitter, then optimizes only with the hardest queries (top 10% by default)

2. **Grouped EH Calculation**: Reduces EH matrix size by grouping nodes that are reachable within 2 hops, significantly reducing computation time

3. **Optimized Edge Selection**: Uses heap-based approach instead of full sort for selecting fixing edges

4. **Dynamic Update Support**: Tracks added edges with timestamps and supports removing expired edges

## Implementation Details

### Hard Query Detection

The implementation uses a two-stage approach:
- **Stage 0**: Lightweight metrics from search trace (r_visit, r_early, top1_last1_diff)
- **Stage 1**: Jitter (perturbation stability) - adds small noise to query and measures result stability

### Grouped EH Calculation

Nodes are grouped by 2-hop reachability, and EH is calculated on the reduced group graph, then mapped back to the original node space. This typically reduces matrix size by 60-85%.

### Optimized Edge Fixing

Instead of sorting all candidate edges, uses a heap to select edges one by one, reducing sorting overhead.

