#include "ourslib/graph/hnsw_ours.h"
#include "tools/data_loader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <random>
#include <numeric>
#include <cmath>

using namespace ours;

// Performance metrics for each component
struct ComponentMetrics {
    std::vector<double> latencies_us;  // in microseconds
    
    double getMean() const {
        if(latencies_us.empty()) return 0.0;
        return std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / latencies_us.size();
    }
    
    double getPercentile(double p) const {
        if(latencies_us.empty()) return 0.0;
        std::vector<double> sorted = latencies_us;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = (size_t)(p * sorted.size());
        if(idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted[idx];
    }
    
    double getP50() const { return getPercentile(0.50); }
    double getP95() const { return getPercentile(0.95); }
    double getP99() const { return getPercentile(0.99); }
};

// Overall performance breakdown
struct PerformanceBreakdown {
    ComponentMetrics compute_gq;        // ComputeGq (for analysis, not in total)
    ComponentMetrics grouping;          // GroupNodesByReachability (for analysis, not in total)
    ComponentMetrics calc_hardness;     // CalculateHardnessGrouped
    ComponentMetrics bitset_prep;       // Bitset preparation
    ComponentMetrics get_edges;         // getDefectsFixingEdgesOptimized
    ComponentMetrics add_edges;         // Add edges to graph
    ComponentMetrics other_overhead;     // Other overhead (should be < 3%)
    ComponentMetrics total;             // Total NGFixOptimized time (matching NGFix's NGFix)
};

// Wrapper to measure NGFixOptimized with detailed breakdown
PerformanceBreakdown MeasureNGFixOptimized(
    HNSW_Ours<float>* index,
    float* query_data,
    int* gt,
    size_t Nq,
    size_t Kh) {
    
    PerformanceBreakdown breakdown;
    
    // Total time - matching NGFix's NGFix function (gt is already provided, no AKNNGroundTruth)
    // Calculate S first (very fast, negligible overhead)
    size_t S_val = std::min((size_t)200, 2*Nq);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 1. CalculateHardnessGrouped (this is the main optimization, replaces CalculateHardness)
    // Note: This includes GroupNodesByReachability internally, so we measure the whole thing
    auto hardness_start = std::chrono::high_resolution_clock::now();
    auto hardness_result = index->CalculateHardnessGroupedWithMapping(gt, Nq, Kh, S_val, query_data);
    auto& H = hardness_result.H;
    auto hardness_end = std::chrono::high_resolution_clock::now();
    auto hardness_us = std::chrono::duration_cast<std::chrono::microseconds>(hardness_end - hardness_start).count();
    breakdown.calc_hardness.latencies_us.push_back(hardness_us);
    
    // The "Other Overhead" includes:
    // - GroupNodesByReachability (called inside CalculateHardnessGroupedWithMapping)
    // - Memory allocations for maps, vectors, etc.
    // - Function call overhead
    // But we can't easily separate these without modifying the implementation
    
    // For breakdown analysis only (not included in total):
    // Measure ComputeGq separately (for comparison with original)
    auto gq_start = std::chrono::high_resolution_clock::now();
    auto Gq = index->ComputeGq(gt, S_val);
    auto gq_end = std::chrono::high_resolution_clock::now();
    auto gq_us = std::chrono::duration_cast<std::chrono::microseconds>(gq_end - gq_start).count();
    breakdown.compute_gq.latencies_us.push_back(gq_us);
    
    // Measure Grouping separately (for analysis)
    auto group_start = std::chrono::high_resolution_clock::now();
    auto groups = GroupNodesByReachability(index, gt, std::min(Nq, S_val), 2);
    auto group_end = std::chrono::high_resolution_clock::now();
    auto group_us = std::chrono::duration_cast<std::chrono::microseconds>(group_end - group_start).count();
    breakdown.grouping.latencies_us.push_back(group_us);
    
    // 2. Build bitset
    auto bitset_start = std::chrono::high_resolution_clock::now();
    std::bitset<MAX_Nq> f[Nq];
    for(int i = 0; i < (int)Nq; ++i){
        for(int j = 0; j < (int)Nq; ++j){
            f[i][j] = (H[i][j] <= Kh) ? 1 : 0;
        }
    }
    auto bitset_end = std::chrono::high_resolution_clock::now();
    auto bitset_us = std::chrono::duration_cast<std::chrono::microseconds>(bitset_end - bitset_start).count();
    breakdown.bitset_prep.latencies_us.push_back(bitset_us);
    
    // 3. getDefectsFixingEdgesOptimized (iterative approach - no full sort needed)
    auto edges_start = std::chrono::high_resolution_clock::now();
    const std::vector<int>* group_ptr = hardness_result.node_idx_to_group.empty() ? nullptr : &hardness_result.node_idx_to_group;
    std::vector<std::vector<float>> dist_cache;  // Distance cache for this query
    auto new_edges = index->getDefectsFixingEdgesOptimized(f, H, query_data, gt, Nq, Kh, group_ptr, &dist_cache);
    auto edges_end = std::chrono::high_resolution_clock::now();
    auto edges_us = std::chrono::duration_cast<std::chrono::microseconds>(edges_end - edges_start).count();
    breakdown.get_edges.latencies_us.push_back(edges_us);
    
    // 4. Add edges to graph
    auto add_start = std::chrono::high_resolution_clock::now();
    size_t ts = index->current_timestamp.fetch_add(1);
    size_t mex = index->getMEX();
    for(auto [u, vs] : new_edges) {
        std::unique_lock <std::shared_mutex> lock(index->getNodeLock(u));
        for(auto [v, eh] : vs) {
            index->Graph[u].add_ngfix_neighbors(v, eh, mex);
            index->added_edges[u].push_back({u, v, eh, ts});
        }
    }
    auto add_end = std::chrono::high_resolution_clock::now();
    auto add_us = std::chrono::duration_cast<std::chrono::microseconds>(add_end - add_start).count();
    breakdown.add_edges.latencies_us.push_back(add_us);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    breakdown.total.latencies_us.push_back(total_us);
    
    // Calculate other overhead (everything not explicitly measured)
    double measured_sum = hardness_us + bitset_us + edges_us + add_us;
    double other_overhead_us = total_us - measured_sum;
    if(other_overhead_us < 0) other_overhead_us = 0;  // Handle measurement errors
    breakdown.other_overhead.latencies_us.push_back(other_overhead_us);
    
    return breakdown;
}

void PrintMetrics(const std::string& name, const ComponentMetrics& metrics) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::setw(30) << std::left << name << ": "
              << std::setw(10) << std::right << metrics.getMean() << " us (avg), "
              << std::setw(10) << metrics.getP50() << " us (P50), "
              << std::setw(10) << metrics.getP95() << " us (P95), "
              << std::setw(10) << metrics.getP99() << " us (P99)" << std::endl;
}

int main(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base_index_path")
            paths["base_index_path"] = argv[i + 1];
        if (arg == "--train_query_path")
            paths["train_query_path"] = argv[i + 1];
        if (arg == "--train_gt_path")
            paths["train_gt_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--result_dir")
            paths["result_dir"] = argv[i + 1];
        if (arg == "--K")
            paths["K"] = argv[i + 1];
        if (arg == "--num_queries")
            paths["num_queries"] = argv[i + 1];
    }
    
    std::string base_index_path = paths["base_index_path"];
    std::string train_query_path = paths["train_query_path"];
    std::string train_gt_path = paths["train_gt_path"];
    std::string result_dir = paths.count("result_dir") ? paths["result_dir"] : "./";
    std::string metric_str = paths["metric"];
    size_t k = paths.count("K") ? std::stoi(paths["K"]) : 100;
    size_t num_queries = paths.count("num_queries") ? std::stoi(paths["num_queries"]) : 1000;
    
    std::cout << "=== Performance Analysis Configuration ===" << std::endl;
    std::cout << "Base index path: " << base_index_path << std::endl;
    std::cout << "Train query path: " << train_query_path << std::endl;
    std::cout << "Train GT path: " << train_gt_path << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "Number of queries to test: " << num_queries << std::endl;
    std::cout << std::endl;
    
    // Load data
    size_t train_number = 0;
    size_t train_gt_dim = 0, vecdim = 0;
    
    auto train_query = LoadData<float>(train_query_path, train_number, vecdim);
    auto train_gt = LoadData<int>(train_gt_path, train_number, train_gt_dim);
    
    std::cout << "Loaded " << train_number << " training queries" << std::endl;
    std::cout << "Vector dimension: " << vecdim << std::endl;
    std::cout << std::endl;
    
    Metric metric;
    if(metric_str == "ip_float") {
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        metric = L2_float;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }
    
    // Load base index
    std::cout << "=== Loading Base Index ===" << std::endl;
    auto index = new HNSW_Ours<float>(metric, base_index_path);
    index->printGraphInfo();
    std::cout << std::endl;
    
    // Randomly select queries
    std::vector<size_t> query_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, train_number - 1);
    
    for(size_t i = 0; i < num_queries && i < train_number; ++i) {
        query_indices.push_back(dis(gen));
    }
    
    std::cout << "=== Running Performance Analysis on " << query_indices.size() << " Random Queries ===" << std::endl;
    
    PerformanceBreakdown overall;
    
    for(size_t idx = 0; idx < query_indices.size(); ++idx) {
        size_t i = query_indices[idx];
        
        if(idx % 100 == 0 && idx > 0) {
            std::cout << "  Processing query " << idx << "/" << query_indices.size() << std::endl;
        }
        
        auto query_data = train_query + i * vecdim;
        auto gt = train_gt + i * train_gt_dim;
        
        // Measure performance
        auto breakdown = MeasureNGFixOptimized(index, query_data, gt, 100, 100);
        
        // Accumulate metrics
        overall.compute_gq.latencies_us.insert(
            overall.compute_gq.latencies_us.end(),
            breakdown.compute_gq.latencies_us.begin(),
            breakdown.compute_gq.latencies_us.end());
        
        overall.grouping.latencies_us.insert(
            overall.grouping.latencies_us.end(),
            breakdown.grouping.latencies_us.begin(),
            breakdown.grouping.latencies_us.end());
        
        overall.calc_hardness.latencies_us.insert(
            overall.calc_hardness.latencies_us.end(),
            breakdown.calc_hardness.latencies_us.begin(),
            breakdown.calc_hardness.latencies_us.end());
        
        overall.bitset_prep.latencies_us.insert(
            overall.bitset_prep.latencies_us.end(),
            breakdown.bitset_prep.latencies_us.begin(),
            breakdown.bitset_prep.latencies_us.end());
        
        overall.get_edges.latencies_us.insert(
            overall.get_edges.latencies_us.end(),
            breakdown.get_edges.latencies_us.begin(),
            breakdown.get_edges.latencies_us.end());
        
        overall.add_edges.latencies_us.insert(
            overall.add_edges.latencies_us.end(),
            breakdown.add_edges.latencies_us.begin(),
            breakdown.add_edges.latencies_us.end());
        
        overall.other_overhead.latencies_us.insert(
            overall.other_overhead.latencies_us.end(),
            breakdown.other_overhead.latencies_us.begin(),
            breakdown.other_overhead.latencies_us.end());
        
        overall.total.latencies_us.insert(
            overall.total.latencies_us.end(),
            breakdown.total.latencies_us.begin(),
            breakdown.total.latencies_us.end());
    }
    
    std::cout << std::endl;
    std::cout << "=== Performance Breakdown (Average over " << query_indices.size() << " queries) ===" << std::endl;
    std::cout << std::endl;
    
    // Note: AKNNGroundTruth is not included because gt is provided as input (matching NGFix)
    PrintMetrics("CalculateHardnessGrouped", overall.calc_hardness);
    PrintMetrics("Bitset Preparation", overall.bitset_prep);
    PrintMetrics("getDefectsFixingEdges", overall.get_edges);
    PrintMetrics("Add Edges", overall.add_edges);
    PrintMetrics("Other Overhead", overall.other_overhead);
    PrintMetrics("TOTAL NGFixOptimized", overall.total);
    
    std::cout << std::endl;
    std::cout << "=== Breakdown Analysis (for reference, not in total) ===" << std::endl;
    PrintMetrics("ComputeGq (original)", overall.compute_gq);
    PrintMetrics("GroupNodesByReachability", overall.grouping);
    
    std::cout << std::endl;
    std::cout << "=== Percentage Breakdown (of TOTAL) ===" << std::endl;
    double total_avg = overall.total.getMean();
    if(total_avg > 0) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  CalculateHardnessGrouped:" << std::setw(6) << (overall.calc_hardness.getMean() / total_avg * 100) << "%" << std::endl;
        std::cout << "  Bitset Preparation:      " << std::setw(6) << (overall.bitset_prep.getMean() / total_avg * 100) << "%" << std::endl;
        std::cout << "  getDefectsFixingEdges:   " << std::setw(6) << (overall.get_edges.getMean() / total_avg * 100) << "%" << std::endl;
        std::cout << "  Add Edges:               " << std::setw(6) << (overall.add_edges.getMean() / total_avg * 100) << "%" << std::endl;
        std::cout << "  Other Overhead:          " << std::setw(6) << (overall.other_overhead.getMean() / total_avg * 100) << "%" << std::endl;
        
        // Verify sum
        double sum_pct = (overall.calc_hardness.getMean() + overall.bitset_prep.getMean() + 
                          overall.get_edges.getMean() + overall.add_edges.getMean() + 
                          overall.other_overhead.getMean()) / total_avg * 100;
        std::cout << "  Sum of components:       " << std::setw(6) << sum_pct << "%" << std::endl;
    }
    
    // Save detailed results to CSV
    std::string csv_path = result_dir + "/performance_breakdown.csv";
    std::ofstream csv_file(csv_path);
    csv_file << std::fixed << std::setprecision(4);
    csv_file << "Component,Mean_us,P50_us,P95_us,P99_us,Percentage\n";
    
    if(total_avg > 0) {
        csv_file << "CalculateHardnessGrouped," << overall.calc_hardness.getMean() << ","
                 << overall.calc_hardness.getP50() << "," << overall.calc_hardness.getP95() << ","
                 << overall.calc_hardness.getP99() << "," << (overall.calc_hardness.getMean() / total_avg * 100) << "\n";
        
        csv_file << "Bitset_Preparation," << overall.bitset_prep.getMean() << ","
                 << overall.bitset_prep.getP50() << "," << overall.bitset_prep.getP95() << ","
                 << overall.bitset_prep.getP99() << "," << (overall.bitset_prep.getMean() / total_avg * 100) << "\n";
        
        csv_file << "getDefectsFixingEdges," << overall.get_edges.getMean() << ","
                 << overall.get_edges.getP50() << "," << overall.get_edges.getP95() << ","
                 << overall.get_edges.getP99() << "," << (overall.get_edges.getMean() / total_avg * 100) << "\n";
        
        csv_file << "Add_Edges," << overall.add_edges.getMean() << ","
                 << overall.add_edges.getP50() << "," << overall.add_edges.getP95() << ","
                 << overall.add_edges.getP99() << "," << (overall.add_edges.getMean() / total_avg * 100) << "\n";
        
        csv_file << "Other_Overhead," << overall.other_overhead.getMean() << ","
                 << overall.other_overhead.getP50() << "," << overall.other_overhead.getP95() << ","
                 << overall.other_overhead.getP99() << "," << (overall.other_overhead.getMean() / total_avg * 100) << "\n";
        
        csv_file << "TOTAL," << overall.total.getMean() << ","
                 << overall.total.getP50() << "," << overall.total.getP95() << ","
                 << overall.total.getP99() << ",100.00\n";
        
        // Additional breakdown for analysis
        csv_file << "\n# Breakdown Analysis (not included in total)\n";
        csv_file << "ComputeGq_original," << overall.compute_gq.getMean() << ","
                 << overall.compute_gq.getP50() << "," << overall.compute_gq.getP95() << ","
                 << overall.compute_gq.getP99() << ",N/A\n";
        
        csv_file << "GroupNodesByReachability," << overall.grouping.getMean() << ","
                 << overall.grouping.getP50() << "," << overall.grouping.getP95() << ","
                 << overall.grouping.getP99() << ",N/A\n";
    }
    
    csv_file.close();
    
    std::cout << std::endl;
    std::cout << "Results saved to: " << csv_path << std::endl;
    
    // Cleanup
    delete[] train_query;
    delete[] train_gt;
    delete index;
    
    return 0;
}

