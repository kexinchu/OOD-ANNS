#include "ourslib/graph/hnsw_ours.h"
#include <unordered_map>
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <unordered_set>

using namespace ours;

// Calculate recall
float CalculateRecall(const std::vector<std::pair<float, id_t>>& results, int* gt, size_t k) {
    std::unordered_set<id_t> gt_set;
    for(int i = 0; i < k; ++i) {
        gt_set.insert(gt[i]);
    }
    
    int acc = 0;
    for(const auto& p : results) {
        if(gt_set.find(p.second) != gt_set.end()) {
            acc++;
            gt_set.erase(p.second);
        }
    }
    
    return (float)acc / k;
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
        if (arg == "--test_query_path")
            paths["test_query_path"] = argv[i + 1];
        if (arg == "--test_gt_path")
            paths["test_gt_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--result_dir")
            paths["result_dir"] = argv[i + 1];
        if (arg == "--K")
            paths["K"] = argv[i + 1];
        if (arg == "--num_test_queries")
            paths["num_test_queries"] = argv[i + 1];
        if (arg == "--num_train_queries")
            paths["num_train_queries"] = argv[i + 1];
    }
    
    std::string base_index_path = paths["base_index_path"];
    std::string train_query_path = paths["train_query_path"];
    std::string train_gt_path = paths["train_gt_path"];
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string result_dir = paths.count("result_dir") ? paths["result_dir"] : "./";
    std::string metric_str = paths["metric"];
    size_t k = paths.count("K") ? std::stoi(paths["K"]) : 100;
    size_t num_test_queries = paths.count("num_test_queries") ? std::stoi(paths["num_test_queries"]) : 1000;
    size_t num_train_queries = paths.count("num_train_queries") ? std::stoi(paths["num_train_queries"]) : 1000;
    size_t ef_search = 100;
    
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Base index path: " << base_index_path << std::endl;
    std::cout << "Train query path: " << train_query_path << std::endl;
    std::cout << "Train GT path: " << train_gt_path << std::endl;
    std::cout << "Test query path: " << test_query_path << std::endl;
    std::cout << "Test GT path: " << test_gt_path << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "efSearch: " << ef_search << std::endl;
    std::cout << "Num test queries: " << num_test_queries << std::endl;
    std::cout << "Num train queries: " << num_train_queries << std::endl;
    std::cout << std::endl;
    
    // Load data
    size_t train_number = 0, test_number = 0;
    size_t train_gt_dim = 0, test_gt_dim = 0, vecdim = 0;
    
    auto train_query = LoadData<float>(train_query_path, train_number, vecdim);
    auto train_gt = LoadData<int>(train_gt_path, train_number, train_gt_dim);
    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    
    std::cout << "Loaded " << train_number << " training queries" << std::endl;
    std::cout << "Loaded " << test_number << " test queries" << std::endl;
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
    auto base_index = new HNSW_Ours<float>(metric, base_index_path);
    base_index->printGraphInfo();
    std::cout << std::endl;
    
    // Step 1: Detect hard queries using our method
    std::cout << "=== Step 1: Detecting Hard Queries ===" << std::endl;
    std::vector<std::pair<float, size_t>> query_scores;  // (score, query_idx)
    
    size_t actual_train = std::min(num_train_queries, train_number);
    for(size_t i = 0; i < actual_train; ++i) {
        if(i % 100 == 0 && i > 0) {
            std::cout << "  Processing query " << i << "/" << actual_train << std::endl;
        }
        
        auto query_data = train_query + i * vecdim;
        auto metrics = DetectHardQuery(base_index, query_data, k, ef_search, vecdim);
        
        // Combined score: higher is harder
        float combined_score = metrics.hardness_score * 0.5f + metrics.jitter * 0.5f;
        query_scores.push_back({combined_score, i});
    }
    
    // Sort by score (highest first = hardest first)
    std::sort(query_scores.begin(), query_scores.end(), std::greater<std::pair<float, size_t>>());
    
    std::cout << "Detected " << query_scores.size() << " queries" << std::endl;
    if(!query_scores.empty()) {
        std::cout << "Hardest query score: " << query_scores[0].first << std::endl;
        std::cout << "Easiest query score: " << query_scores.back().first << std::endl;
    }
    std::cout << std::endl;
    
    // Step 2: Optimize with top hard queries
    std::cout << "=== Step 2: Optimizing with Hard Queries ===" << std::endl;
    auto working_index = new HNSW_Ours<float>(metric, base_index_path);
    
    // Use top 10% hardest queries
    size_t num_hard_queries = std::min((size_t)(actual_train * 0.1), (size_t)100);
    std::cout << "Using top " << num_hard_queries << " hardest queries for optimization" << std::endl;
    
    auto opt_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(size_t idx = 0; idx < num_hard_queries; ++idx) {
        size_t i = query_scores[idx].second;
        if(idx % 10 == 0 && idx > 0) {
            std::cout << "  Optimizing query " << idx << "/" << num_hard_queries << std::endl;
        }
        
        auto query_data = train_query + i * vecdim;
        auto gt = train_gt + i * train_gt_dim;
        
        // Use optimized NGFix
        working_index->NGFixOptimized(query_data, gt, 100, 100);
        working_index->NGFixOptimized(query_data, gt, 10, 10);
    }
    
    auto opt_end = std::chrono::high_resolution_clock::now();
    auto opt_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(opt_end - opt_start).count();
    std::cout << "Optimization completed in " << opt_time_ms << " ms" << std::endl;
    std::cout << std::endl;
    
    // Step 3: Test on test queries
    std::cout << "=== Step 3: Testing on Test Queries ===" << std::endl;
    working_index->printGraphInfo();
    
    double total_recall = 0;
    double total_latency_us = 0;
    size_t actual_test = std::min(num_test_queries, test_number);
    
    for(size_t i = 0; i < actual_test; ++i) {
        if(i % 100 == 0 && i > 0) {
            std::cout << "  Testing query " << i << "/" << actual_test << std::endl;
        }
        
        auto query_data = test_query + i * vecdim;
        auto gt = test_gt + i * test_gt_dim;
        
        auto start = std::chrono::high_resolution_clock::now();
        size_t ndc = 0;
        auto results = working_index->searchKnn(query_data, k, ef_search, ndc);
        auto end = std::chrono::high_resolution_clock::now();
        auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        float recall = CalculateRecall(results, gt, k);
        total_recall += recall;
        total_latency_us += latency_us;
    }
    
    double avg_recall = total_recall / actual_test;
    double avg_latency_ms = (total_latency_us / actual_test) / 1000.0;
    
    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Average Recall: " << std::fixed << std::setprecision(4) << avg_recall << std::endl;
    std::cout << "Average Latency: " << avg_latency_ms << " ms" << std::endl;
    std::cout << std::endl;
    
    // Save results
    std::string csv_path = result_dir + "/ours_results.csv";
    std::ofstream csv_file(csv_path);
    csv_file << std::fixed << std::setprecision(4);
    csv_file << "Method,Recall,Latency_ms,Num_Hard_Queries\n";
    csv_file << "Ours," << avg_recall << "," << avg_latency_ms << "," << num_hard_queries << "\n";
    csv_file.close();
    
    std::cout << "Results saved to: " << csv_path << std::endl;
    
    // Save optimized index
    std::string index_path = result_dir + "/ours_optimized.index";
    working_index->StoreIndex(index_path);
    std::cout << "Optimized index saved to: " << index_path << std::endl;
    
    // Cleanup
    delete[] train_query;
    delete[] train_gt;
    delete[] test_query;
    delete[] test_gt;
    delete base_index;
    delete working_index;
    
    return 0;
}

