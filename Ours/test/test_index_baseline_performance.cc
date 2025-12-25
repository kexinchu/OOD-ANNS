#include "../../NGFix/ngfixlib/graph/hnsw_ngfix.h"
#include "../../NGFix/test/tools/data_loader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>

using namespace ngfixlib;

// Calculate recall
float CalculateRecall(const std::vector<std::pair<float, id_t>>& results, int* gt, size_t k) {
    std::unordered_set<id_t> gt_set;
    for(size_t i = 0; i < k; ++i) {
        gt_set.insert(gt[i]);
    }
    
    size_t acc = 0;
    for(const auto& p : results) {
        if(gt_set.find(p.second) != gt_set.end()) {
            acc++;
            gt_set.erase(p.second);
        }
    }
    
    return (float)acc / k;
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::unordered_map<std::string, std::string> paths;
    for(int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--index_path" && i + 1 < argc)
            paths["index_path"] = argv[++i];
        if(arg == "--query_path" && i + 1 < argc)
            paths["query_path"] = argv[++i];
        if(arg == "--gt_path" && i + 1 < argc)
            paths["gt_path"] = argv[++i];
        if(arg == "--metric" && i + 1 < argc)
            paths["metric"] = argv[++i];
        if(arg == "--K" && i + 1 < argc)
            paths["K"] = argv[++i];
        if(arg == "--ef_search" && i + 1 < argc)
            paths["ef_search"] = argv[++i];
        if(arg == "--num_queries" && i + 1 < argc)
            paths["num_queries"] = argv[++i];
    }
    
    // Default values
    std::string index_path = paths["index_path"];
    std::string query_path = paths["query_path"];
    std::string gt_path = paths["gt_path"];
    std::string metric_str = paths.count("metric") ? paths["metric"] : "ip_float";
    size_t k = paths.count("K") ? std::stoi(paths["K"]) : 100;
    size_t ef_search = paths.count("ef_search") ? std::stoul(paths["ef_search"]) : 1000;
    size_t num_queries = paths.count("num_queries") ? std::stoul(paths["num_queries"]) : 60000;
    
    std::cout << "=== Index Baseline Performance Test ===" << std::endl;
    std::cout << "Index path: " << index_path << std::endl;
    std::cout << "Query path: " << query_path << std::endl;
    std::cout << "GT path: " << gt_path << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "ef_search: " << ef_search << std::endl;
    std::cout << "Number of queries: " << num_queries << std::endl;
    std::cout << std::endl;
    
    // Load data
    size_t query_number = 0;
    size_t query_vecdim = 0;
    size_t gt_number = 0;
    size_t gt_dim = 0;
    
    std::cout << "=== Loading Queries ===" << std::endl;
    auto query_data = LoadData<float>(query_path, query_number, query_vecdim);
    std::cout << "Loaded " << query_number << " queries, dimension: " << query_vecdim << std::endl;
    
    // Limit to num_queries
    if(query_number > num_queries) {
        query_number = num_queries;
        std::cout << "Using first " << num_queries << " queries" << std::endl;
    }
    
    // Determine metric
    ngfixlib::Metric metric;
    if(metric_str == "ip_float") {
        metric = ngfixlib::IP_float;
    } else if(metric_str == "l2_float") {
        metric = ngfixlib::L2_float;
    } else {
        std::cerr << "ERROR: Unsupported metric type." << std::endl;
        delete[] query_data;
        return 1;
    }
    
    // Load GT data
    std::cout << "=== Loading Ground Truth ===" << std::endl;
    auto gt_data = LoadData<int>(gt_path, gt_number, gt_dim);
    std::cout << "Loaded " << gt_number << " GT entries, dimension: " << gt_dim << std::endl;
    
    if(gt_dim < k) {
        std::cerr << "WARNING: GT dimension (" << gt_dim 
                  << ") is less than K (" << k << "). Using available GT." << std::endl;
    }
    
    if(gt_number < query_number) {
        std::cerr << "WARNING: GT number (" << gt_number 
                  << ") is less than query number (" << query_number << "). Using minimum." << std::endl;
        query_number = std::min(gt_number, query_number);
    }
    
    // Load index
    std::cout << "=== Loading Index (NGFix) ===" << std::endl;
    HNSW_NGFix<float>* index = nullptr;
    try {
        index = new HNSW_NGFix<float>(metric, index_path);
        index->printGraphInfo();
        
        size_t index_dim = index->dim;
        std::cout << "Index dimension: " << index_dim << std::endl;
        std::cout << "Query dimension: " << query_vecdim << std::endl;
        
        if(index_dim != query_vecdim) {
            std::cerr << "ERROR: Dimension mismatch! Index=" << index_dim 
                      << ", Query=" << query_vecdim << std::endl;
            delete[] query_data;
            delete[] gt_data;
            delete index;
            return 1;
        }
    } catch(const std::exception& e) {
        std::cerr << "ERROR loading index: " << e.what() << std::endl;
        delete[] query_data;
        delete[] gt_data;
        return 1;
    }
    
    std::cout << std::endl;
    
    // Test queries
    std::cout << "=== Testing " << query_number << " Queries ===" << std::endl;
    
    double sum_recall = 0.0;
    double sum_latency_ms = 0.0;
    size_t sum_ndc = 0;
    
    auto test_start = std::chrono::high_resolution_clock::now();
    
    for(size_t i = 0; i < query_number; ++i) {
        float* query = query_data + i * query_vecdim;
        int* gt = gt_data + i * gt_dim;
        
        size_t ndc = 0;
        auto search_start = std::chrono::high_resolution_clock::now();
        auto results = index->searchKnn(query, k, ef_search, ndc);
        auto search_end = std::chrono::high_resolution_clock::now();
        
        double latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            search_end - search_start).count() / 1000.0;
        
        float recall = CalculateRecall(results, gt, k);
        
        sum_recall += recall;
        sum_latency_ms += latency_ms;
        sum_ndc += ndc;
        
        if((i + 1) % 10000 == 0) {
            std::cout << "  Processed " << (i + 1) << " / " << query_number << " queries" << std::endl;
        }
    }
    
    auto test_end = std::chrono::high_resolution_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        test_end - test_start).count();
    
    double avg_recall = sum_recall / query_number;
    double avg_latency_ms = sum_latency_ms / query_number;
    double avg_ndc = (double)sum_ndc / query_number;
    
    std::cout << std::endl;
    std::cout << "=== Test Results ===" << std::endl;
    std::cout << "Number of queries tested: " << query_number << std::endl;
    std::cout << "Average Recall: " << std::fixed << std::setprecision(6) << avg_recall << std::endl;
    std::cout << "Average Latency (ms): " << std::fixed << std::setprecision(4) << avg_latency_ms << std::endl;
    std::cout << "Average NDC: " << std::fixed << std::setprecision(2) << avg_ndc << std::endl;
    std::cout << "Total test time: " << total_time_ms << " ms (" << (total_time_ms / 1000.0) << " seconds)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) 
              << (query_number * 1000.0 / total_time_ms) << " QPS" << std::endl;
    
    // Save results to file
    std::string result_file = "index_baseline_performance.csv";
    std::ofstream out(result_file);
    if(out.is_open()) {
        out << "index_path,query_path,gt_path,k,ef_search,num_queries,avg_recall,avg_latency_ms,avg_ndc,total_time_ms,throughput_qps\n";
        out << index_path << ","
            << query_path << ","
            << gt_path << ","
            << k << ","
            << ef_search << ","
            << query_number << ","
            << std::fixed << std::setprecision(6) << avg_recall << ","
            << std::fixed << std::setprecision(4) << avg_latency_ms << ","
            << std::fixed << std::setprecision(2) << avg_ndc << ","
            << total_time_ms << ","
            << std::fixed << std::setprecision(2) << (query_number * 1000.0 / total_time_ms) << "\n";
        out.close();
        std::cout << "Results saved to: " << result_file << std::endl;
    }
    
    delete index;
    delete[] query_data;
    delete[] gt_data;
    
    return 0;
}

