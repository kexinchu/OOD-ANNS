#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <fstream>
#include <algorithm>
#include <mutex>
#include <sys/time.h>
#include <unordered_set>
using namespace ngfixlib;

struct QueryResult {
    int64_t timestamp_us;
    int query_id;
    float recall;
    int64_t latency_us;
    int operation_progress;  // number of insertions/deletions completed
};

std::mutex result_mutex;
std::vector<QueryResult> all_results;
std::atomic<int> operation_progress{0};
std::atomic<bool> operation_complete{false};

void search_thread(HNSW_NGFix<float>* hnsw_ngfix, 
                   float* test_query, 
                   int* test_gt,
                   size_t test_number,
                   size_t test_gt_dim,
                   size_t vecdim,
                   size_t k,
                   size_t efs,
                   int target_qps) {
    const int64_t interval_us = 1000000 / target_qps;  // microseconds between queries
    auto start_time = std::chrono::high_resolution_clock::now();
    int query_idx = 0;
    int query_count = 0;

    while (!operation_complete.load()) {
        auto query_start = std::chrono::high_resolution_clock::now();
        int64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            query_start - start_time).count();
        
        // Get current operation progress
        int current_progress = operation_progress.load();
        
        // Perform search
        size_t ndc = 0;
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);
        
        auto aknns = hnsw_ngfix->searchKnn(test_query + query_idx * vecdim, k, efs, ndc);
        
        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t latency_us = (newVal.tv_sec * Converter + newVal.tv_usec) - 
                            (val.tv_sec * Converter + val.tv_usec);
        
        // Calculate recall
        std::unordered_set<id_t> gtset;
        auto gt = test_gt + query_idx * test_gt_dim;
        for(int i = 0; i < k; ++i) {
            gtset.insert(gt[i]);
        }
        
        int acc = 0;
        for(int i = 0; i < k; ++i) {
            if(gtset.find(aknns[i].second) != gtset.end()) {
                ++acc;
                gtset.erase(aknns[i].second);
            }
        }
        float recall = (float)acc / k;
        
        // Store result
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            all_results.push_back({timestamp_us, query_idx, recall, latency_us, current_progress});
        }
        
        query_count++;
        query_idx = (query_idx + 1) % test_number;  // Cycle through queries
        
        // Sleep to maintain target QPS
        auto query_end = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            query_end - query_start).count();
        int64_t sleep_us = interval_us - query_duration;
        if (sleep_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }
    }
    
    std::cout << "Search thread completed. Total queries: " << query_count << std::endl;
}

void insert_thread(HNSW_NGFix<float>* hnsw_ngfix,
                   float* base_data,
                   size_t insert_st_id,
                   size_t insert_count,
                   size_t vecdim,
                   size_t efC) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for(size_t i = insert_st_id; i < insert_st_id + insert_count; ++i) {
        hnsw_ngfix->InsertPoint(i, efC, base_data + i * vecdim);
        operation_progress.store(i - insert_st_id + 1);
        
        if((i - insert_st_id + 1) % 1000 == 0) {
            std::cout << "Inserted " << (i - insert_st_id + 1) << " / " << insert_count << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insertion completed. Total time: " << diff << " ms" << std::endl;
    
    operation_complete.store(true);
}

void delete_thread(HNSW_NGFix<float>* hnsw_ngfix,
                   size_t delete_start,
                   size_t delete_count) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for(size_t i = delete_start; i < delete_start + delete_count; ++i) {
        hnsw_ngfix->DeletePointByFlag(i);
        operation_progress.store(i - delete_start + 1);
        
        if((i - delete_start + 1) % 1000 == 0) {
            std::cout << "Deleted " << (i - delete_start + 1) << " / " << delete_count << std::endl;
        }
    }
    
    // Final cleanup
    hnsw_ngfix->DeleteAllFlagPointsByNGFix();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Deletion completed. Total time: " << diff << " ms" << std::endl;
    
    operation_complete.store(true);
}

void calculate_stats(const std::vector<QueryResult>& results, 
                     double& avg_recall, double& avg_latency,
                     double& p95_latency, double& p99_latency) {
    if (results.empty()) {
        avg_recall = 0;
        avg_latency = 0;
        p95_latency = 0;
        p99_latency = 0;
        return;
    }
    
    double sum_recall = 0;
    std::vector<int64_t> latencies;
    
    for (const auto& r : results) {
        sum_recall += r.recall;
        latencies.push_back(r.latency_us);
    }
    
    avg_recall = sum_recall / results.size();
    avg_latency = 0;
    for (auto l : latencies) {
        avg_latency += l;
    }
    avg_latency /= latencies.size();
    
    std::sort(latencies.begin(), latencies.end());
    p95_latency = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    p99_latency = latencies[static_cast<size_t>(latencies.size() * 0.99)];
}

int main(int argc, char* argv[]) {
    std::string operation_type = "";  // "insert" or "delete"
    size_t insert_st_id = 0, insert_count = 0;
    size_t delete_start = 0, delete_count = 0;
    size_t efC = 500;
    size_t efs = 500;
    size_t k = 100;
    int target_qps = 1;
    
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--operation") operation_type = argv[i + 1];
        if (arg == "--index_path") paths["index_path"] = argv[i + 1];
        if (arg == "--test_query_path") paths["test_query_path"] = argv[i + 1];
        if (arg == "--test_gt_path") paths["test_gt_path"] = argv[i + 1];
        if (arg == "--base_data_path") paths["base_data_path"] = argv[i + 1];
        if (arg == "--metric") paths["metric"] = argv[i + 1];
        if (arg == "--result_path") paths["result_path"] = argv[i + 1];
        if (arg == "--K") k = std::stoi(argv[i + 1]);
        if (arg == "--efs") efs = std::stoi(argv[i + 1]);
        if (arg == "--efC") efC = std::stoi(argv[i + 1]);
        if (arg == "--insert_st_id") insert_st_id = std::stoi(argv[i + 1]);
        if (arg == "--insert_count") insert_count = std::stoi(argv[i + 1]);
        if (arg == "--delete_start") delete_start = std::stoi(argv[i + 1]);
        if (arg == "--delete_count") delete_count = std::stoi(argv[i + 1]);
        if (arg == "--qps") target_qps = std::stoi(argv[i + 1]);
    }
    
    if (operation_type != "insert" && operation_type != "delete") {
        std::cerr << "Error: --operation must be 'insert' or 'delete'" << std::endl;
        return 1;
    }
    
    std::string index_path = paths["index_path"];
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string metric_str = paths["metric"];
    std::string result_path = paths["result_path"];
    
    std::cout << "Operation: " << operation_type << std::endl;
    std::cout << "Index path: " << index_path << std::endl;
    std::cout << "Query path: " << test_query_path << std::endl;
    std::cout << "GT path: " << test_gt_path << std::endl;
    std::cout << "Result path: " << result_path << std::endl;
    std::cout << "QPS: " << target_qps << std::endl;
    
    // Load data
    size_t test_number = 0, vecdim = 0;
    size_t test_gt_dim = 0;
    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    
    Metric metric;
    if(metric_str == "ip_float") {
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        metric = L2_float;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }
    
    // Load index
    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, index_path);
    hnsw_ngfix->printGraphInfo();
    
    // Prepare for insertion if needed
    float* base_data = nullptr;
    size_t base_number = 0;
    if (operation_type == "insert") {
        std::string base_data_path = paths["base_data_path"];
        base_data = LoadData<float>(base_data_path, base_number, vecdim);
        hnsw_ngfix->resize(base_number);
        std::cout << "Will insert " << insert_count << " vectors starting from ID " << insert_st_id << std::endl;
    } else {
        std::cout << "Will delete " << delete_count << " vectors starting from ID " << delete_start << std::endl;
    }
    
    // Start threads
    std::thread search_t, update_t;
    
    if (operation_type == "insert") {
        search_t = std::thread(search_thread, hnsw_ngfix, test_query, test_gt, 
                              test_number, test_gt_dim, vecdim, k, efs, target_qps);
        update_t = std::thread(insert_thread, hnsw_ngfix, base_data, 
                              insert_st_id, insert_count, vecdim, efC);
    } else {
        search_t = std::thread(search_thread, hnsw_ngfix, test_query, test_gt, 
                              test_number, test_gt_dim, vecdim, k, efs, target_qps);
        update_t = std::thread(delete_thread, hnsw_ngfix, delete_start, delete_count);
    }
    
    // Wait for threads
    update_t.join();
    search_t.join();
    
    // Calculate and output statistics
    double avg_recall, avg_latency, p95_latency, p99_latency;
    calculate_stats(all_results, avg_recall, avg_latency, p95_latency, p99_latency);
    
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Total queries: " << all_results.size() << std::endl;
    std::cout << "Average recall: " << avg_recall << std::endl;
    std::cout << "Average latency: " << avg_latency / 1000.0 << " ms" << std::endl;
    std::cout << "P95 latency: " << p95_latency / 1000.0 << " ms" << std::endl;
    std::cout << "P99 latency: " << p99_latency / 1000.0 << " ms" << std::endl;
    
    // Write detailed results to CSV
    std::ofstream output(result_path);
    output << "timestamp_us,query_id,recall,latency_us,operation_progress\n";
    for (const auto& r : all_results) {
        output << r.timestamp_us << "," << r.query_id << "," << r.recall 
               << "," << r.latency_us << "," << r.operation_progress << "\n";
    }
    output.close();
    
    std::cout << "Results written to: " << result_path << std::endl;
    
    delete[] test_query;
    delete[] test_gt;
    if (base_data) delete[] base_data;
    delete hnsw_ngfix;
    
    return 0;
}

