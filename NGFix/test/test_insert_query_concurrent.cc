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
#include <unordered_map>
#include <cassert>
using namespace ngfixlib;

struct QueryResult {
    int64_t timestamp_us;
    int query_id;
    float recall;
    int64_t latency_us;
    int insert_progress;  // number of insertions completed
    bool thread_safe;    // whether query completed without crash/exception
};

struct InsertResult {
    int64_t timestamp_us;
    int insert_id;
    int64_t latency_us;
    std::string phase;  // "insert" or "rebuild"
};

std::mutex result_mutex;
std::vector<QueryResult> query_results;
std::vector<InsertResult> insert_results;
std::atomic<int> insert_progress{0};
std::atomic<bool> operation_complete{false};
std::atomic<int> thread_safety_errors{0};

void query_thread(HNSW_NGFix<float>* hnsw_ngfix, 
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
        
        // Get current insert progress
        int current_progress = insert_progress.load();
        
        // Perform search with exception handling for thread safety
        size_t ndc = 0;
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);
        
        bool thread_safe = true;
        std::vector<std::pair<float, id_t> > aknns;
        
        try {
            aknns = hnsw_ngfix->searchKnn(test_query + query_idx * vecdim, k, efs, ndc);
        } catch (const std::exception& e) {
            thread_safe = false;
            thread_safety_errors.fetch_add(1);
            std::cerr << "Query thread safety error: " << e.what() << std::endl;
            // Continue with empty results
        } catch (...) {
            thread_safe = false;
            thread_safety_errors.fetch_add(1);
            std::cerr << "Query unknown error occurred" << std::endl;
        }
        
        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t latency_us = (newVal.tv_sec * Converter + newVal.tv_usec) - 
                            (val.tv_sec * Converter + val.tv_usec);
        
        // Calculate recall (only if we got valid results)
        float recall = 0.0f;
        if (thread_safe && !aknns.empty()) {
            std::unordered_set<id_t> gtset;
            auto gt = test_gt + query_idx * test_gt_dim;
            for(int i = 0; i < k && i < test_gt_dim; ++i) {
                gtset.insert(gt[i]);
            }
            
            int acc = 0;
            for(size_t i = 0; i < aknns.size() && i < k; ++i) {
                if(gtset.find(aknns[i].second) != gtset.end()) {
                    ++acc;
                    gtset.erase(aknns[i].second);
                }
            }
            recall = (float)acc / k;
        }
        
        // Store result
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            query_results.push_back({timestamp_us, query_idx, recall, latency_us, current_progress, thread_safe});
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
    
    std::cout << "Query thread completed. Total queries: " << query_count << std::endl;
}

void insert_thread(HNSW_NGFix<float>* hnsw_ngfix,
                   float* base_data,
                   size_t insert_st_id,
                   size_t insert_count,
                   size_t vecdim,
                   size_t efC,
                   float partial_rebuild_ratio,
                   int target_qps,
                   size_t rebuild_interval) {
    auto start_time = std::chrono::high_resolution_clock::now();
    const int64_t interval_us = 1000000 / target_qps;  // microseconds between insertions
    
    std::cout << "Inserting points with periodic partial rebuild (every " << rebuild_interval << " inserts)..." << std::endl;
    
    for(size_t i = insert_st_id; i < insert_st_id + insert_count; ++i) {
        auto insert_start_time = std::chrono::high_resolution_clock::now();
        int64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            insert_start_time - start_time).count();
        
        bool thread_safe = true;
        try {
            hnsw_ngfix->InsertPoint(i, efC, base_data + i * vecdim);
        } catch (const std::exception& e) {
            thread_safe = false;
            thread_safety_errors.fetch_add(1);
            std::cerr << "Insert thread safety error: " << e.what() << std::endl;
        } catch (...) {
            thread_safe = false;
            thread_safety_errors.fetch_add(1);
            std::cerr << "Insert unknown error occurred" << std::endl;
        }
        
        auto insert_end_time = std::chrono::high_resolution_clock::now();
        int64_t latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
            insert_end_time - insert_start_time).count();
        
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            insert_results.push_back({timestamp_us, (int)i, latency_us, "insert"});
        }
        
        insert_progress.store(i - insert_st_id + 1);
        
        if((i - insert_st_id + 1) % 1000 == 0) {
            std::cout << "Inserted " << (i - insert_st_id + 1) << " / " << insert_count << std::endl;
        }
        
        // Trigger partial rebuild periodically during insertion
        if (partial_rebuild_ratio > 0.0f && (i - insert_st_id + 1) % rebuild_interval == 0) {
            std::cout << "Triggering partial rebuild (PartialRemoveEdges, ratio=" << partial_rebuild_ratio 
                      << ") after " << (i - insert_st_id + 1) << " inserts..." << std::endl;
            auto rebuild_start = std::chrono::high_resolution_clock::now();
            int64_t rebuild_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                rebuild_start - start_time).count();
            
            bool rebuild_thread_safe = true;
            try {
                hnsw_ngfix->PartialRemoveEdges(partial_rebuild_ratio);
            } catch (const std::exception& e) {
                rebuild_thread_safe = false;
                thread_safety_errors.fetch_add(1);
                std::cerr << "Partial rebuild thread safety error: " << e.what() << std::endl;
            } catch (...) {
                rebuild_thread_safe = false;
                thread_safety_errors.fetch_add(1);
                std::cerr << "Partial rebuild unknown error occurred" << std::endl;
            }
            
            auto rebuild_end = std::chrono::high_resolution_clock::now();
            int64_t rebuild_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                rebuild_end - rebuild_start).count();
            
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                insert_results.push_back({rebuild_timestamp, -1, rebuild_latency, "rebuild"});
            }
            
            std::cout << "Partial rebuild completed. Latency: " << rebuild_latency / 1000.0 << " ms" << std::endl;
        }
        
        // Sleep to maintain target QPS
        auto insert_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            insert_end_time - insert_start_time).count();
        int64_t sleep_us = interval_us - insert_duration;
        if (sleep_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time).count();
    std::cout << "Insertion completed. Total time: " << diff << " ms" << std::endl;
    
    operation_complete.store(true);
}

void calculate_stats(const std::vector<QueryResult>& results, 
                     double& avg_recall, double& avg_latency,
                     double& p50_latency, double& p95_latency, double& p99_latency,
                     double& thread_safety_rate) {
    if (results.empty()) {
        avg_recall = 0;
        avg_latency = 0;
        p50_latency = 0;
        p95_latency = 0;
        p99_latency = 0;
        thread_safety_rate = 0;
        return;
    }
    
    double sum_recall = 0;
    std::vector<int64_t> latencies;
    int safe_count = 0;
    
    for (const auto& r : results) {
        if (r.thread_safe) {
            sum_recall += r.recall;
            latencies.push_back(r.latency_us);
            safe_count++;
        }
    }
    
    thread_safety_rate = (double)safe_count / results.size();
    
    if (safe_count > 0) {
        avg_recall = sum_recall / safe_count;
    } else {
        avg_recall = 0;
    }
    
    if (latencies.empty()) {
        avg_latency = 0;
        p50_latency = 0;
        p95_latency = 0;
        p99_latency = 0;
    } else {
        std::sort(latencies.begin(), latencies.end());
        avg_latency = 0;
        for (auto l : latencies) {
            avg_latency += l;
        }
        avg_latency /= latencies.size();
        
        p50_latency = latencies[static_cast<size_t>(latencies.size() * 0.50)];
        p95_latency = latencies[static_cast<size_t>(latencies.size() * 0.95)];
        p99_latency = latencies[static_cast<size_t>(latencies.size() * 0.99)];
    }
}

int main(int argc, char* argv[]) {
    size_t insert_st_id = 0, insert_count = 0;
    size_t efC = 500;
    size_t efs = 500;
    size_t k = 100;
    float partial_rebuild_ratio = 0.2f;  // Default 20%
    size_t rebuild_interval = 200;  // Trigger rebuild every N inserts
    int target_qps = 128;
    
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
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
        if (arg == "--partial_rebuild_ratio") partial_rebuild_ratio = std::stof(argv[i + 1]);
        if (arg == "--rebuild_interval") rebuild_interval = std::stoi(argv[i + 1]);
        if (arg == "--qps") target_qps = std::stoi(argv[i + 1]);
    }
    
    std::string index_path = paths["index_path"];
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string base_data_path = paths["base_data_path"];
    std::string metric_str = paths["metric"];
    std::string result_path = paths["result_path"];
    
    std::cout << "=== Insert + Query Concurrent Test ===" << std::endl;
    std::cout << "Index path: " << index_path << std::endl;
    std::cout << "Query path: " << test_query_path << std::endl;
    std::cout << "GT path: " << test_gt_path << std::endl;
    std::cout << "Base data path: " << base_data_path << std::endl;
    std::cout << "Result path: " << result_path << std::endl;
    std::cout << "Insert QPS: " << target_qps << std::endl;
    std::cout << "Query QPS: " << target_qps << std::endl;
    std::cout << "Partial rebuild ratio: " << partial_rebuild_ratio << std::endl;
    std::cout << "Rebuild interval: every " << rebuild_interval << " inserts" << std::endl;
    std::cout << "Will insert " << insert_count << " vectors starting from ID " << insert_st_id << std::endl;
    
    // Load data
    size_t test_number = 0, vecdim = 0;
    size_t test_gt_dim = 0;
    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    
    size_t base_number = 0;
    auto base_data = LoadData<float>(base_data_path, base_number, vecdim);
    
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
    
    // Resize if needed
    if (insert_st_id + insert_count > hnsw_ngfix->max_elements) {
        hnsw_ngfix->resize(insert_st_id + insert_count);
    }
    
    // Start threads
    std::thread query_t, insert_t;
    
    query_t = std::thread(query_thread, hnsw_ngfix, test_query, test_gt, 
                          test_number, test_gt_dim, vecdim, k, efs, target_qps);
    insert_t = std::thread(insert_thread, hnsw_ngfix, base_data, 
                          insert_st_id, insert_count, vecdim, efC, partial_rebuild_ratio, target_qps, rebuild_interval);
    
    // Wait for threads
    insert_t.join();
    query_t.join();
    
    // Calculate and output statistics
    double avg_recall, avg_latency, p50_latency, p95_latency, p99_latency, thread_safety_rate;
    calculate_stats(query_results, avg_recall, avg_latency, p50_latency, p95_latency, p99_latency, thread_safety_rate);
    
    std::cout << "\n=== Query Statistics ===" << std::endl;
    std::cout << "Total queries: " << query_results.size() << std::endl;
    std::cout << "Thread safety rate: " << thread_safety_rate * 100.0 << "%" << std::endl;
    std::cout << "Thread safety errors: " << thread_safety_errors.load() << std::endl;
    std::cout << "Average recall: " << avg_recall << std::endl;
    std::cout << "Average latency: " << avg_latency / 1000.0 << " ms" << std::endl;
    std::cout << "P50 latency: " << p50_latency / 1000.0 << " ms" << std::endl;
    std::cout << "P95 latency: " << p95_latency / 1000.0 << " ms" << std::endl;
    std::cout << "P99 latency: " << p99_latency / 1000.0 << " ms" << std::endl;
    
    // Calculate insert statistics
    if (!insert_results.empty()) {
        std::vector<int64_t> insert_latencies, rebuild_latencies;
        for (const auto& r : insert_results) {
            if (r.phase == "insert") {
                insert_latencies.push_back(r.latency_us);
            } else if (r.phase == "rebuild") {
                rebuild_latencies.push_back(r.latency_us);
            }
        }
        
        std::cout << "\n=== Insert Statistics ===" << std::endl;
        std::cout << "Total insert operations: " << insert_latencies.size() << std::endl;
        if (!insert_latencies.empty()) {
            std::sort(insert_latencies.begin(), insert_latencies.end());
            double avg_insert = 0;
            for (auto l : insert_latencies) avg_insert += l;
            avg_insert /= insert_latencies.size();
            std::cout << "Average insert latency: " << avg_insert / 1000.0 << " ms" << std::endl;
            std::cout << "P95 insert latency: " << insert_latencies[static_cast<size_t>(insert_latencies.size() * 0.95)] / 1000.0 << " ms" << std::endl;
        }
        if (!rebuild_latencies.empty()) {
            std::sort(rebuild_latencies.begin(), rebuild_latencies.end());
            double avg_rebuild = 0;
            for (auto l : rebuild_latencies) avg_rebuild += l;
            avg_rebuild /= rebuild_latencies.size();
            std::cout << "Total rebuild operations: " << rebuild_latencies.size() << std::endl;
            std::cout << "Average rebuild latency: " << avg_rebuild / 1000.0 << " ms" << std::endl;
            std::cout << "Min rebuild latency: " << rebuild_latencies[0] / 1000.0 << " ms" << std::endl;
            std::cout << "Max rebuild latency: " << rebuild_latencies.back() / 1000.0 << " ms" << std::endl;
        }
    }
    
    // Write detailed results to CSV
    std::ofstream output(result_path);
    output << "type,timestamp_us,id,recall,latency_us,progress,phase,thread_safe\n";
    
    // Write query results
    for (const auto& r : query_results) {
        output << "query," << r.timestamp_us << "," << r.query_id << "," << r.recall 
               << "," << r.latency_us << "," << r.insert_progress << ",," 
               << (r.thread_safe ? "1" : "0") << "\n";
    }
    
    // Write insert results
    for (const auto& r : insert_results) {
        output << "insert," << r.timestamp_us << "," << r.insert_id << ",,"
               << r.latency_us << ",," << r.phase << ",1\n";
    }
    
    output.close();
    
    std::cout << "\nResults written to: " << result_path << std::endl;
    
    delete[] test_query;
    delete[] test_gt;
    delete[] base_data;
    delete hnsw_ngfix;
    
    return 0;
}

