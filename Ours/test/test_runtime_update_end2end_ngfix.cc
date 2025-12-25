#include "../NGFix/ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <queue>
#include <ctime>

using namespace ngfixlib;

// Global variables for test control
std::atomic<bool> stop_test{false};
std::atomic<size_t> total_searches{0};
std::atomic<size_t> total_inserts{0};
std::atomic<size_t> successful_inserts{0};
std::atomic<size_t> failed_inserts{0};
std::atomic<size_t> total_deletes{0};
std::atomic<size_t> successful_deletes{0};
std::atomic<size_t> failed_deletes{0};

// Statistics for new node impact analysis
std::atomic<size_t> total_new_nodes_in_results{0};  // Count of new nodes (ID >= initial_n) in search results
std::atomic<size_t> total_result_nodes{0};  // Total nodes in all search results
size_t initial_index_size = 0;  // Initial index size (set in main)

// Thread-safe queue for inserted node IDs (for deletion)
std::mutex inserted_ids_mutex;
std::queue<id_t> inserted_ids_queue;  // Queue of inserted node IDs available for deletion

// Thread-safe counter for next insert ID (shared across all threads)
std::atomic<size_t> next_insert_id_global{0};

// Statistics per minute
struct MinuteStats {
    size_t query_count;
    double avg_recall;
    double avg_ndc;
    double avg_latency_ms;
    std::chrono::system_clock::time_point timestamp;
};

// Thread-safe queue for query results
struct QueryResult {
    size_t query_idx;
    float recall;
    size_t ndc;
    double latency_ms;
};

std::mutex stats_mutex;
std::vector<QueryResult> query_results_buffer;
std::vector<MinuteStats> minute_stats;

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

// Test thread: configurable QPS (default search 400, insert 100) - NGFix baseline (no graph fixing)
void TestThread(HNSW_NGFix<float>* index,
                float* train_queries,
                int* train_gt,
                float* additional_vectors,
                size_t num_train_queries,
                size_t num_additional_vectors,
                size_t vecdim,
                size_t k,
                size_t ef_search,
                size_t search_qps,
                size_t insert_qps) {
    
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<size_t> query_dist(0, num_train_queries - 1);
    std::uniform_int_distribution<size_t> vector_dist(0, num_additional_vectors - 1);
    
    // Rate control from user config
    // FIXED: Insert and delete QPS are distributed across 8 threads
    // Delete QPS is automatically calculated as insert_qps / 9 to maintain 9:1 ratio
    const double search_interval_ms = 1000.0 / std::max<size_t>(1, search_qps);
    const double insert_interval_ms = 1000.0 / std::max<size_t>(1, (insert_qps * 0.9) / 8.0);  // Distributed across 8 threads
    const double delete_interval_ms = 1000.0 / std::max<size_t>(1, (insert_qps * 0.1) / 8.0);  // Distributed across 8 threads, 9:1 ratio
    
    auto last_search_time = std::chrono::steady_clock::now();
    auto last_insert_time = std::chrono::steady_clock::now();
    auto last_delete_time = std::chrono::steady_clock::now();
    
    // FIXED: Use atomic counter for thread-safe insert ID allocation
    // Each thread will get unique IDs using atomic fetch_add
    size_t insert_count_since_last_delete = 0;  // Track insert count for 10:1 ratio
    
    while(!stop_test.load()) {
        auto now = std::chrono::steady_clock::now();
        
        // Check if we should do a search (400 QPS)
        auto search_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - last_search_time).count() / 1000.0;
        
        if(search_elapsed >= search_interval_ms) {
            // Perform search
            size_t q_idx = query_dist(rng);
            float* query_data = train_queries + q_idx * vecdim;
            int* gt = train_gt + q_idx * k;
            
            try {
                auto search_start = std::chrono::high_resolution_clock::now();
                size_t ndc = 0;
                auto results = index->searchKnn(query_data, k, ef_search, ndc);
                auto search_end = std::chrono::high_resolution_clock::now();
                
                double latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    search_end - search_start).count() / 1000.0;
                
                float recall = CalculateRecall(results, gt, k);
                
                // Analyze new node impact: count how many new nodes (ID >= initial_index_size) appear in results
                size_t new_nodes_in_result = 0;
                for(const auto& p : results) {
                    if(p.second >= initial_index_size) {
                        new_nodes_in_result++;
                    }
                }
                total_new_nodes_in_results.fetch_add(new_nodes_in_result);
                total_result_nodes.fetch_add(results.size());
                
                // DEBUG: Log first few searches with new nodes in results
                static std::atomic<size_t> debug_search_count{0};
                if(new_nodes_in_result > 0 && debug_search_count.fetch_add(1) < 10) {
                    std::cout << "[DEBUG SEARCH] Query " << q_idx << " found " << new_nodes_in_result 
                              << " new nodes in results (total " << results.size() << " results)" << std::endl;
                }
                
                // Store result for statistics
                {
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    query_results_buffer.push_back({q_idx, recall, ndc, latency_ms});
                }
                
                total_searches.fetch_add(1);
                // FIXED: Update to target time point to maintain accurate QPS
                last_search_time += std::chrono::microseconds((int64_t)(search_interval_ms * 1000));
                if(last_search_time > now) {
                last_search_time = now;
                }
            } catch(...) {
                // Ignore errors, but still update time to maintain QPS
                last_search_time += std::chrono::microseconds((int64_t)(search_interval_ms * 1000));
                if(last_search_time > now) {
                    last_search_time = now;
                }
            }
        }
        
        // Check if we should do an insert (100 QPS)
        auto insert_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - last_insert_time).count() / 1000.0;
        
        // FIXED: Get next insert ID atomically (thread-safe)
        size_t current_next_id = next_insert_id_global.load();
        if(insert_elapsed >= insert_interval_ms && current_next_id < index->max_elements) {
            // Perform insert - ALWAYS use additional_vectors (they are different from base data)
            if(additional_vectors != nullptr && num_additional_vectors > 0) {
                size_t vec_idx = vector_dist(rng);
                float* vec_data = additional_vectors + vec_idx * vecdim;
                
                try {
                    // FIXED: Atomically get and increment insert ID
                    id_t new_id = next_insert_id_global.fetch_add(1);
                    size_t n_before = index->n.load();
                    size_t efC = 200;
                    index->InsertPoint(new_id, efC, vec_data);
                    size_t n_after = index->n.load();
                    
                    if(n_after > n_before) {
                        // Insert succeeded - add to queue for potential deletion
                        {
                            std::lock_guard<std::mutex> lock(inserted_ids_mutex);
                            inserted_ids_queue.push(new_id);
                        }
                        successful_inserts.fetch_add(1);
                        insert_count_since_last_delete++;
                    } else {
                        failed_inserts.fetch_add(1);
                    }
                    
                    total_inserts.fetch_add(1);
                    // FIXED: Update to target time point to maintain accurate QPS
                    last_insert_time += std::chrono::microseconds((int64_t)(insert_interval_ms * 1000));
                    if(last_insert_time > now) {
                    last_insert_time = now;
                    }
                } catch(...) {
                    failed_inserts.fetch_add(1);
                    total_inserts.fetch_add(1);
                }
            }
        }
        
        // Check if we should do a delete (maintaining 10:1 ratio with insert)
        auto delete_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - last_delete_time).count() / 1000.0;
        
        // Delete condition: we have 10+ inserts accumulated OR time interval elapsed (to prevent starvation)
        bool should_delete = (insert_count_since_last_delete >= 10) || 
                            (delete_elapsed >= delete_interval_ms && insert_count_since_last_delete > 0);
        
        if(should_delete) {
            // Try to get an inserted node ID to delete
            id_t node_to_delete = 0;
            bool has_node_to_delete = false;
            
            {
                std::lock_guard<std::mutex> lock(inserted_ids_mutex);
                if(!inserted_ids_queue.empty()) {
                    node_to_delete = inserted_ids_queue.front();
                    inserted_ids_queue.pop();
                    has_node_to_delete = true;
                }
            }
            
            if(has_node_to_delete) {
                // FIXED: Only count as attempted delete if we actually try to delete
                total_deletes.fetch_add(1);
                
                try {
                    // Check if node is still valid (not already deleted)
                    if(node_to_delete < index->max_elements && node_to_delete < index->n.load() && 
                       !index->is_deleted(node_to_delete)) {
                        // Check if we're about to delete the entry point
                        // Note: DeletePointByFlag has protection, but we should verify entry point after deletion
                        size_t old_entry_point = index->entry_point;
                        
                        // Perform delete using NGFix method
                        index->DeletePointByFlag(node_to_delete);
                        
                        // If entry point was deleted or became invalid, it will be handled by searchKnn
                        // But we can proactively check and reset if needed
                        if(index->entry_point >= index->max_elements || 
                           index->entry_point >= index->n.load() || 
                           index->is_deleted(index->entry_point)) {
                            // Entry point is invalid, SetEntryPoint will be called on next insert
                            // For now, just log it (searchKnn will handle it)
                        }
                        
                        successful_deletes.fetch_add(1);
                        insert_count_since_last_delete = 0;  // Reset counter after successful delete
                    } else {
                        // Node already deleted or invalid - count as failed delete
                        failed_deletes.fetch_add(1);
                        if(insert_count_since_last_delete >= 10) {
                            insert_count_since_last_delete = 0;
                        }
                    }
                    // FIXED: Update to target time point to maintain accurate QPS
                    last_delete_time += std::chrono::microseconds((int64_t)(delete_interval_ms * 1000));
                    if(last_delete_time > now) {
                        last_delete_time = now;
                    }
                } catch(const std::exception& e) {
                    // FIXED: total_deletes already incremented above, only increment failed_deletes here
                    failed_deletes.fetch_add(1);
                    // FIXED: Update to target time point to maintain accurate QPS
                    last_delete_time += std::chrono::microseconds((int64_t)(delete_interval_ms * 1000));
                    if(last_delete_time > now) {
                        last_delete_time = now;
                    }
                    // Reset counter on error if we had enough inserts
                    if(insert_count_since_last_delete >= 10) {
                        insert_count_since_last_delete = 0;
                    }
                } catch(...) {
                    // FIXED: total_deletes already incremented above, only increment failed_deletes here
                    failed_deletes.fetch_add(1);
                    // FIXED: Update to target time point to maintain accurate QPS
                    last_delete_time += std::chrono::microseconds((int64_t)(delete_interval_ms * 1000));
                    if(last_delete_time > now) {
                        last_delete_time = now;
                    }
                    // Reset counter on error if we had enough inserts
                    if(insert_count_since_last_delete >= 10) {
                        insert_count_since_last_delete = 0;
                    }
                }
            } else {
                // No nodes available for deletion yet, wait for more inserts
                // Only update time if we've waited long enough to prevent busy waiting
                if(delete_elapsed >= delete_interval_ms * 2) {
                    // FIXED: Update to target time point to maintain accurate QPS
                    last_delete_time += std::chrono::microseconds((int64_t)(delete_interval_ms * 1000));
                    if(last_delete_time > now) {
                        last_delete_time = now;
                    }  // Prevent too frequent checks
                    // If we have enough inserts but no nodes to delete, reset counter
                    if(insert_count_since_last_delete >= 10) {
                        insert_count_since_last_delete = 0;
                    }
                }
            }
        }
        
        // Small sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

// Statistics collection thread: collect stats every minute
void StatisticsThread(HNSW_NGFix<float>* index,
                      size_t vecdim,
                      size_t k,
                      size_t ef_search,
                      const std::string& result_file) {
    
    auto start_time = std::chrono::steady_clock::now();
    size_t minute_count = 0;
    
    while(!stop_test.load()) {
        std::this_thread::sleep_for(std::chrono::minutes(1));
        
        if(stop_test.load()) break;
        
        minute_count++;
        auto now = std::chrono::steady_clock::now();
        
        // Collect statistics from buffer
        std::vector<QueryResult> current_results;
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            current_results = query_results_buffer;
            query_results_buffer.clear();  // Clear for next minute
        }
        
        if(current_results.empty()) {
            std::cout << "\n[Minute " << minute_count << "] No queries in this minute" << std::endl;
            continue;
        }
        
        // Calculate statistics
        size_t query_count = current_results.size();
        double sum_recall = 0.0;
        double sum_ndc = 0.0;
        double sum_latency = 0.0;
        
        for(const auto& result : current_results) {
            sum_recall += result.recall;
            sum_ndc += result.ndc;
            sum_latency += result.latency_ms;
        }
        
        double avg_recall = sum_recall / query_count;
        double avg_ndc = sum_ndc / query_count;
        double avg_latency = sum_latency / query_count;
        
        MinuteStats stats;
        stats.query_count = query_count;
        stats.avg_recall = avg_recall;
        stats.avg_ndc = avg_ndc;
        stats.avg_latency_ms = avg_latency;
        stats.timestamp = std::chrono::system_clock::now();
        
        minute_stats.push_back(stats);
        
        // Get current index size
        size_t current_index_size = index->n.load();
        
        // Print statistics
        std::cout << "\n=== Minute " << minute_count << " Statistics (NGFix) ===" << std::endl;
        std::cout << "Query count: " << query_count << std::endl;
        std::cout << "Average recall: " << std::fixed << std::setprecision(4) << avg_recall << std::endl;
        std::cout << "Average NDC: " << std::fixed << std::setprecision(2) << avg_ndc << std::endl;
        std::cout << "Average latency (ms): " << std::fixed << std::setprecision(2) << avg_latency << std::endl;
        std::cout << "Total searches: " << total_searches.load() << std::endl;
        std::cout << "Total inserts (attempted): " << total_inserts.load() << std::endl;
        std::cout << "Successful inserts: " << successful_inserts.load() << std::endl;
        std::cout << "Failed inserts: " << failed_inserts.load() << std::endl;
        std::cout << "Total deletes (attempted): " << total_deletes.load() << std::endl;
        std::cout << "Successful deletes: " << successful_deletes.load() << std::endl;
        std::cout << "Failed deletes: " << failed_deletes.load() << std::endl;
        std::cout << "Insert/Delete ratio: " << std::fixed << std::setprecision(2) 
                  << (successful_inserts.load() > 0 ? (double)successful_inserts.load() / std::max((size_t)1, successful_deletes.load()) : 0.0) 
                  << ":1" << std::endl;
        std::cout << "Current index size (n): " << current_index_size << std::endl;
        // New node impact analysis
        size_t total_results = total_result_nodes.load();
        size_t total_new = total_new_nodes_in_results.load();
        if(total_results > 0) {
            double new_node_ratio = (double)total_new / total_results * 100.0;
            std::cout << "New nodes in results: " << total_new << " / " << total_results 
                      << " (" << std::fixed << std::setprecision(2) << new_node_ratio << "%)" << std::endl;
        }
        
        // Save to file
        std::ofstream out(result_file, std::ios::app);
        if(out.is_open()) {
            auto time_t = std::chrono::system_clock::to_time_t(stats.timestamp);
            out << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                << query_count << ","
                << avg_recall << ","
                << avg_ndc << ","
                << avg_latency << ","
                << total_searches.load() << ","
                << total_inserts.load() << ","
                << successful_inserts.load() << ","
                << failed_inserts.load() << ","
                << total_deletes.load() << ","
                << successful_deletes.load() << ","
                << failed_deletes.load() << ","
                << current_index_size << "\n";
            out.close();
        }
    }
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::unordered_map<std::string, std::string> paths;
    for(int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--base_index_path" && i + 1 < argc)
            paths["base_index_path"] = argv[++i];
        if(arg == "--train_query_path" && i + 1 < argc)
            paths["train_query_path"] = argv[++i];
        if(arg == "--train_gt_path" && i + 1 < argc)
            paths["train_gt_path"] = argv[++i];
        if(arg == "--additional_vector_path" && i + 1 < argc)
            paths["additional_vector_path"] = argv[++i];
        if(arg == "--metric" && i + 1 < argc)
            paths["metric"] = argv[++i];
        if(arg == "--result_dir" && i + 1 < argc)
            paths["result_dir"] = argv[++i];
        if(arg == "--K" && i + 1 < argc)
            paths["K"] = argv[++i];
        if(arg == "--duration_minutes" && i + 1 < argc)
            paths["duration_minutes"] = argv[++i];
        if(arg == "--search_qps" && i + 1 < argc)
            paths["search_qps"] = argv[++i];
        if(arg == "--insert_qps" && i + 1 < argc)
            paths["insert_qps"] = argv[++i];
    }
    
    // Default values
    std::string base_index_path = paths["base_index_path"];
    std::string train_query_path = paths["train_query_path"];
    std::string train_gt_path = paths["train_gt_path"];
    std::string additional_vector_path = paths["additional_vector_path"];
    std::string result_dir = paths.count("result_dir") ? paths["result_dir"] : "./data/runtime_update_test";
    std::string metric_str = paths.count("metric") ? paths["metric"] : "ip_float";
    size_t k = paths.count("K") ? std::stoi(paths["K"]) : 100;
    size_t duration_minutes = paths.count("duration_minutes") ? std::stoi(paths["duration_minutes"]) : 60;
    size_t ef_search = 1000;  // Fixed efSearch=1000
    size_t search_qps = paths.count("search_qps") ? std::stoul(paths["search_qps"]) : 1000;
    size_t insert_qps = paths.count("insert_qps") ? std::stoul(paths["insert_qps"]) : 140;
    
    std::cout << "=== Runtime Update End-to-End Test (NGFix Baseline) ===" << std::endl;
    std::cout << "Base index: " << base_index_path << std::endl;
    std::cout << "Train queries: " << train_query_path << std::endl;
    std::cout << "Train GT: " << train_gt_path << std::endl;
    std::cout << "Additional vectors: " << additional_vector_path << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "ef_search: " << ef_search << std::endl;
    std::cout << "Duration: " << duration_minutes << " minutes (" << (duration_minutes / 60.0) << " hours)" << std::endl;
    std::cout << "Search QPS: " << search_qps << std::endl;
    std::cout << "Insert QPS: " << insert_qps << std::endl;
    std::cout << "Note: No graph fixing capability" << std::endl;
    std::cout << std::endl;
    
    // Load data
    size_t train_number = 0;
    size_t train_gt_dim = 0, vecdim = 0;
    size_t additional_number = 0, additional_vecdim = 0;
    
    auto train_query = LoadData<float>(train_query_path, train_number, vecdim);
    
    // Determine metric first
    ngfixlib::Metric metric;
    if(metric_str == "ip_float") {
        metric = ngfixlib::IP_float;
    } else if(metric_str == "l2_float") {
        metric = ngfixlib::L2_float;
    } else {
        std::cerr << "ERROR: Unsupported metric type." << std::endl;
        delete[] train_query;
        return 1;
    }
    
    // Load GT data
    int* train_gt = nullptr;
    size_t actual_train_number = train_number;
    
    if(!train_gt_path.empty()) {
        std::cout << "=== Loading Ground Truth from File ===" << std::endl;
        train_gt = LoadData<int>(train_gt_path, actual_train_number, train_gt_dim);
        
        if(train_gt_dim < k) {
            std::cerr << "WARNING: GT dimension (" << train_gt_dim 
                      << ") is less than K (" << k << "). Using available GT." << std::endl;
        }
        
        if(actual_train_number != train_number) {
            std::cerr << "WARNING: GT number (" << actual_train_number 
                      << ") != Query number (" << train_number << "). Using minimum." << std::endl;
            actual_train_number = std::min(actual_train_number, train_number);
        }
        
        std::cout << "Loaded " << actual_train_number << " GT entries, dimension: " << train_gt_dim << std::endl;
        std::cout << std::endl;
    } else {
        std::cerr << "ERROR: train_gt_path is required!" << std::endl;
        delete[] train_query;
        return 1;
    }
    
    train_number = actual_train_number;
    
    // Load additional vectors (required for insert)
    float* additional_vectors = nullptr;
    
    if(!additional_vector_path.empty()) {
        auto loaded_additional = LoadData<float>(additional_vector_path, additional_number, additional_vecdim);
        if(additional_vecdim == vecdim && additional_number > 0) {
            additional_vectors = loaded_additional;
            std::cout << "Loaded " << additional_number << " additional vectors from file" << std::endl;
        } else {
            std::cerr << "ERROR: Additional vector file dimension mismatch! Expected " << vecdim 
                      << ", got " << additional_vecdim << std::endl;
            delete[] train_query;
            delete[] train_gt;
            return 1;
        }
    } else {
        std::cerr << "ERROR: additional_vector_path is required for insert operations!" << std::endl;
        delete[] train_query;
        delete[] train_gt;
        return 1;
    }
    
    std::cout << "Loaded " << train_number << " training queries" << std::endl;
    std::cout << "Loaded " << additional_number << " additional vectors" << std::endl;
    std::cout << "Vector dimension: " << vecdim << std::endl;
    std::cout << std::endl;
    
    // Load index
    std::cout << "=== Loading Index (NGFix) ===" << std::endl;
    HNSW_NGFix<float>* index = nullptr;
    try {
        index = new HNSW_NGFix<float>(metric, base_index_path);
        index->printGraphInfo();
        
        // Check dimension match
        size_t index_dim = index->dim;
        std::cout << "Index dimension: " << index_dim << std::endl;
        std::cout << "Query dimension: " << vecdim << std::endl;
        
        if(index_dim != vecdim) {
            std::cerr << "ERROR: Dimension mismatch! Index=" << index_dim 
                      << ", Query=" << vecdim << std::endl;
            std::cerr << "Index file dimension does not match query dimension." << std::endl;
            std::cerr << "Please use an index file built with " << vecdim << " dimensional vectors." << std::endl;
            delete[] train_query;
            delete[] train_gt;
            if(additional_vectors != nullptr) delete[] additional_vectors;
            return 1;
        }
    } catch(const std::exception& e) {
        std::cerr << "ERROR loading index: " << e.what() << std::endl;
        std::cerr << "This may be due to dimension mismatch or incompatible index format." << std::endl;
        delete[] train_query;
        delete[] train_gt;
        if(additional_vectors != nullptr) delete[] additional_vectors;
        return 1;
    }
    
    // Check if we need to resize for inserts
    size_t current_n = index->n.load();
    size_t max_elements = index->max_elements;
    initial_index_size = current_n;  // Store initial index size for new node analysis
    next_insert_id_global.store(current_n);  // Initialize atomic counter to current index size
    std::cout << "Current index size: " << current_n << std::endl;
    std::cout << "Max elements: " << max_elements << std::endl;
    
    // Always resize to allow inserts (based on configured insert_qps)
    size_t inserts_needed = duration_minutes * 60 * insert_qps;  // Total inserts in test duration
    size_t new_max = max_elements + std::max(inserts_needed, max_elements / 5);  // At least 20% more
    
    if(new_max > max_elements) {
        std::cout << "Resizing index to allow inserts..." << std::endl;
        std::cout << "  Current max: " << max_elements << std::endl;
        std::cout << "  New max: " << new_max << std::endl;
        std::cout << "  Expected inserts: " << inserts_needed << std::endl;
        index->resize(new_max);
        std::cout << "Resize complete. New max_elements: " << index->max_elements << std::endl;
    } else {
        std::cout << "Index has enough space for inserts" << std::endl;
    }
    
    std::cout << std::endl;
    
    // Create result file
    std::string result_file = result_dir + "/runtime_update_results_ngfix.csv";
    std::ofstream out(result_file);
    if(out.is_open()) {
        out << "timestamp,query_count,avg_recall,avg_ndc,avg_latency_ms,total_searches,total_inserts,successful_inserts,failed_inserts,total_deletes,successful_deletes,failed_deletes,index_size\n";
        out.close();
    }
    
    // Start threads (NO connectivity enhancement thread for NGFix baseline)
    // FIXED: Create 8 test threads, each with 125 QPS (total 1000 QPS)
    const size_t num_test_threads = 8;  // 8 threads * 125 QPS = 1000 QPS total
    const size_t qps_per_thread = 125;  // Each thread handles 125 QPS
    std::vector<std::thread> test_threads;
    for(size_t i = 0; i < num_test_threads; ++i) {
        test_threads.emplace_back(TestThread, index, train_query, train_gt, additional_vectors,
                           train_number, additional_number, vecdim, k, ef_search,
                                 qps_per_thread, insert_qps);
    }
    
    std::thread stats_thread(StatisticsThread, index, vecdim, k, ef_search, result_file);
    
    std::cout << "=== Started test threads (NGFix baseline, no graph fixing) ===" << std::endl;
    std::cout << "Test will run for " << duration_minutes << " minutes" << std::endl;
    std::cout << std::endl;
    
    // Wait for duration
    std::this_thread::sleep_for(std::chrono::minutes(duration_minutes));
    
    // Stop test
    std::cout << "\n=== Stopping Test ===" << std::endl;
    stop_test.store(true);
    
    // Wait for threads
    // Join all test threads
    for(auto& t : test_threads) {
        t.join();
    }
    stats_thread.join();
    
    delete index;
    delete[] train_query;
    delete[] train_gt;
    if(additional_vectors != nullptr) {
        delete[] additional_vectors;
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "Total searches: " << total_searches.load() << std::endl;
    std::cout << "Total inserts (attempted): " << total_inserts.load() << std::endl;
    std::cout << "Successful inserts: " << successful_inserts.load() << std::endl;
    std::cout << "Failed inserts: " << failed_inserts.load() << std::endl;
    std::cout << "Total deletes (attempted): " << total_deletes.load() << std::endl;
    std::cout << "Successful deletes: " << successful_deletes.load() << std::endl;
    std::cout << "Failed deletes: " << failed_deletes.load() << std::endl;
    std::cout << "Final Insert/Delete ratio: " << std::fixed << std::setprecision(2) 
              << (successful_inserts.load() > 0 ? (double)successful_inserts.load() / std::max((size_t)1, successful_deletes.load()) : 0.0) 
              << ":1" << std::endl;
    std::cout << "Results saved to: " << result_file << std::endl;
    
    return 0;
}
