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
#include <random>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <sys/time.h>
#include <iomanip>
#include <sstream>
using namespace ngfixlib;

struct QueryResult {
    int64_t timestamp_us;
    int query_id;
    float recall;
    int64_t latency_us;
};

struct MinuteStats {
    int64_t minute;
    double avg_recall;
    int query_count;
};

struct NGFixUpdateEvent {
    int64_t timestamp_us;
    int64_t duration_us;
    int ops_count_before;  // Number of insert/delete ops before this update
    std::string update_type;  // "insert_batch" or "delete_batch"
};

std::mutex result_mutex;
std::vector<QueryResult> all_results;
std::vector<NGFixUpdateEvent> ngfix_updates;
std::atomic<bool> test_running{true};
std::atomic<int> insert_count{0};
std::atomic<int> delete_count{0};
std::atomic<int> total_ops_count{0};  // Total insert + delete operations
std::atomic<int> noisy_insert_count{0};  // Count of inserts with noise

// For tracking deleted nodes that can be reused
std::mutex deleted_nodes_mutex;
std::queue<id_t> deleted_nodes;
std::unordered_set<id_t> deleted_set;
std::unordered_map<id_t, std::vector<float>> deleted_node_data; // Store data for deleted nodes

// Base graph size (8M)
const size_t BASE_SIZE = 8000000;
// Additional data size (2M)
const size_t ADDITIONAL_SIZE = 2000000;

void search_thread(HNSW_NGFix<float>* hnsw_ngfix, 
                   float* test_query, 
                   int* test_gt,
                   size_t test_number,
                   size_t test_gt_dim,
                   size_t vecdim,
                   size_t k,
                   size_t efs,
                   int target_qps,
                   size_t base_size,
                   std::chrono::high_resolution_clock::time_point global_start_time,
                   size_t selected_query_count) {
    const int64_t interval_us = 1000000 / target_qps;
    int query_idx = 0;
    int query_count = 0;
    int64_t current_minute = -1;

    while (test_running.load()) {
        auto query_start = std::chrono::high_resolution_clock::now();
        int64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            query_start - global_start_time).count();
        int64_t query_minute = timestamp_us / (60LL * 1000000LL);
        
        // Reset query index at the start of each new minute
        if (query_minute > current_minute) {
            query_idx = 0;
            current_minute = query_minute;
        }
        
        // If we've completed all selected queries, wait until next minute
        if (query_idx >= selected_query_count) {
            // Calculate time until next minute
            int64_t current_second = timestamp_us / 1000000LL;
            int64_t next_minute_start_us = (query_minute + 1) * 60LL * 1000000LL;
            int64_t wait_us = next_minute_start_us - timestamp_us;
            if (wait_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
            }
            continue;
        }
        
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
        
        // Calculate recall - only consider results in base graph (first 8M)
        std::unordered_set<id_t> gtset;
        auto gt = test_gt + query_idx * test_gt_dim;
        
        // Filter GT to only include nodes in base graph (first 8M)
        int valid_gt_count = 0;
        for(int i = 0; i < k; ++i) {
            if(gt[i] < base_size) {
                gtset.insert(gt[i]);
                valid_gt_count++;
            }
        }
        
        // Filter search results to only include nodes in base graph
        int acc = 0;
        for(int i = 0; i < k; ++i) {
            if(aknns[i].second < base_size) {
                if(gtset.find(aknns[i].second) != gtset.end()) {
                    ++acc;
                    gtset.erase(aknns[i].second);
                }
            }
        }
        
        // Calculate recall based on valid GT count
        float recall = (valid_gt_count > 0) ? (float)acc / valid_gt_count : 0.0f;
        
        // Store result
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            all_results.push_back({timestamp_us, query_idx, recall, latency_us});
        }
        
        query_count++;
        query_idx++;  // Move to next query in the selected set
        
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

// Function to add random noise to a vector
void addRandomNoise(float* data, size_t vecdim, std::mt19937& rng, float noise_scale = 0.01f) {
    std::uniform_real_distribution<float> noise_dist(-noise_scale, noise_scale);
    for (size_t i = 0; i < vecdim; ++i) {
        data[i] += noise_dist(rng);
    }
}

void random_ops_thread(HNSW_NGFix<float>* hnsw_ngfix,
                      float* additional_data,
                      size_t vecdim,
                      size_t efC,
                      int target_qps,
                      size_t base_size,
                      size_t additional_size,
                      std::mt19937& rng,
                      std::chrono::high_resolution_clock::time_point global_start_time) {
    const int64_t interval_us = 1000000 / target_qps;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::uniform_real_distribution<float> op_dist(0.0f, 1.0f);
    std::uniform_int_distribution<id_t> additional_dist(base_size, base_size + additional_size - 1);
    
    // For selecting 12 inserts per second to add noise
    // With 128 QPS, ~64 inserts per second, we want 12 with noise
    // So probability = 12/64 = 0.1875
    const float noise_probability = 12.0f / (target_qps / 2.0f);  // target_qps/2 is approximate insert rate
    
    int ops_count = 0;
    int insert_ops_in_current_second = 0;
    int noisy_inserts_in_current_second = 0;
    int64_t current_second = -1;
    
    // Track which inserts in current second should have noise (randomly select 12)
    std::vector<bool> noise_selection;  // Will be resized to insert count, 12 randomly set to true
    
    while (test_running.load()) {
        auto op_start = std::chrono::high_resolution_clock::now();
        int64_t op_timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            op_start - global_start_time).count();
        int64_t op_second = op_timestamp_us / 1000000LL;
        
        // Reset insert counter for new second and pre-select 12 inserts for noise
        if (op_second > current_second) {
            // Report statistics for previous second (if not first second)
            if (current_second >= 0 && insert_ops_in_current_second > 0) {
                std::cout << "[Second " << current_second << "] " << insert_ops_in_current_second 
                          << " inserts, " << noisy_inserts_in_current_second << " with noise" << std::endl;
            }
            
            insert_ops_in_current_second = 0;
            noisy_inserts_in_current_second = 0;
            current_second = op_second;
            
            // Pre-select 12 random positions for noise in this second
            // Estimate: with 128 QPS, ~64 inserts per second
            int estimated_inserts = target_qps / 2;
            noise_selection.clear();
            noise_selection.resize(estimated_inserts, false);
            
            // Randomly select 12 positions
            std::vector<int> positions(estimated_inserts);
            std::iota(positions.begin(), positions.end(), 0);
            std::shuffle(positions.begin(), positions.end(), rng);
            
            for (int i = 0; i < std::min(12, estimated_inserts); ++i) {
                noise_selection[positions[i]] = true;
            }
        }
        
        // Randomly choose insert or delete (50% each)
        bool is_insert = op_dist(rng) < 0.5f;
        
        if (is_insert) {
            insert_ops_in_current_second++;
            
            // Insert: choose from additional data or deleted nodes
            id_t insert_id;
            bool use_deleted = false;
            
            {
                std::lock_guard<std::mutex> lock(deleted_nodes_mutex);
                if (!deleted_nodes.empty() && op_dist(rng) < 0.5f) {
                    // Use a deleted node
                    insert_id = deleted_nodes.front();
                    deleted_nodes.pop();
                    deleted_set.erase(insert_id);
                    use_deleted = true;
                } else {
                    // Use additional data
                    insert_id = additional_dist(rng);
                }
            }
            
            if (use_deleted || insert_id < base_size + additional_size) {
                // Get data for insertion
                float* original_data;
                std::vector<float> data_copy;
                bool need_noise = false;
                
                if (use_deleted) {
                    // Reuse stored data for deleted node
                    std::lock_guard<std::mutex> lock(deleted_nodes_mutex);
                    if (deleted_node_data.find(insert_id) != deleted_node_data.end()) {
                        data_copy = deleted_node_data[insert_id];
                        original_data = data_copy.data();
                    } else {
                        // Fallback to additional data
                        original_data = additional_data + (insert_id - base_size) * vecdim;
                    }
                } else {
                    original_data = additional_data + (insert_id - base_size) * vecdim;
                }
                
                // Check if this insert should have noise (pre-selected at start of second)
                int insert_index = insert_ops_in_current_second - 1;  // 0-based index
                if (insert_index < (int)noise_selection.size() && 
                    noise_selection[insert_index]) {
                    need_noise = true;
                    noisy_inserts_in_current_second++;
                }
                
                // Prepare data for insertion (copy if we need to add noise)
                float* data_to_insert;
                if (need_noise) {
                    // Copy data and add noise
                    if (use_deleted && !data_copy.empty()) {
                        // data_copy already exists
                    } else {
                        data_copy = std::vector<float>(original_data, original_data + vecdim);
                    }
                    addRandomNoise(data_copy.data(), vecdim, rng, 0.01f);
                    data_to_insert = data_copy.data();
                    noisy_insert_count++;
                } else {
                    // Use original data directly
                    if (use_deleted && !data_copy.empty()) {
                        data_to_insert = data_copy.data();
                    } else {
                        data_to_insert = original_data;
                    }
                }
                
                try {
                    hnsw_ngfix->InsertPoint(insert_id, efC, data_to_insert);
                    insert_count++;
                    total_ops_count++;
                    
                    // Note: Insert operations don't automatically trigger NGFix update
                    // NGFix update is only triggered by DeleteAllFlagPointsByNGFix() calls
                    
                    if (ops_count % 100 == 0) {
                        std::cout << "Inserted node " << insert_id 
                                  << (need_noise ? " [WITH NOISE]" : "") 
                                  << " (total inserts: " << insert_count.load() 
                                  << ", noisy: " << noisy_insert_count.load() << ")" << std::endl;
                    }
                } catch (...) {
                    // Skip if insertion fails
                }
            }
        } else {
            // Delete: choose a random node from base_size to base_size + additional_size
            id_t delete_id = additional_dist(rng);
            
            // Check if already deleted
            {
                std::lock_guard<std::mutex> lock(deleted_nodes_mutex);
                if (deleted_set.find(delete_id) != deleted_set.end()) {
                    // Already deleted, skip
                    goto sleep_op;
                }
            }
            
            try {
                // Store data before deletion - we need to get it from additional_data
                // Since we can't access getData directly, we'll use the additional_data
                float* node_data = additional_data + (delete_id - base_size) * vecdim;
                std::vector<float> data_copy(node_data, node_data + vecdim);
                
                hnsw_ngfix->DeletePointByFlag(delete_id);
                delete_count++;
                total_ops_count++;
                
                {
                    std::lock_guard<std::mutex> lock(deleted_nodes_mutex);
                    deleted_nodes.push(delete_id);
                    deleted_set.insert(delete_id);
                    deleted_node_data[delete_id] = std::move(data_copy);
                }
                
                // Periodically clean up deleted nodes and trigger NGFix update
                int current_delete_count = delete_count.load();
                if (current_delete_count % 1000 == 0 && current_delete_count > 0) {
                    auto update_start = std::chrono::high_resolution_clock::now();
                    int64_t update_start_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        update_start - global_start_time).count();
                    
                    hnsw_ngfix->DeleteAllFlagPointsByNGFix();
                    
                    auto update_end = std::chrono::high_resolution_clock::now();
                    int64_t update_end_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        update_end - global_start_time).count();
                    int64_t update_duration_us = update_end_us - update_start_us;
                    
                    {
                        std::lock_guard<std::mutex> lock(result_mutex);
                        ngfix_updates.push_back({update_start_us, update_duration_us, 
                                                total_ops_count.load(), "delete_batch"});
                    }
                    
                    std::cout << "NGFix update (delete_batch) completed at " << update_start_us / 1000000.0 
                              << "s, duration: " << update_duration_us / 1000.0 << "ms" << std::endl;
                }
                
                if (ops_count % 100 == 0) {
                    std::cout << "Deleted node " << delete_id << " (total deletes: " << delete_count.load() << ")" << std::endl;
                }
            } catch (...) {
                // Skip if deletion fails
            }
        }
        
        ops_count++;
        
sleep_op:
        // Sleep to maintain target QPS
        auto op_end = std::chrono::high_resolution_clock::now();
        auto op_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            op_end - op_start).count();
        int64_t sleep_us = interval_us - op_duration;
        if (sleep_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }
    }
    
    std::cout << "Random ops thread completed. Total ops: " << ops_count << std::endl;
}

void stats_thread(const std::string& output_path, int64_t test_duration_seconds, 
                  std::chrono::high_resolution_clock::time_point global_start_time) {
    std::vector<MinuteStats> minute_stats;
    int64_t last_recorded_minute = -1;
    
    while (test_running.load()) {
        auto current_time = std::chrono::high_resolution_clock::now();
        int64_t elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - global_start_time).count();
        
        if (elapsed_seconds >= test_duration_seconds) {
            break;
        }
        
        int64_t current_minute = elapsed_seconds / 60;
        
        // Record stats at the end of each minute
        // Wait until we're at least 1 second into the next minute to ensure all queries from previous minute are recorded
        if (current_minute > last_recorded_minute && elapsed_seconds >= ((last_recorded_minute + 1) * 60 + 1)) {
            // Calculate average recall for the completed minute
            int64_t minute_to_record = last_recorded_minute + 1;
            std::lock_guard<std::mutex> lock(result_mutex);
            
            // Use elapsed time in microseconds for comparison
            int64_t minute_start_us = minute_to_record * 60LL * 1000000LL;
            int64_t minute_end_us = (minute_to_record + 1) * 60LL * 1000000LL;
            
            double sum_recall = 0.0;
            int count = 0;
            
            for (const auto& result : all_results) {
                // Check if result is in this minute's time window
                if (result.timestamp_us >= minute_start_us && result.timestamp_us < minute_end_us) {
                    sum_recall += result.recall;
                    count++;
                }
            }
            
            double avg_recall = (count > 0) ? sum_recall / count : 0.0;
            minute_stats.push_back({minute_to_record, avg_recall, count});
            last_recorded_minute = minute_to_record;
            
            std::cout << "Minute " << minute_to_record << ": avg_recall=" << std::fixed << std::setprecision(6) << avg_recall 
                      << ", query_count=" << count << ", total_results=" << all_results.size() << std::endl;
        }
        
        // Sleep for a short time to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Wait a bit to ensure all queries are recorded
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Recalculate all minutes from scratch to ensure accuracy
    std::vector<NGFixUpdateEvent> ngfix_updates_copy;
    {
        std::lock_guard<std::mutex> lock(result_mutex);
        
        // Find the maximum minute from all results
        int64_t max_minute = 0;
        for (const auto& result : all_results) {
            int64_t result_minute = result.timestamp_us / (60LL * 1000000LL);
            if (result_minute > max_minute) {
                max_minute = result_minute;
            }
        }
        
        // Clear existing stats and recalculate
        minute_stats.clear();
        
        // Record all minutes from 0 to max_minute
        for (int64_t m = 0; m <= max_minute; ++m) {
            int64_t minute_start_us = m * 60LL * 1000000LL;
            int64_t minute_end_us = (m + 1) * 60LL * 1000000LL;
            
            double sum_recall = 0.0;
            int count = 0;
            
            for (const auto& result : all_results) {
                if (result.timestamp_us >= minute_start_us && result.timestamp_us < minute_end_us) {
                    sum_recall += result.recall;
                    count++;
                }
            }
            
            double avg_recall = (count > 0) ? sum_recall / count : 0.0;
            minute_stats.push_back({m, avg_recall, count});
            
            std::cout << "Minute " << m << ": avg_recall=" << std::fixed << std::setprecision(6) << avg_recall 
                      << ", query_count=" << count << std::endl;
        }
        
        // Copy ngfix_updates while holding the lock
        ngfix_updates_copy = ngfix_updates;
    }
    
    // Write JSON output
    std::ofstream output(output_path);
    output << "{\n";
    output << "  \"test_duration_minutes\": " << (test_duration_seconds / 60) << ",\n";
    output << "  \"minute_stats\": [\n";
    
    for (size_t i = 0; i < minute_stats.size(); ++i) {
        const auto& stat = minute_stats[i];
        output << "    {\n";
        output << "      \"minute\": " << stat.minute << ",\n";
        output << "      \"avg_recall\": " << std::fixed << std::setprecision(6) << stat.avg_recall << ",\n";
        output << "      \"query_count\": " << stat.query_count << "\n";
        output << "    }";
        if (i < minute_stats.size() - 1) {
            output << ",";
        }
        output << "\n";
    }
    
    output << "  ],\n";
    output << "  \"ngfix_updates\": [\n";
    
    for (size_t i = 0; i < ngfix_updates_copy.size(); ++i) {
        const auto& update = ngfix_updates_copy[i];
        output << "    {\n";
        output << "      \"timestamp_seconds\": " << std::fixed << std::setprecision(3) << (update.timestamp_us / 1000000.0) << ",\n";
        output << "      \"duration_ms\": " << std::fixed << std::setprecision(3) << (update.duration_us / 1000.0) << ",\n";
        output << "      \"ops_count_before\": " << update.ops_count_before << ",\n";
        output << "      \"update_type\": \"" << update.update_type << "\"\n";
        output << "    }";
        if (i < ngfix_updates_copy.size() - 1) {
            output << ",";
        }
        output << "\n";
    }
    
    output << "  ]\n";
    output << "}\n";
    output.close();
    
    std::cout << "Statistics written to: " << output_path << std::endl;
}

int main(int argc, char* argv[]) {
    std::string index_path;
    std::string test_query_path;
    std::string test_gt_path;
    std::string additional_data_path;
    std::string metric_str;
    std::string result_path;
    size_t k = 100;
    size_t efs = 100;
    size_t efC = 500;
    int search_qps = 1;
    int ops_qps = 32;
    int test_duration_hours = 2;
    size_t selected_query_count = 5000;  // Use first 5k queries
    
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--index_path") index_path = argv[i + 1];
        if (arg == "--test_query_path") test_query_path = argv[i + 1];
        if (arg == "--test_gt_path") test_gt_path = argv[i + 1];
        if (arg == "--additional_data_path") additional_data_path = argv[i + 1];
        if (arg == "--metric") metric_str = argv[i + 1];
        if (arg == "--result_path") result_path = argv[i + 1];
        if (arg == "--K") k = std::stoi(argv[i + 1]);
        if (arg == "--efs") efs = std::stoi(argv[i + 1]);
        if (arg == "--efC") efC = std::stoi(argv[i + 1]);
        if (arg == "--search_qps") search_qps = std::stoi(argv[i + 1]);
        if (arg == "--ops_qps") ops_qps = std::stoi(argv[i + 1]);
        if (arg == "--test_duration_hours") test_duration_hours = std::stoi(argv[i + 1]);
        if (arg == "--selected_query_count") selected_query_count = std::stoi(argv[i + 1]);
    }
    
    std::cout << "Index path: " << index_path << std::endl;
    std::cout << "Query path: " << test_query_path << std::endl;
    std::cout << "GT path: " << test_gt_path << std::endl;
    std::cout << "Additional data path: " << additional_data_path << std::endl;
    std::cout << "Result path: " << result_path << std::endl;
    std::cout << "Search QPS: " << search_qps << std::endl;
    std::cout << "Ops QPS: " << ops_qps << std::endl;
    std::cout << "Test duration: " << test_duration_hours << " hours" << std::endl;
    std::cout << "Selected query count: " << selected_query_count << std::endl;
    
    // Load data
    size_t test_number = 0, vecdim = 0;
    size_t test_gt_dim = 0;
    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    
    // Ensure selected_query_count doesn't exceed available queries
    if (selected_query_count > test_number) {
        selected_query_count = test_number;
        std::cout << "Warning: selected_query_count adjusted to " << selected_query_count << std::endl;
    }
    
    size_t additional_number = 0;
    float* additional_data = LoadData<float>(additional_data_path, additional_number, vecdim);
    
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
    
    // Resize to accommodate additional data
    size_t max_size = BASE_SIZE + ADDITIONAL_SIZE;
    hnsw_ngfix->resize(max_size);
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    
    // Start threads with synchronized start time
    int64_t test_duration_seconds = test_duration_hours * 3600;
    if (test_duration_seconds <= 0) {
        test_duration_seconds = 120; // Default to 2 minutes for testing
    }
    
    auto global_start_time = std::chrono::high_resolution_clock::now();
    
    std::thread search_t(search_thread, hnsw_ngfix, test_query, test_gt, 
                         test_number, test_gt_dim, vecdim, k, efs, search_qps, BASE_SIZE, global_start_time, selected_query_count);
    std::thread ops_t(random_ops_thread, hnsw_ngfix, additional_data, vecdim, efC, 
                      ops_qps, BASE_SIZE, ADDITIONAL_SIZE, std::ref(rng), global_start_time);
    std::thread stats_t(stats_thread, result_path, test_duration_seconds, global_start_time);
    
    // Wait for test duration
    std::this_thread::sleep_for(std::chrono::seconds(test_duration_seconds));
    
    // Stop test
    test_running.store(false);
    
    // Wait for threads
    search_t.join();
    ops_t.join();
    stats_t.join();
    
    // Final cleanup
    delete[] test_query;
    delete[] test_gt;
    delete[] additional_data;
    delete hnsw_ngfix;
    
    std::cout << "Test completed. Total inserts: " << insert_count.load() 
              << ", Total deletes: " << delete_count.load() 
              << ", Noisy inserts: " << noisy_insert_count.load() << std::endl;
    
    return 0;
}

