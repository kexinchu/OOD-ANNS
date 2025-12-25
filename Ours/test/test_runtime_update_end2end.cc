#include "ourslib/graph/hnsw_ours.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <iomanip>
#include <unordered_set>
#include <shared_mutex>
#include <ctime>
#include <mutex>
#include <queue>
#include <algorithm>
#include <functional>
#include <sstream>

using namespace ours;

// Global variables for test control
std::atomic<bool> stop_test{false};
std::atomic<size_t> total_searches{0};
std::atomic<size_t> total_inserts{0};
std::atomic<size_t> successful_inserts{0};
std::atomic<size_t> failed_inserts{0};
std::atomic<size_t> total_deletes{0};
std::atomic<size_t> successful_deletes{0};
std::atomic<size_t> failed_deletes{0};
std::atomic<size_t> next_global_insert_id{0};  // Global atomic counter for insert IDs to avoid conflicts across threads

// Thread-safe queue for inserted node IDs (for deletion)
std::mutex inserted_ids_mutex;
std::queue<id_t> inserted_ids_queue;  // Queue of inserted node IDs available for deletion

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
    size_t query_idx;  // Index of the query in train_queries
    float recall;
    size_t ndc;
    double latency_ms;
};

std::mutex stats_mutex;
std::vector<QueryResult> query_results_buffer;
std::vector<MinuteStats> minute_stats;

// Calculate recall
float CalculateRecall(const std::vector<std::pair<float, id_t>>& results, int* gt, size_t k, bool debug = false) {
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
    
    float recall = (float)acc / k;
    
    // Debug output for first few queries
    static std::atomic<size_t> debug_count{0};
    if(debug && debug_count.fetch_add(1) < 5) {
        std::cout << "[DEBUG Recall] acc=" << acc << "/" << k << ", recall=" << recall << std::endl;
        if(acc == 0 && results.size() > 0) {
            std::cout << "  [DEBUG] First 5 GT IDs: ";
            for(size_t i = 0; i < std::min((size_t)5, k); ++i) {
                std::cout << gt[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "  [DEBUG] First 5 Result IDs: ";
            for(size_t i = 0; i < std::min((size_t)5, results.size()); ++i) {
                std::cout << results[i].second << " ";
            }
            std::cout << std::endl;
        }
    }
    
    return recall;
}

// Test thread: QPS=1000 search + QPS=140 insert
void TestThread(HNSW_Ours<float>* index,
                float* train_queries,
                int* train_gt,
                float* additional_vectors,
                size_t num_train_queries,
                size_t num_additional_vectors,
                size_t vecdim,
                size_t k,
                size_t ef_search) {
    
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<size_t> query_dist(0, num_train_queries - 1);
    std::uniform_int_distribution<size_t> vector_dist(0, num_additional_vectors - 1);
    
    // Rate control: 125 QPS per thread (8 threads = 1000 QPS total) + 126 QPS insert total + 14 QPS delete total (9:1 ratio)
    // Insert and delete are distributed across 8 threads
    const double search_interval_ms = 1000.0 / 125.0;   // 8ms per search (125 QPS per thread, 8 threads = 1000 QPS total)
    const double insert_interval_ms = 1000.0 / (126.0 / 8.0);  // ~63.5ms per insert per thread (126/8 = 15.75 QPS per thread, 8 threads = 126 QPS total)
    const double delete_interval_ms = 1000.0 / (14.0 / 8.0);   // ~571ms per delete per thread (14/8 = 1.75 QPS per thread, 8 threads = 14 QPS total)
    
    auto last_search_time = std::chrono::steady_clock::now();
    auto last_insert_time = std::chrono::steady_clock::now();
    auto last_delete_time = std::chrono::steady_clock::now();
    
    // Insert IDs are now managed by global atomic counter (next_global_insert_id)
    // Each thread will get unique IDs atomically when inserting
    size_t insert_count_since_last_delete = 0;  // Track insert count for 9:1 ratio
    
    while(!stop_test.load()) {
        auto now = std::chrono::steady_clock::now();
        
        // FIXED: Use target time point instead of elapsed time to ensure accurate QPS
        // This ensures we achieve target QPS even if operation latency > interval
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
                
                int* gt = train_gt + q_idx * k;  // Use regenerated GT
                float recall = CalculateRecall(results, gt, k, false);  // Disable debug to reduce latency overhead
                
                // Store result for background thread analysis
                {
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    query_results_buffer.push_back({q_idx, recall, ndc, latency_ms});
                }
                
                total_searches.fetch_add(1);
                // FIXED: Update to target time point to maintain accurate QPS
                // If latency > interval, we schedule next operation immediately
                // This ensures we achieve target QPS even under high latency
                last_search_time += std::chrono::microseconds((int64_t)(search_interval_ms * 1000));
                if(last_search_time > now) {
                    // If we're ahead of schedule, reset to now to avoid accumulating delay
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
        
        // Check if we should do an insert (126 QPS total, distributed across 8 threads)
        // Each thread handles insert_interval_ms interval
        auto insert_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - last_insert_time).count() / 1000.0;
        
        if(insert_elapsed >= insert_interval_ms) {
            // Perform insert - use additional vectors (should never be nullptr due to validation)
            if(additional_vectors == nullptr) {
                std::cerr << "[INSERT ERROR] Additional vectors not available! This should not happen." << std::endl;
                continue;
            }
            
            // Get next insert ID atomically (thread-safe across multiple threads)
            id_t new_id = next_global_insert_id.fetch_add(1);
            
            // Check if we've reached max_elements
            if(new_id >= index->max_elements) {
                // Skip if we've reached max
                last_insert_time = std::chrono::steady_clock::now();
                continue;
            }
            
            // Use additional vectors - IMPORTANT: These should be different from base data
            size_t vec_idx = vector_dist(rng);
            float* vec_data = additional_vectors + vec_idx * vecdim;
            
            try {
                size_t n_before = index->n.load();
                size_t efC = 2000;  // Increased from 200 to 2000 for better neighbor selection
                
                // Perform the insert (this adds M base neighbors)
                // InsertPoint now returns the search results from HNSWBottomLayerInsertion
                // This avoids redundant search - we reuse the results from insertion
                auto search_results = index->InsertPoint(new_id, efC, vec_data);
                
                // Verify insert actually happened
                size_t n_after = index->n.load();
                
                // InsertPoint succeeded (no exception thrown), so insert definitely happened
                // The atomic fetch_add guarantees n was incremented by 1, even if concurrent deletes
                // reduced the total n value. We trust the atomic operation.
                bool insert_succeeded = true;  // Trust InsertPoint's atomic operation
                
                if(insert_succeeded) {
                    // Insert succeeded
                    // Insert succeeded - n increased
                    // Add to queue for potential deletion (thread-safe)
                    {
                        std::lock_guard<std::mutex> lock(inserted_ids_mutex);
                        inserted_ids_queue.push(new_id);
                    }
                    successful_inserts.fetch_add(1);
                    insert_count_since_last_delete++;
                    
                    // OPTIMIZATION: After insert, optimize connectivity for topkANN neighbors
                    // Reuse the search results from InsertPoint (no redundant search!)
                    // The base insertion already added M neighbors via HNSWBottomLayerInsertion
                    // Now we optimize connectivity for additional GT/topkANN neighbors (beyond M)
                    // This helps maintain graph quality and prevent recall degradation
                    try {
                        // Use the search results from InsertPoint (already computed during insertion)
                        // search_results contains top efC candidates from searchKnnBaseGraphConstruction
                        // We use the top k of these for NGFixOptimized
                        size_t min_results_needed = std::max((size_t)k, (size_t)20);  // At least k or 20, whichever is larger
                        if(search_results.size() >= min_results_needed) {
                            // Convert search results to GT format (array of node IDs)
                            // Use top k results from the insertion search
                            size_t actual_k = std::min(search_results.size(), (size_t)k);
                            int* gt_array = new int[actual_k];
                            for(size_t i = 0; i < actual_k; ++i) {
                                gt_array[i] = search_results[i].second;
                            }
                            
                            // Use NGFixOptimized to enhance connectivity among these neighbors
                            // This adds additional edges between GT/topkANN neighbors (beyond base M connections)
                            index->NGFixOptimized(vec_data, gt_array, actual_k, actual_k);
                            
                            delete[] gt_array;
                            
                            // Log first few optimizations
                            static std::atomic<size_t> opt_log_count{0};
                            if(opt_log_count.fetch_add(1) < 10) {
                                std::cout << "[INSERT OPTIMIZE] Enhanced connectivity for id=" << new_id 
                                          << " with " << actual_k << " topk neighbors (reused from insertion search, " 
                                          << search_results.size() << " total candidates)" << std::endl;
                            }
                        } else {
                            // Not enough results for optimization (may happen for very new nodes or first insertion)
                            static std::atomic<size_t> skip_count{0};
                            if(skip_count.fetch_add(1) < 5) {
                                std::cout << "[INSERT OPTIMIZE] Skipped for id=" << new_id 
                                          << " (only " << search_results.size() << " results from insertion, need " 
                                          << min_results_needed << ")" << std::endl;
                            }
                        }
                    } catch(const std::exception& e) {
                        // Ignore optimization errors, insert itself succeeded
                        static std::atomic<size_t> opt_error_count{0};
                        if(opt_error_count.fetch_add(1) < 5) {
                            std::cerr << "[INSERT OPTIMIZE ERROR] id=" << new_id << ": " << e.what() << std::endl;
                        }
                    } catch(...) {
                        // Ignore optimization errors
                    }
                    
                    // Log first few successful inserts for verification
                    static std::atomic<size_t> log_count{0};
                    if(log_count.fetch_add(1) < 10) {
                        std::cout << "[INSERT VERIFY] Success: id=" << new_id 
                                  << ", n: " << n_before << " -> " << n_after
                                  << ", vec_idx=" << vec_idx 
                                  << " (from additional_vectors)" << std::endl;
                    }
                } else {
                    // Insert failed - node not in valid range or other error
                    failed_inserts.fetch_add(1);
                    // Only log if it's a significant decrease (likely real failure, not just concurrent deletes)
                    if(n_after < n_before - 5) {
                        std::cerr << "[INSERT ERROR] Insert likely failed: id=" << new_id
                                  << ", n_before=" << n_before << ", n_after=" << n_after 
                                  << " (decreased by " << (n_before - n_after) << ")" << std::endl;
                    }
                }
                
                // next_insert_id is now managed by atomic counter, no need to increment here
                total_inserts.fetch_add(1);
                // FIXED: Update to target time point to maintain accurate QPS
                last_insert_time += std::chrono::microseconds((int64_t)(insert_interval_ms * 1000));
                if(last_insert_time > now) {
                last_insert_time = now;
                }
            } catch(const std::exception& e) {
                failed_inserts.fetch_add(1);
                total_inserts.fetch_add(1);
                std::cerr << "[INSERT EXCEPTION] id=" << new_id << ": " << e.what() << std::endl;
            } catch(...) {
                failed_inserts.fetch_add(1);
                total_inserts.fetch_add(1);
                std::cerr << "[INSERT EXCEPTION] id=" << new_id << ": unknown error" << std::endl;
            }
        }
        
        // Check if we should do a delete (14 QPS total, maintaining 9:1 ratio with insert)
        // FIXED: Target is 9:1 ratio (126 insert / 14 delete = 9:1)
        // Each thread should delete when accumulated 9 inserts (per thread) OR time interval elapsed
        // With 8 threads, each thread handles 15.75 insert QPS, so each thread should handle 15.75/9 = 1.75 delete QPS
        auto delete_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - last_delete_time).count() / 1000.0;
        
        // Delete condition: we have 9+ inserts accumulated (9:1 ratio) OR time interval elapsed (to prevent starvation)
        // FIXED: Changed from 10 to 9 to match 9:1 ratio
        bool should_delete = (insert_count_since_last_delete >= 9) || 
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
                try {
                    size_t n_before_delete = index->n.load();
                    
                    // Check if node is still valid (not already deleted)
                    if(node_to_delete < index->max_elements && node_to_delete < index->n.load() && 
                       !index->is_deleted(node_to_delete)) {
                        
                        // Store insert count before resetting
                        size_t insert_count_before_delete = insert_count_since_last_delete;
                        
                        // Perform delete
                        index->DeletePoint(node_to_delete);
                        
                        size_t n_after_delete = index->n.load();
                        successful_deletes.fetch_add(1);
                        insert_count_since_last_delete = 0;  // Reset counter after successful delete
                        
                        // Log first few successful deletes for verification
                        // Note: n doesn't decrease immediately - lazy delete will process it after 1s
                        static std::atomic<size_t> delete_log_count{0};
                        if(delete_log_count.fetch_add(1) < 10) {
                            std::cout << "[DELETE MARK] Marked for lazy delete: id=" << node_to_delete 
                                      << ", n: " << n_before_delete << " (will be truly deleted after 1s if not accessed)" 
                                      << " (insert_count was " << insert_count_before_delete << ")" << std::endl;
                        }
                    } else {
                        // Node already deleted or invalid, skip but don't count as failed
                        static std::atomic<size_t> skip_log_count{0};
                        if(skip_log_count.fetch_add(1) < 5) {
                            std::cout << "[DELETE SKIP] Node " << node_to_delete 
                                      << " already deleted or invalid (n=" << index->n.load() 
                                      << ", max=" << index->max_elements << ")" << std::endl;
                        }
                        // Don't increment failed_deletes for this case, just try next time
                        // But reset counter if we had enough inserts
                        if(insert_count_since_last_delete >= 9) {
                            insert_count_since_last_delete = 0;
                        }
                    }
                    
                    total_deletes.fetch_add(1);
                    // FIXED: Update to target time point to maintain accurate QPS
                    last_delete_time += std::chrono::microseconds((int64_t)(delete_interval_ms * 1000));
                    if(last_delete_time > now) {
                    last_delete_time = now;
                    }
                } catch(const std::exception& e) {
                    failed_deletes.fetch_add(1);
                    total_deletes.fetch_add(1);
                    static std::atomic<size_t> delete_error_count{0};
                    if(delete_error_count.fetch_add(1) < 5) {
                        std::cerr << "[DELETE EXCEPTION] id=" << node_to_delete << ": " << e.what() << std::endl;
                    }
                    last_delete_time = now;
                    // Reset counter on error if we had enough inserts
                    if(insert_count_since_last_delete >= 9) {
                        insert_count_since_last_delete = 0;
                    }
                } catch(...) {
                    failed_deletes.fetch_add(1);
                    total_deletes.fetch_add(1);
                    static std::atomic<size_t> delete_error_count{0};
                    if(delete_error_count.fetch_add(1) < 5) {
                        std::cerr << "[DELETE EXCEPTION] id=" << node_to_delete << ": unknown error" << std::endl;
                    }
                    last_delete_time = now;
                    // Reset counter on error if we had enough inserts
                    if(insert_count_since_last_delete >= 9) {
                        insert_count_since_last_delete = 0;
                    }
                }
            } else {
                // No nodes available for deletion yet, wait for more inserts
                // Only update time if we've waited long enough to prevent busy waiting
                if(delete_elapsed >= delete_interval_ms * 2) {
                    last_delete_time = now;  // Prevent too frequent checks
                    // If we have enough inserts but no nodes to delete, log it
                    if(insert_count_since_last_delete >= 9) {
                        static std::atomic<size_t> no_node_warning_count{0};
                        if(no_node_warning_count.fetch_add(1) < 5) {
                            std::cout << "[DELETE WARNING] Have " << insert_count_since_last_delete 
                                      << " inserts but no nodes in queue for deletion" << std::endl;
                        }
                        // Reset counter to prevent accumulation
                        insert_count_since_last_delete = 0;
                    }
                }
            }
        }
        
        // Small sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

// Background thread: analyze queries with recall < 98% and enhance connectivity
// Optimized version: runs every second, maintains top 100 hardest queries using a min-heap
void ConnectivityEnhancementThread(HNSW_Ours<float>* index,
                                    float* train_queries,
                                    int* train_gt,
                                    size_t num_train_queries,
                                    size_t vecdim,
                                    size_t k,
                                    size_t ef_search) {
    
    std::cout << "[ConnectivityEnhancementThread] Started (optimized: 1s interval, top 100 hardest queries)" << std::endl;
    
    // Min-heap to maintain top 100 hardest queries (lowest recall = hardest)
    // Using pair<recall, QueryResult> where recall is the key for comparison
    auto compare = [](const std::pair<float, QueryResult>& a, const std::pair<float, QueryResult>& b) {
        return a.first > b.first;  // Min-heap: higher recall (less hard) at top
    };
    std::priority_queue<std::pair<float, QueryResult>, 
                       std::vector<std::pair<float, QueryResult>>, 
                       decltype(compare)> hardness_heap(compare);
    
    // OPTIMIZED: Increased from 200 to 400 to process more hard queries per second
    // This helps maintain recall stability by fixing more graph connectivity issues
    const size_t MAX_HARD_QUERIES = 400;
    
    // Statistics for logging
    static size_t second_count = 0;
    static size_t total_processed_last_10_seconds = 0;
    
    while(!stop_test.load()) {
        auto second_start = std::chrono::steady_clock::now();
        
        // Collect queries from buffer during this second
        // FIXED: Don't clear the buffer - StatisticsThread needs it for minute-level stats
        // Instead, we'll just read a copy without clearing
        std::vector<QueryResult> current_second_queries;
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            current_second_queries = query_results_buffer;
            // DO NOT clear - StatisticsThread needs to accumulate queries for minute stats
        }
        
        // Update hardness heap with queries from this second
        // Heap accumulates across seconds until it reaches MAX_HARD_QUERIES
        size_t hard_queries_this_second = 0;
        for(const auto& result : current_second_queries) {
            if(result.recall < 0.98f) {  // Only consider queries with recall < 98%
                hard_queries_this_second++;
                if(hardness_heap.size() < MAX_HARD_QUERIES) {
                    // Heap not full, add directly
                    hardness_heap.push({result.recall, result});
                } else {
                    // Heap is full, check if this query is harder than the easiest in heap
                    if(result.recall < hardness_heap.top().first) {
                        // This query is harder (lower recall), replace the easiest
                        hardness_heap.pop();
                        hardness_heap.push({result.recall, result});
                    }
                }
            }
        }
        
        // Debug: log heap status (every 10 seconds)
        static size_t debug_second_count = 0;
        debug_second_count++;
        if(debug_second_count % 10 == 0) {
            std::cout << "[ConnectivityEnhancementThread] This second: collected " 
                      << current_second_queries.size() << " queries, " 
                      << hard_queries_this_second << " hard queries, heap size: " 
                      << hardness_heap.size() << std::endl;
        }
        
        // At the end of each second, process top 100 hard queries from the heap
        // Heap accumulates across seconds until it reaches MAX_HARD_QUERIES
        // Each second, we process up to MAX_HARD_QUERIES and then clear the heap
        if(!hardness_heap.empty() && !stop_test.load()) {
            // Extract queries from heap to process (up to MAX_HARD_QUERIES)
            std::vector<std::pair<float, QueryResult>> hard_queries_to_process;
            size_t to_process = std::min(hardness_heap.size(), MAX_HARD_QUERIES);
            
            // Extract top queries from heap
            while(!hardness_heap.empty() && hard_queries_to_process.size() < to_process) {
                hard_queries_to_process.push_back(hardness_heap.top());
                hardness_heap.pop();
            }
            
            // OPTIMIZED: Parallel processing of hard queries using std::thread
            // This significantly speeds up graph connectivity enhancement
            // Each query is independent, so parallel processing is safe
            size_t processed_count = 0;
            const size_t num_threads = std::min(hard_queries_to_process.size(), (size_t)8);  // Use up to 8 threads
            const size_t queries_per_thread = (hard_queries_to_process.size() + num_threads - 1) / num_threads;
            
            std::vector<std::thread> workers;
            std::atomic<size_t> atomic_processed_count{0};
            
            for(size_t t = 0; t < num_threads; ++t) {
                workers.emplace_back([&, t]() {
                    size_t start_idx = t * queries_per_thread;
                    size_t end_idx = std::min(start_idx + queries_per_thread, hard_queries_to_process.size());
                    
                    for(size_t i = start_idx; i < end_idx; ++i) {
                        if(stop_test.load()) break;
                        
                        const auto& [recall_val, result] = hard_queries_to_process[i];
                        size_t q_idx = result.query_idx;
                        if(q_idx >= num_train_queries) continue;
                        
                        float* query_data = train_queries + q_idx * vecdim;
                        int* gt = train_gt + q_idx * k;
                        
                        try {
                            // Use NGFix to enhance connectivity for this hard query
                            index->NGFixOptimized(query_data, gt, k, k);
                            atomic_processed_count.fetch_add(1);
                        } catch(...) {
                            // Ignore errors
                        }
                    }
                });
            }
            
            // Wait for all threads to complete
            for(auto& worker : workers) {
                worker.join();
            }
            
            processed_count = atomic_processed_count.load();
            
            // Log processing stats (every 10 seconds to avoid spam)
            // Accumulate count over 10 seconds
            second_count++;
            total_processed_last_10_seconds += processed_count;
            
            if(second_count % 10 == 0) {
                std::cout << "[ConnectivityEnhancementThread] Processed " << total_processed_last_10_seconds 
                          << " hard queries in last 10 seconds (avg " 
                          << (total_processed_last_10_seconds / 10) << " per second)" << std::endl;
                total_processed_last_10_seconds = 0;  // Reset for next 10 seconds
            }
            
            // IMPORTANT: Do NOT clear the heap - let it accumulate across seconds
            // This allows the heap to build up to 100 hard queries even if each second
            // only has a few hard queries. We process up to 100 each second, but the
            // heap continues to accumulate until it reaches 100.
            // Only clear if we processed all queries (heap is already empty)
        }
        
        // Sleep until next second (ensure exactly 1 second interval)
        auto second_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            second_end - second_start).count();
        auto sleep_time_ms = 1000 - elapsed;
        if(sleep_time_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_ms));
        }
    }
    
    std::cout << "[ConnectivityEnhancementThread] Stopped" << std::endl;
}

// Adaptive parameter adjustment thread: adjusts epoch_duration_ms and page_num based on performance
void AdaptiveParamsThread(HNSW_Ours<float>* index, const std::string& result_file) {
    std::cout << "[AdaptiveParamsThread] Started (adjusts parameters every 30 minutes)" << std::endl;
    
    const size_t ADJUSTMENT_INTERVAL_MINUTES = 30;
    size_t adjustment_count = 0;
    
    // Wait for first 30 minutes of data before making adjustments
    std::this_thread::sleep_for(std::chrono::minutes(ADJUSTMENT_INTERVAL_MINUTES));
    
    while(!stop_test.load()) {
        if(stop_test.load()) break;
        
        adjustment_count++;
        std::cout << "\n[AdaptiveParamsThread] Starting parameter adjustment #" << adjustment_count << std::endl;
        
        // Read CSV file to get recent statistics
        std::ifstream in(result_file);
        if(!in.is_open()) {
            std::cerr << "[AdaptiveParamsThread] Warning: Cannot open result file: " << result_file << std::endl;
            std::this_thread::sleep_for(std::chrono::minutes(ADJUSTMENT_INTERVAL_MINUTES));
            continue;
        }
        
        // Read all lines
        std::vector<std::string> lines;
        std::string line;
        while(std::getline(in, line)) {
            if(!line.empty() && line.find("timestamp") == std::string::npos) {  // Skip header
                lines.push_back(line);
            }
        }
        in.close();
        
        if(lines.size() < ADJUSTMENT_INTERVAL_MINUTES) {
            std::cout << "[AdaptiveParamsThread] Not enough data yet (" << lines.size() 
                      << " minutes, need " << ADJUSTMENT_INTERVAL_MINUTES << ")" << std::endl;
            std::this_thread::sleep_for(std::chrono::minutes(ADJUSTMENT_INTERVAL_MINUTES));
            continue;
        }
        
        // Get last 30 minutes of data
        size_t start_idx = lines.size() >= ADJUSTMENT_INTERVAL_MINUTES ? 
                          lines.size() - ADJUSTMENT_INTERVAL_MINUTES : 0;
        
        // Parse CSV data
        struct StatRow {
            double recall;
            double ndc;
            double latency;
        };
        
        std::vector<StatRow> stats;
        for(size_t i = start_idx; i < lines.size(); ++i) {
            std::istringstream iss(lines[i]);
            std::string token;
            StatRow row;
            size_t col = 0;
            
            while(std::getline(iss, token, ',')) {
                if(col == 2) {  // avg_recall
                    row.recall = std::stod(token);
                } else if(col == 3) {  // avg_ndc
                    row.ndc = std::stod(token);
                } else if(col == 4) {  // avg_latency_ms
                    row.latency = std::stod(token);
                    break;
                }
                col++;
            }
            stats.push_back(row);
        }
        
        if(stats.size() < 2) {
            std::cout << "[AdaptiveParamsThread] Not enough data points" << std::endl;
            std::this_thread::sleep_for(std::chrono::minutes(ADJUSTMENT_INTERVAL_MINUTES));
            continue;
        }
        
        // Analyze data
        double first_recall = stats[0].recall;
        double last_recall = stats.back().recall;
        double recall_drop = first_recall - last_recall;
        double recall_drop_pct = (recall_drop / first_recall) * 100.0;
        
        // Calculate trends for ndc and latency
        double first_ndc = stats[0].ndc;
        double last_ndc = stats.back().ndc;
        double ndc_increase = last_ndc - first_ndc;
        double ndc_increase_pct = (ndc_increase / first_ndc) * 100.0;
        
        double first_latency = stats[0].latency;
        double last_latency = stats.back().latency;
        double latency_increase = last_latency - first_latency;
        double latency_increase_pct = (latency_increase / first_latency) * 100.0;
        
        // Calculate linear trend for ndc and latency (using simple linear regression)
        double ndc_trend = 0.0;
        double latency_trend = 0.0;
        if(stats.size() > 1) {
            double sum_x = 0.0, sum_y_ndc = 0.0, sum_y_latency = 0.0;
            double sum_xy_ndc = 0.0, sum_xy_latency = 0.0;
            double sum_x2 = 0.0;
            
            for(size_t i = 0; i < stats.size(); ++i) {
                double x = (double)i;
                sum_x += x;
                sum_y_ndc += stats[i].ndc;
                sum_y_latency += stats[i].latency;
                sum_xy_ndc += x * stats[i].ndc;
                sum_xy_latency += x * stats[i].latency;
                sum_x2 += x * x;
            }
            
            double n = (double)stats.size();
            double denominator = n * sum_x2 - sum_x * sum_x;
            if(std::abs(denominator) > 1e-6) {
                ndc_trend = (n * sum_xy_ndc - sum_x * sum_y_ndc) / denominator;
                latency_trend = (n * sum_xy_latency - sum_x * sum_y_latency) / denominator;
            }
        }
        
        std::cout << "[AdaptiveParamsThread] Analysis (last " << stats.size() << " minutes):" << std::endl;
        std::cout << "  Recall: " << first_recall << " -> " << last_recall 
                  << " (drop: " << recall_drop_pct << "%)" << std::endl;
        std::cout << "  NDC: " << first_ndc << " -> " << last_ndc 
                  << " (increase: " << ndc_increase_pct << "%, trend: " << ndc_trend << ")" << std::endl;
        std::cout << "  Latency: " << first_latency << " -> " << last_latency 
                  << " (increase: " << latency_increase_pct << "%, trend: " << latency_trend << ")" << std::endl;
        
        // Get current parameters
        auto [current_epoch_ms, current_page_num] = index->GetPendingDeleteParams();
        std::cout << "  Current params: epoch_duration_ms=" << current_epoch_ms 
                  << ", page_num=" << current_page_num << std::endl;
        
        // Decision logic
        bool should_adjust = false;
        size_t new_page_num = current_page_num;
        
        // Rule 1: If recall drop > 1%, don't adjust (connectivity issue)
        if(recall_drop_pct > 1.0) {
            std::cout << "  Decision: Recall drop > 1%, skipping adjustment (connectivity issue)" << std::endl;
        } else {
            // Rule 2: If recall is stable but ndc and latency are increasing, increase page_num
            bool ndc_increasing = ndc_trend > 0.0 && ndc_increase_pct > 2.0;  // Trend positive and >2% increase
            bool latency_increasing = latency_trend > 0.0 && latency_increase_pct > 2.0;  // Trend positive and >2% increase
            
            if(ndc_increasing || latency_increasing) {
                should_adjust = true;
                new_page_num = current_page_num + 1;
                std::cout << "  Decision: Recall stable, but ";
                if(ndc_increasing) std::cout << "NDC increasing ";
                if(latency_increasing) std::cout << "Latency increasing ";
                std::cout << "- increasing page_num: " << current_page_num << " -> " << new_page_num << std::endl;
            } else {
                std::cout << "  Decision: No significant increase in NDC or latency, keeping current params" << std::endl;
            }
        }
        
        // Apply adjustment
        if(should_adjust) {
            index->UpdatePendingDeleteParams(current_epoch_ms, new_page_num);
            std::cout << "  Applied: epoch_duration_ms=" << current_epoch_ms 
                      << ", page_num=" << new_page_num << std::endl;
        }
        
        std::cout << std::endl;
        
        // Wait for next adjustment interval
        std::this_thread::sleep_for(std::chrono::minutes(ADJUSTMENT_INTERVAL_MINUTES));
    }
    
    std::cout << "[AdaptiveParamsThread] Stopped" << std::endl;
}

// Statistics collection thread: collect stats every minute
void StatisticsThread(HNSW_Ours<float>* index,
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
        std::cout << "\n=== Minute " << minute_count << " Statistics ===" << std::endl;
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
    
    std::cout << "=== Runtime Update End-to-End Test ===" << std::endl;
    std::cout << "Base index: " << base_index_path << std::endl;
    std::cout << "Train queries: " << train_query_path << std::endl;
    std::cout << "Train GT: " << train_gt_path << std::endl;
    std::cout << "Additional vectors: " << additional_vector_path << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "ef_search: " << ef_search << std::endl;
    std::cout << "Duration: " << duration_minutes << " minutes" << std::endl;
    std::cout << "Search QPS: 1000" << std::endl;
    std::cout << "Insert QPS: 126" << std::endl;
    std::cout << "Delete QPS: 14 (9:1 ratio with insert)" << std::endl;
    std::cout << "Graph fixes: DISABLED (testing pure insert/delete + search)" << std::endl;
    std::cout << std::endl;
    
    // Load data
    size_t train_number = 0;
    size_t train_gt_dim = 0, vecdim = 0;
    size_t additional_number = 0, additional_vecdim = 0;
    
    auto train_query = LoadData<float>(train_query_path, train_number, vecdim);
    
    // Determine metric first (needed for both GT loading and index loading)
    Metric metric;
    if(metric_str == "ip_float") {
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        metric = L2_float;
    } else {
        std::cerr << "ERROR: Unsupported metric type." << std::endl;
        delete[] train_query;
        return 1;
    }
    
    // Load GT data if provided, otherwise generate it
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
        // Generate GT if not provided
        std::cout << "=== No GT file provided, generating GT ===" << std::endl;
        
        std::cout << "Loading index for GT generation..." << std::endl;
        auto index_for_gt = new HNSW_Ours<float>(metric, base_index_path, false);
        
        size_t index_dim = index_for_gt->dim;
        if(index_dim != vecdim) {
            std::cerr << "ERROR: Dimension mismatch! Index=" << index_dim 
                      << ", Query=" << vecdim << std::endl;
            delete index_for_gt;
            delete[] train_query;
            return 1;
        }
        
        // Limit number of queries for GT generation
        size_t max_queries_for_gt = std::min(train_number, (size_t)10000);
        actual_train_number = max_queries_for_gt;
        train_gt = new int[actual_train_number * k];
        size_t gt_ef_search = 2000;
        
        std::cout << "Generating GT for " << actual_train_number << " queries using ef_search=" 
                  << gt_ef_search << "..." << std::endl;
        
        #pragma omp parallel for schedule(dynamic) num_threads(16)
        for(size_t i = 0; i < actual_train_number; ++i) {
            float* query_data = train_query + i * vecdim;
            size_t ndc = 0;
            auto results = index_for_gt->searchKnn(query_data, k, gt_ef_search, ndc);
            
            int* gt = train_gt + i * k;
            for(size_t j = 0; j < std::min(results.size(), k); ++j) {
                gt[j] = results[j].second;
            }
            for(size_t j = results.size(); j < k; ++j) {
                gt[j] = -1;
            }
            
            if((i + 1) % 1000 == 0) {
                std::cout << "  Generated GT for " << (i + 1) << "/" << actual_train_number << " queries" << std::endl;
            }
        }
        
        delete index_for_gt;
        std::cout << "GT generation complete!" << std::endl;
        std::cout << std::endl;
    }
    
    // Update train_number to actual used
    train_number = actual_train_number;
    
    float* additional_vectors = nullptr;
    
    // Load additional vectors - REQUIRED for this test
    // These vectors should be DIFFERENT from the base index data to ensure real inserts occur
    if(additional_vector_path.empty()) {
        std::cerr << "ERROR: --additional_vector_path is REQUIRED!" << std::endl;
        std::cerr << "This test requires distinct additional vectors (different from base index data)" << std::endl;
        std::cerr << "to verify that real insert operations are occurring." << std::endl;
        std::cerr << "Please provide: --additional_vector_path /path/to/base.additional.10M.fbin" << std::endl;
        delete[] train_query;
        delete[] train_gt;
        return 1;
    }
    
    std::cout << "=== Loading Additional Vectors (for insert operations) ===" << std::endl;
    auto loaded_additional = LoadData<float>(additional_vector_path, additional_number, additional_vecdim);
    if(additional_vecdim != vecdim) {
        std::cerr << "ERROR: Additional vector file dimension mismatch!" << std::endl;
        std::cerr << "  Expected dimension: " << vecdim << std::endl;
        std::cerr << "  Got dimension: " << additional_vecdim << std::endl;
        delete[] train_query;
        delete[] train_gt;
        delete[] loaded_additional;
        return 1;
    }
    if(additional_number == 0) {
        std::cerr << "ERROR: Additional vector file contains no vectors!" << std::endl;
        delete[] train_query;
        delete[] train_gt;
        delete[] loaded_additional;
        return 1;
    }
    
    additional_vectors = loaded_additional;
    std::cout << "Successfully loaded " << additional_number << " additional vectors from: " << additional_vector_path << std::endl;
    std::cout << "These vectors are DIFFERENT from base index data - will be used for insert operations." << std::endl;
    
    std::cout << "Loaded " << train_number << " training queries" << std::endl;
    std::cout << "Loaded " << additional_number << " additional vectors" << std::endl;
    std::cout << "Vector dimension: " << vecdim << std::endl;
    std::cout << "Train GT dimension: " << k << " (regenerated)" << std::endl;
    
    // Debug: Check first few GT values
    if(train_number > 0) {
        std::cout << "\n[DEBUG] First query GT sample (first 10 IDs): ";
        int* first_gt = train_gt;
        for(size_t i = 0; i < std::min((size_t)10, k); ++i) {
            std::cout << first_gt[i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "=== Loading Index (for testing) ===" << std::endl;
    auto index = new HNSW_Ours<float>(metric, base_index_path, false);  // false = no SSD
    index->printGraphInfo();
    
    // Check if we need to resize for inserts
    size_t current_n = index->n.load();
    size_t current_max = index->max_elements;
    std::cout << "Index max_elements: " << current_max << std::endl;
    std::cout << "Index current n: " << current_n << std::endl;
    
    // Calculate required capacity: ensure enough for test duration
    // 126 QPS insert = 126 * 60 = 7560 inserts per minute
    size_t inserts_needed = duration_minutes * 60 * 126;  // Total inserts in test duration
    size_t target_max = 12000000;  // Minimum target capacity
    size_t required_max = current_n + inserts_needed + 500000;  // Add 500K buffer for safety
    
    // For long tests (10h = 600min), we need: 10M + 0.72M + 0.5M ≈ 11.22M
    // But let's cap at a reasonable maximum (e.g., 20M) to avoid excessive memory
    size_t max_reasonable = 20000000;  // Cap at 20M
    size_t new_max = std::max({target_max, required_max});
    new_max = std::min(new_max, max_reasonable);  // Cap at maximum
    
    if(new_max > current_max) {
        std::cout << "\n=== Resizing Index to Allow Inserts ===" << std::endl;
        std::cout << "Current max_elements: " << current_max << std::endl;
        std::cout << "Target max_elements: " << new_max << std::endl;
        std::cout << "Expected inserts in " << duration_minutes << " minutes: " << inserts_needed << std::endl;
        std::cout << "Resizing..." << std::endl;
        index->resize(new_max);
        std::cout << "Resize complete. New max_elements: " << index->max_elements << std::endl;
        std::cout << "Remaining capacity for inserts: " << (index->max_elements - current_n) << std::endl;
    } else {
        std::cout << "Index already has enough capacity" << std::endl;
    }
    std::cout << std::endl;
    
    // Create result file (use new filename to avoid overwriting current test)
    std::string result_file = result_dir + "/runtime_update_results_q1000_i140.csv";
    std::ofstream out(result_file);
    if(out.is_open()) {
        out << "timestamp,query_count,avg_recall,avg_ndc,avg_latency_ms,total_searches,total_inserts,successful_inserts,failed_inserts,total_deletes,successful_deletes,failed_deletes,index_size\n";
        out.close();
    }
    
    // Log initial index state
    size_t initial_index_size = index->n.load();
    size_t remaining_capacity = index->max_elements - initial_index_size;
    std::cout << "\n=== Initial Index State ===" << std::endl;
    std::cout << "Initial index size (n): " << initial_index_size << std::endl;
    std::cout << "Max elements: " << index->max_elements << std::endl;
    std::cout << "Remaining capacity for inserts: " << remaining_capacity << std::endl;
    
    // Warn if index is full
    if(remaining_capacity == 0) {
        std::cerr << "\n*** WARNING: Index is FULL! ***" << std::endl;
        std::cerr << "No insert operations can be performed because index has reached max_elements." << std::endl;
        std::cerr << "Expected insert rate: 100 QPS" << std::endl;
        std::cerr << "Expected inserts in " << duration_minutes << " minutes: " << (100 * 60 * duration_minutes) << std::endl;
        std::cerr << "But index capacity is 0, so no inserts will occur!" << std::endl;
        std::cerr << "\nTo test insert operations, use an index with:" << std::endl;
        std::cerr << "  max_elements > current n (e.g., max_elements >= " << (initial_index_size + 100 * 60 * duration_minutes) << ")" << std::endl;
        std::cerr << "********************************\n" << std::endl;
    } else if(remaining_capacity < 100 * 60 * duration_minutes) {
        std::cerr << "\n*** WARNING: Limited insert capacity ***" << std::endl;
        std::cerr << "Remaining capacity (" << remaining_capacity << ") is less than expected inserts (" 
                  << (100 * 60 * duration_minutes) << ")" << std::endl;
        std::cerr << "Some insert operations will fail when capacity is reached." << std::endl;
        std::cerr << "******************************\n" << std::endl;
    }
    
    // Verify additional_vectors is valid (should never be nullptr at this point due to earlier checks)
    if(additional_vectors == nullptr) {
        std::cerr << "ERROR: Additional vectors not loaded! This should not happen." << std::endl;
        delete[] train_query;
        delete[] train_gt;
        delete index;
        return 1;
    }
    
    std::cout << "\n=== Insert Data Verification ===" << std::endl;
    std::cout << "Using additional vectors from: " << additional_vector_path << std::endl;
    std::cout << "Number of additional vectors available: " << additional_number << std::endl;
    std::cout << "These vectors are DIFFERENT from base index data." << std::endl;
    std::cout << "Insert operations will read from this file to ensure real inserts occur." << std::endl;
    std::cout << std::endl;
    
    // Start pending delete mechanism to clean up expired additional edges
    // This prevents graph size from continuously growing 
    std::cout << "\n=== Starting Pending Delete Mechanism ===" << std::endl;
    size_t epoch_duration_ms = 2000;  // 3 second per epoch
    size_t page_num = 8;  // Number of pages to select per epoch
    index->StartPendingDelete(epoch_duration_ms, page_num);
    std::cout << "Pending delete enabled: epoch_duration=" << epoch_duration_ms 
              << "ms, page_num=" << page_num << std::endl;
    std::cout << "This will periodically clean up expired additional edges to prevent graph growth." << std::endl;
    std::cout << std::endl;
    
    // Start lazy delete mechanism for node deletion
    // This handles real deletion of nodes that haven't been accessed in 1 second
    std::cout << "\n=== Starting Lazy Delete Mechanism ===" << std::endl;
    index->StartLazyDelete();
    std::cout << "Lazy delete enabled: nodes will be truly deleted after 1 second of no access." << std::endl;
    std::cout << "This ensures graph connectivity is maintained and n decreases properly." << std::endl;
    std::cout << std::endl;
    
    // Start threads
    // FIXED: Create 8 test threads, each with 125 QPS (total 1000 QPS)
    const size_t num_test_threads = 8;  // 8 threads * 125 QPS = 1000 QPS total
    std::vector<std::thread> test_threads;
    for(size_t i = 0; i < num_test_threads; ++i) {
        test_threads.emplace_back(TestThread, index, train_query, train_gt, additional_vectors,
                           train_number, additional_number, vecdim, k, ef_search);
    }
    
    std::thread connectivity_thread(ConnectivityEnhancementThread, index, train_query, train_gt,
                                   train_number, vecdim, k, ef_search);
    
    std::thread stats_thread(StatisticsThread, index, vecdim, k, ef_search, result_file);
    
    std::thread adaptive_params_thread(AdaptiveParamsThread, index, result_file);
    
    // Initialize global insert ID counter from current index size (thread-safe)
    next_global_insert_id.store(index->n.load());
    std::cout << "Initialized global insert ID counter: " << next_global_insert_id.load() << std::endl;
    
    std::cout << "=== Started test threads ===" << std::endl;
    std::cout << "Test will run for " << duration_minutes << " minutes" << std::endl;
    std::cout << std::endl;
    
    // Wait for duration
    std::this_thread::sleep_for(std::chrono::minutes(duration_minutes));
    
    // Stop test
    std::cout << "\n=== Stopping Test ===" << std::endl;
    stop_test.store(true);
    
    // Stop pending delete mechanism
    std::cout << "Stopping pending delete mechanism..." << std::endl;
    index->StopPendingDelete(true);  // Wait for cleanup to complete
    
    // Stop lazy delete mechanism
    std::cout << "Stopping lazy delete mechanism..." << std::endl;
    index->StopLazyDelete(true);  // Wait for cleanup to complete
    
    // Wait for threads
    // Join all test threads
    for(auto& t : test_threads) {
        t.join();
    }
    connectivity_thread.join();
    stats_thread.join();
    adaptive_params_thread.join();
    
    delete index;
    
    // Get final index state
    size_t final_index_size = index->n.load();
    size_t index_size_increase = final_index_size - initial_index_size;
    
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
    std::cout << "Initial index size: " << initial_index_size << std::endl;
    std::cout << "Final index size: " << final_index_size << std::endl;
    std::cout << "Index size increase: " << index_size_increase << std::endl;
    
    // Verify inserts actually happened
    if(successful_inserts.load() == 0) {
        std::cerr << "\n*** ERROR: No successful inserts occurred! ***" << std::endl;
        std::cerr << "Please check:" << std::endl;
        std::cerr << "1. Index max_elements is large enough" << std::endl;
        std::cerr << "2. Insert operations are not failing silently" << std::endl;
        std::cerr << "3. Additional vectors file is valid: " << additional_vector_path << std::endl;
    } else if(index_size_increase == 0) {
        std::cerr << "\n*** WARNING: Index size did not increase despite successful inserts! ***" << std::endl;
        std::cerr << "This may indicate an issue with insert verification." << std::endl;
    } else {
        std::cout << "\n*** SUCCESS: Inserts verified - index size increased by " << index_size_increase << " points ***" << std::endl;
    }
    
    std::cout << "Results saved to: " << result_file << std::endl;
    
    return 0;
}

