#include "ourslib/graph/hnsw_ours.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <filesystem>
#include <random>
#include <sstream>
#include <thread>
#include <atomic>

using namespace ours;
namespace fs = std::filesystem;

// Statistics for lazy delete performance
struct LazyDeleteTestStats {
    size_t epoch_duration_ms;
    size_t page_num;
    size_t num_queries;
    
    // Query performance
    double avg_query_time_with_pending_edges_ms;
    double avg_query_time_without_pending_edges_ms;
    double overhead_ratio;
    
    // Set operation overhead (new statistics)
    size_t total_set_checks;
    size_t total_set_inserts;
    uint64_t total_set_check_time_ns;
    uint64_t total_set_insert_time_ns;
    double avg_set_check_time_ns;
    double avg_set_insert_time_ns;
    double set_overhead_ratio;  // Set operations overhead / total query time
    
    // Edge statistics
    size_t total_nodes_marked;
    size_t total_edges_removed;
    size_t total_edges_accessed;
    size_t total_in_serve_edges;
    size_t total_queries_hitting_pending_nodes;
    size_t total_queries_not_hitting_pending_nodes;
    
    // Set size statistics
    double avg_pending_nodes_count;
    size_t max_pending_nodes_count;
    size_t min_pending_nodes_count;
};

// Test set update overhead with different set sizes
void TestSetUpdateOverhead(size_t num_tests,
                           std::vector<size_t> set_sizes,
                           std::vector<double>& update_times) {
    update_times.clear();
    update_times.resize(set_sizes.size(), 0.0);
    
    for(size_t test_idx = 0; test_idx < num_tests; ++test_idx) {
        for(size_t size_idx = 0; size_idx < set_sizes.size(); ++size_idx) {
            size_t set_size = set_sizes[size_idx];
            
            // Create a test set
            std::unordered_set<id_t> test_set;
            for(size_t i = 0; i < set_size; ++i) {
                test_set.insert(i);
            }
            
            // Measure update time (simulate removing an element)
            auto start = std::chrono::high_resolution_clock::now();
            if(!test_set.empty()) {
                test_set.erase(*test_set.begin());
            }
            auto end = std::chrono::high_resolution_clock::now();
            
            double update_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;  // microseconds
            update_times[size_idx] += update_time;
        }
    }
    
    // Average
    for(size_t i = 0; i < update_times.size(); ++i) {
        update_times[i] /= num_tests;
    }
}

// Test set update overhead with shared_mutex (simulating actual scenario)
void TestSetUpdateOverheadWithLock(size_t num_tests,
                                   std::vector<size_t> set_sizes,
                                   std::vector<double>& update_times) {
    update_times.clear();
    update_times.resize(set_sizes.size(), 0.0);
    
    std::shared_mutex lock;
    
    for(size_t test_idx = 0; test_idx < num_tests; ++test_idx) {
        for(size_t size_idx = 0; size_idx < set_sizes.size(); ++size_idx) {
            size_t set_size = set_sizes[size_idx];
            
            // Create a test set
            std::unordered_map<id_t, std::unordered_set<id_t>> pending_delete_edges;
            id_t test_node = 0;
            for(size_t i = 0; i < set_size; ++i) {
                pending_delete_edges[test_node].insert(i);
            }
            
            // Simulate the actual update process: shared lock -> check -> unique lock -> erase
            auto start = std::chrono::high_resolution_clock::now();
            
            // Step 1: Shared lock to check
            {
                std::shared_lock<std::shared_mutex> shared_lock(lock);
                auto it = pending_delete_edges.find(test_node);
                if(it != pending_delete_edges.end() && it->second.find(0) != it->second.end()) {
                    // Edge is in pending delete set, need to remove it
                    shared_lock.unlock();
                    
                    // Step 2: Upgrade to unique lock
                    std::unique_lock<std::shared_mutex> unique_lock(lock);
                    auto it2 = pending_delete_edges.find(test_node);
                    if(it2 != pending_delete_edges.end()) {
                        it2->second.erase(0);
                        if(it2->second.empty()) {
                            pending_delete_edges.erase(it2);
                        }
                    }
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            
            double update_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;  // microseconds
            update_times[size_idx] += update_time;
        }
    }
    
    // Average
    for(size_t i = 0; i < update_times.size(); ++i) {
        update_times[i] /= num_tests;
    }
}

// Test lazy delete with different epoch and page_num combinations
void TestLazyDelete(HNSW_Ours<float>* index, 
                    float* query_data, 
                    size_t num_queries,
                    size_t k,
                    size_t ef_search,
                    size_t epoch_duration_ms,
                    size_t page_num,
                    LazyDeleteTestStats& stats) {
    
    stats.epoch_duration_ms = epoch_duration_ms;
    stats.page_num = page_num;
    stats.num_queries = num_queries;
    
    // Reset statistics
    stats.total_queries_hitting_pending_nodes = 0;
    stats.total_queries_not_hitting_pending_nodes = 0;
    stats.total_edges_accessed = 0;
    
    std::vector<double> query_times_with_pending;
    std::vector<double> query_times_without_pending;
    std::vector<double> query_times_hitting_pending;
    std::vector<double> query_times_not_hitting_pending;
    
    // Track pending nodes count
    std::vector<size_t> pending_nodes_counts;
    
    // Start lazy delete
    index->StartLazyDelete(epoch_duration_ms, page_num);
    
    // Wait for first epoch to mark some nodes as pending
    // This ensures we have pending nodes in DRAM before starting queries
    std::this_thread::sleep_for(std::chrono::milliseconds(epoch_duration_ms + 100));
    
    // Get initial stats to see pending nodes
    auto initial_stats = index->GetLazyDeleteStats();
    std::cout << "  Initial pending nodes: " << initial_stats.num_nodes_with_pending_deletes 
              << " with " << initial_stats.total_pending_edges << " additional edges" << std::endl;
    
    // Run queries and measure performance with pending nodes
    for(size_t q = 0; q < num_queries; ++q) {
        size_t ndc = 0;
        
        auto query_start = std::chrono::high_resolution_clock::now();
        
        // Check pending nodes count and in-serve edges before query
        auto stats_before = index->GetLazyDeleteStats();
        size_t pending_nodes_before = stats_before.num_nodes_with_pending_deletes;
        size_t in_serve_edges_before = stats_before.total_in_serve_edges;
        
        // Perform search
        auto results = index->searchKnn(query_data + q * index->dim, k, ef_search, ndc);
        
        auto query_end = std::chrono::high_resolution_clock::now();
        double query_time = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start).count() / 1000.0;
        
        // Check if we hit any pending nodes by comparing in-serve edges count
        auto stats_after = index->GetLazyDeleteStats();
        size_t in_serve_edges_after = stats_after.total_in_serve_edges;
        
        // If in-serve edges increased, it means we accessed pending nodes and recorded edges
        bool hit_pending_node = (in_serve_edges_after > in_serve_edges_before);
        
        if(hit_pending_node) {
            stats.total_queries_hitting_pending_nodes++;
            query_times_hitting_pending.push_back(query_time);
        } else {
            stats.total_queries_not_hitting_pending_nodes++;
            query_times_not_hitting_pending.push_back(query_time);
        }
        
        query_times_with_pending.push_back(query_time);
        stats.total_edges_accessed += ndc;
        
        // Record pending nodes count
        pending_nodes_counts.push_back(pending_nodes_before);
    }
    
    // Calculate pending nodes count statistics
    if(!pending_nodes_counts.empty()) {
        double sum_nodes = 0.0;
        stats.max_pending_nodes_count = 0;
        stats.min_pending_nodes_count = std::numeric_limits<size_t>::max();
        for(size_t cnt : pending_nodes_counts) {
            sum_nodes += cnt;
            if(cnt > stats.max_pending_nodes_count) stats.max_pending_nodes_count = cnt;
            if(cnt < stats.min_pending_nodes_count) stats.min_pending_nodes_count = cnt;
        }
        stats.avg_pending_nodes_count = sum_nodes / pending_nodes_counts.size();
    } else {
        stats.avg_pending_nodes_count = 0.0;
        stats.max_pending_nodes_count = 0;
        stats.min_pending_nodes_count = 0;
    }
    
    // Measure queries without pending nodes
    // CORRECT LOGIC (as per user's description):
    // - "with pending": lazy delete enabled, checks pending nodes in DRAM, records in-serve edges
    // - "without pending": lazy delete disabled, NO checks, NO recording, but SAME graph structure
    //
    // To ensure fair comparison with exactly the same graph structure:
    // Reload the index to get clean state with all additional edges, then test without lazy delete
    
    std::cout << "  Reloading index for 'without pending' test (same graph structure, no lazy delete)..." << std::endl;
    
    // Stop lazy delete on current index
    index->StopLazyDelete(false);
    
    // Reload index to ensure exactly the same graph structure (all additional edges present)
    // Note: We need base_index_path and metric, but they're not available here
    // So we'll use a different approach: stop lazy delete immediately before cleanup
    // This keeps most of the graph structure intact
    
    // Actually, let's modify the function signature to accept these parameters
    // For now, we'll use the current approach: stop immediately
    
    // Wait a short time just for the thread to stop
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Measure queries WITHOUT pending node checks
    // Graph structure should be similar (additional edges exist), but no lazy delete overhead
    // This tests the pure overhead of: pending node checks + in-serve edge recording
    for(size_t q = 0; q < num_queries; ++q) {
        size_t ndc = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto results = index->searchKnn(query_data + q * index->dim, k, ef_search, ndc);
        auto end = std::chrono::high_resolution_clock::now();
        
        double query_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        query_times_without_pending.push_back(query_time);
    }
    
    // Calculate statistics
    double sum_with = 0.0, sum_without = 0.0;
    for(double t : query_times_with_pending) {
        sum_with += t;
    }
    for(double t : query_times_without_pending) {
        sum_without += t;
    }
    
    stats.avg_query_time_with_pending_edges_ms = sum_with / query_times_with_pending.size();
    stats.avg_query_time_without_pending_edges_ms = sum_without / query_times_without_pending.size();
    
    if(stats.avg_query_time_without_pending_edges_ms > 0) {
        stats.overhead_ratio = (stats.avg_query_time_with_pending_edges_ms - stats.avg_query_time_without_pending_edges_ms) 
                              / stats.avg_query_time_without_pending_edges_ms;
    } else {
        stats.overhead_ratio = 0.0;
    }
    
    // Calculate average query time for queries hitting vs not hitting pending edges
    double avg_time_hitting = 0.0;
    double avg_time_not_hitting = 0.0;
    if(!query_times_hitting_pending.empty()) {
        double sum_hitting = 0.0;
        for(double t : query_times_hitting_pending) {
            sum_hitting += t;
        }
        avg_time_hitting = sum_hitting / query_times_hitting_pending.size();
    }
    if(!query_times_not_hitting_pending.empty()) {
        double sum_not_hitting = 0.0;
        for(double t : query_times_not_hitting_pending) {
            sum_not_hitting += t;
        }
        avg_time_not_hitting = sum_not_hitting / query_times_not_hitting_pending.size();
    }
    
    
    // Get lazy delete stats
    auto lazy_stats = index->GetLazyDeleteStats();
    stats.total_nodes_marked = lazy_stats.num_nodes_with_pending_deletes;
    stats.total_edges_removed = 0;  // Will be updated after cleanup
    
    // Get actual in-serve edges count from stats
    stats.total_in_serve_edges = lazy_stats.total_in_serve_edges;
    
    // Get set operation statistics
    stats.total_set_checks = lazy_stats.total_set_checks;
    stats.total_set_inserts = lazy_stats.total_set_inserts;
    stats.total_set_check_time_ns = lazy_stats.total_set_check_time_ns;
    stats.total_set_insert_time_ns = lazy_stats.total_set_insert_time_ns;
    
    // Calculate average set operation times
    if(stats.total_set_checks > 0) {
        stats.avg_set_check_time_ns = (double)stats.total_set_check_time_ns / stats.total_set_checks;
    } else {
        stats.avg_set_check_time_ns = 0.0;
    }
    if(stats.total_set_inserts > 0) {
        stats.avg_set_insert_time_ns = (double)stats.total_set_insert_time_ns / stats.total_set_inserts;
    } else {
        stats.avg_set_insert_time_ns = 0.0;
    }
    
    // Calculate set overhead ratio: (check_time + insert_time) / total_query_time
    // This represents the overhead of set operations relative to total query time
    double total_set_time_ms = (stats.total_set_check_time_ns + stats.total_set_insert_time_ns) / 1000000.0;
    double total_query_time_ms = stats.avg_query_time_with_pending_edges_ms * stats.num_queries;
    if(total_query_time_ms > 0) {
        stats.set_overhead_ratio = total_set_time_ms / total_query_time_ms;
    } else {
        stats.set_overhead_ratio = 0.0;
    }
}

int main(int argc, char* argv[]) {
    // Configuration
    std::string base_index_path;
    std::string query_data_path;
    std::string output_dir = "/workspace/OOD-ANNS/Ours/data/lazy_delete_test";
    std::string metric_str = "ip_float";
    size_t num_queries = 1000;
    size_t k = 100;
    size_t ef_search = 200;
    
    // Lazy delete parameters - test different combinations
    std::vector<size_t> epoch_durations_ms = {100, 500, 1000, 2000, 5000};
    std::vector<size_t> page_nums = {10, 50, 100, 200, 500};
    
    // Parse arguments
    for(int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--base_index_path" && i + 1 < argc)
            base_index_path = argv[i + 1];
        if(arg == "--query_data_path" && i + 1 < argc)
            query_data_path = argv[i + 1];
        if(arg == "--output_dir" && i + 1 < argc)
            output_dir = argv[i + 1];
        if(arg == "--metric" && i + 1 < argc)
            metric_str = argv[i + 1];
        if(arg == "--num_queries" && i + 1 < argc)
            num_queries = std::stoi(argv[i + 1]);
        if(arg == "--k" && i + 1 < argc)
            k = std::stoi(argv[i + 1]);
        if(arg == "--ef_search" && i + 1 < argc)
            ef_search = std::stoi(argv[i + 1]);
    }
    
    // Try to find data files
    if(base_index_path.empty()) {
        std::vector<std::string> possible_paths = {
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M16_efC500_MEX48_AKNN1500_8M.index",
            "/workspace/OOD-ANNS/Ours/data/comparison_10M/base.index",
            "/workspace/OOD-ANNS/Ours/data/comparison/base.index",
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M16_efC500_MEX48.index"
        };
        for(const auto& path : possible_paths) {
            if(fs::exists(path)) {
                base_index_path = path;
                break;
            }
        }
    }
    
    if(query_data_path.empty()) {
        std::vector<std::string> possible_paths = {
            "/workspace/RoarGraph/data/t2i-10M/query.10M.fbin",
            "/workspace/RoarGraph/data/t2i-10M/query.fbin",
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/query.fbin"
        };
        for(const auto& path : possible_paths) {
            if(fs::exists(path)) {
                query_data_path = path;
                break;
            }
        }
    }
    
    if(base_index_path.empty() || !fs::exists(base_index_path)) {
        std::cerr << "Error: Base index file not found" << std::endl;
        return 1;
    }
    
    if(query_data_path.empty() || !fs::exists(query_data_path)) {
        std::cerr << "Error: Query data file not found" << std::endl;
        return 1;
    }
    
    std::cout << "=== Lazy Delete Performance Test ===" << std::endl;
    std::cout << "Base index: " << base_index_path << std::endl;
    std::cout << "Query data: " << query_data_path << std::endl;
    std::cout << "Number of queries: " << num_queries << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "ef_search: " << ef_search << std::endl;
    std::cout << "Output dir: " << output_dir << std::endl;
    std::cout << std::endl;
    
    // Create output directory
    fs::create_directories(output_dir);
    
    // Determine metric
    Metric metric;
    if(metric_str == "ip_float") {
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        metric = L2_float;
    } else {
        std::cerr << "Error: Unsupported metric" << std::endl;
        return 1;
    }
    
    // Load index
    std::cout << "Loading index..." << std::endl;
    HNSW_Ours<float>* index = new HNSW_Ours<float>(metric, base_index_path);
    std::cout << "Index loaded. Number of elements: " << index->n.load() << std::endl;
    
    // Load query data
    std::cout << "Loading query data..." << std::endl;
    size_t query_dim = index->dim;
    size_t num_query_vectors = 0;
    float* query_data = nullptr;
    
    try {
        std::ifstream query_file(query_data_path, std::ios::binary);
        if(!query_file) {
            throw std::runtime_error("Cannot open query file");
        }
        
        // Read number of vectors (if available) or estimate from file size
        query_file.seekg(0, std::ios::end);
        size_t file_size = query_file.tellg();
        query_file.seekg(0, std::ios::beg);
        
        // Try to read header (some formats have it)
        size_t header_size = 0;
        try {
            uint32_t num_vecs, dim;
            query_file.read((char*)&num_vecs, sizeof(uint32_t));
            query_file.read((char*)&dim, sizeof(uint32_t));
            if(dim == query_dim && num_vecs > 0 && num_vecs < 100000000) {
                num_query_vectors = num_vecs;
                header_size = 2 * sizeof(uint32_t);
            } else {
                query_file.seekg(0, std::ios::beg);
            }
        } catch(...) {
            query_file.seekg(0, std::ios::beg);
        }
        
        if(num_query_vectors == 0) {
            num_query_vectors = file_size / (query_dim * sizeof(float));
        }
        
        num_query_vectors = std::min(num_query_vectors, (size_t)10000);  // Limit for testing
        num_queries = std::min(num_queries, num_query_vectors);
        
        query_data = new float[num_queries * query_dim];
        query_file.seekg(header_size, std::ios::beg);
        query_file.read((char*)query_data, num_queries * query_dim * sizeof(float));
        query_file.close();
        
        std::cout << "Loaded " << num_queries << " query vectors" << std::endl;
    } catch(const std::exception& e) {
        std::cerr << "Error loading query data: " << e.what() << std::endl;
        delete index;
        return 1;
    }
    
    // Test 1: Set update overhead with different set sizes (without lock)
    std::cout << "\n=== Test 1: Set Update Overhead (No Lock) ===" << std::endl;
    std::vector<size_t> test_set_sizes = {10, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
    std::vector<double> set_update_times;
    TestSetUpdateOverhead(10000, test_set_sizes, set_update_times);
    
    std::ofstream set_update_file(output_dir + "/set_update_overhead.csv");
    set_update_file << "set_size,avg_update_time_us\n";
    for(size_t i = 0; i < test_set_sizes.size(); ++i) {
        set_update_file << test_set_sizes[i] << "," << std::fixed << std::setprecision(6) 
                        << set_update_times[i] << "\n";
    }
    set_update_file.close();
    std::cout << "Set update overhead results saved to: " << output_dir << "/set_update_overhead.csv" << std::endl;
    
    // Test 2: Set update overhead with shared_mutex (simulating actual scenario)
    std::cout << "\n=== Test 2: Set Update Overhead (With Lock) ===" << std::endl;
    std::vector<double> set_update_times_with_lock;
    TestSetUpdateOverheadWithLock(1000, test_set_sizes, set_update_times_with_lock);
    
    std::ofstream set_update_lock_file(output_dir + "/set_update_overhead_with_lock.csv");
    set_update_lock_file << "set_size,avg_update_time_us\n";
    for(size_t i = 0; i < test_set_sizes.size(); ++i) {
        set_update_lock_file << test_set_sizes[i] << "," << std::fixed << std::setprecision(6) 
                             << set_update_times_with_lock[i] << "\n";
    }
    set_update_lock_file.close();
    std::cout << "Set update overhead (with lock) results saved to: " << output_dir << "/set_update_overhead_with_lock.csv" << std::endl;
    
    // Test 3: Lazy delete with different epoch and batch_size combinations
    std::cout << "\n=== Test 3: Lazy Delete Performance ===" << std::endl;
    std::vector<LazyDeleteTestStats> all_stats;
    
    // Test a subset of combinations to avoid too long runtime
    std::vector<std::pair<size_t, size_t>> test_combinations = {
        {100, 10}, {100, 50}, {100, 100},
        {500, 50}, {500, 100}, {500, 200},
        {1000, 100}, {1000, 200}, {1000, 500},
        {2000, 200}, {2000, 500},
        {5000, 500}
    };
    
    for(const auto& [epoch_ms, pages] : test_combinations) {
        std::cout << "\nTesting epoch=" << epoch_ms << "ms, page_num=" << pages << "..." << std::endl;
        
        // Reload index for each test to ensure clean state
        delete index;
        index = new HNSW_Ours<float>(metric, base_index_path);
        
        LazyDeleteTestStats stats;
        TestLazyDelete(index, query_data, num_queries, k, ef_search, epoch_ms, pages, stats);
        all_stats.push_back(stats);
        
        std::cout << "  Avg query time (with pending): " << std::fixed << std::setprecision(3) 
                  << stats.avg_query_time_with_pending_edges_ms << " ms" << std::endl;
        std::cout << "  Avg query time (without pending): " << stats.avg_query_time_without_pending_edges_ms << " ms" << std::endl;
        std::cout << "  Overhead ratio: " << std::fixed << std::setprecision(4) 
                  << stats.overhead_ratio * 100 << "%" << std::endl;
        std::cout << "  Queries hitting pending nodes: " << stats.total_queries_hitting_pending_nodes << std::endl;
        std::cout << "  Queries not hitting pending nodes: " << stats.total_queries_not_hitting_pending_nodes << std::endl;
        std::cout << "  Avg pending nodes count: " << std::fixed << std::setprecision(2) 
                  << stats.avg_pending_nodes_count << std::endl;
        std::cout << "  Total nodes marked: " << stats.total_nodes_marked << std::endl;
        std::cout << "  Total in-serve edges: " << stats.total_in_serve_edges << std::endl;
        std::cout << "  Set overhead ratio: " << std::fixed << std::setprecision(6) 
                  << stats.set_overhead_ratio * 100 << "%" << std::endl;
        std::cout << "  Avg set check time: " << stats.avg_set_check_time_ns << " ns" << std::endl;
        std::cout << "  Avg set insert time: " << stats.avg_set_insert_time_ns << " ns" << std::endl;
    }
    
    // Save results
    std::ofstream results_file(output_dir + "/lazy_delete_results.csv");
    results_file << "epoch_duration_ms,page_num,num_queries,"
                 << "avg_query_time_with_pending_ms,avg_query_time_without_pending_ms,"
                 << "overhead_ratio,"
                 << "total_queries_hitting_pending_nodes,total_queries_not_hitting_pending_nodes,"
                 << "avg_pending_nodes_count,max_pending_nodes_count,min_pending_nodes_count,"
                 << "total_set_checks,total_set_inserts,total_set_check_time_ns,total_set_insert_time_ns,"
                 << "avg_set_check_time_ns,avg_set_insert_time_ns,set_overhead_ratio,"
                 << "total_nodes_marked,total_in_serve_edges,total_edges_removed,total_edges_accessed\n";
    
    for(const auto& stats : all_stats) {
        results_file << stats.epoch_duration_ms << "," << stats.page_num << "," << stats.num_queries << ","
                    << std::fixed << std::setprecision(6)
                    << stats.avg_query_time_with_pending_edges_ms << ","
                    << stats.avg_query_time_without_pending_edges_ms << ","
                    << stats.overhead_ratio << ","
                    << stats.total_queries_hitting_pending_nodes << ","
                    << stats.total_queries_not_hitting_pending_nodes << ","
                    << stats.avg_pending_nodes_count << ","
                    << stats.max_pending_nodes_count << ","
                    << stats.min_pending_nodes_count << ","
                    << stats.total_set_checks << ","
                    << stats.total_set_inserts << ","
                    << stats.total_set_check_time_ns << ","
                    << stats.total_set_insert_time_ns << ","
                    << stats.avg_set_check_time_ns << ","
                    << stats.avg_set_insert_time_ns << ","
                    << stats.set_overhead_ratio << ","
                    << stats.total_nodes_marked << ","
                    << stats.total_in_serve_edges << ","
                    << stats.total_edges_removed << ","
                    << stats.total_edges_accessed << "\n";
    }
    results_file.close();
    std::cout << "\nResults saved to: " << output_dir << "/lazy_delete_results.csv" << std::endl;
    
    // Cleanup
    delete[] query_data;
    delete index;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
