#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
using namespace ngfixlib;

std::string getCurrentTimestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// Detailed overhead statistics
struct OverheadStats {
    // Time breakdown (in microseconds)
    uint64_t total_time_us = 0;
    uint64_t distance_computation_time_us = 0;
    uint64_t is_deleted_check_time_us = 0;
    uint64_t lock_acquire_time_us = 0;
    uint64_t queue_operation_time_us = 0;
    uint64_t neighbor_access_time_us = 0;
    uint64_t visited_check_time_us = 0;
    uint64_t other_time_us = 0;
    
    // Counts
    size_t num_distance_computations = 0;
    size_t num_is_deleted_checks = 0;
    size_t num_lock_acquires = 0;
    size_t num_queue_operations = 0;
    size_t num_neighbor_accesses = 0;
    size_t num_visited_checks = 0;
    size_t num_deleted_nodes_visited = 0;
    size_t num_valid_nodes_visited = 0;
    
    // Query results
    double recall = 0;
    size_t ndc = 0;
    double rderr = 0;
};

// Modified searchKnn with detailed overhead tracking
template<typename T>
std::pair<std::vector<std::pair<float, id_t>>, OverheadStats>
searchKnnWithOverheadTracking(HNSW_NGFix<T>* searcher, T* query_data, int* gt, 
                              size_t k, size_t ef, size_t test_gt_dim) {
    OverheadStats stats;
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    size_t ndc = 0;
    size_t hop_limit = std::numeric_limits<int>::max();
    size_t hop = 0;
    if(ef < k) {
        hop_limit = ef;
        ef = k;
    }

    Search_QuadHeap q0(ef, searcher->visited_list_pool_);
    auto q = &q0;
    
    if(searcher->entry_point >= searcher->max_elements) {
        return std::make_pair(std::vector<std::pair<float, id_t>>(), stats);
    }
    
    // Entry point distance computation
    auto start_dist = std::chrono::high_resolution_clock::now();
    float dist = searcher->getQueryDist(searcher->entry_point, query_data);
    auto end_dist = std::chrono::high_resolution_clock::now();
    stats.distance_computation_time_us += 
        std::chrono::duration_cast<std::chrono::microseconds>(end_dist - start_dist).count();
    stats.num_distance_computations++;
    
    // is_deleted check for entry point
    auto start_check = std::chrono::high_resolution_clock::now();
    bool is_entry_deleted = searcher->is_deleted(searcher->entry_point);
    auto end_check = std::chrono::high_resolution_clock::now();
    stats.is_deleted_check_time_us += 
        std::chrono::duration_cast<std::chrono::microseconds>(end_check - start_check).count();
    stats.num_is_deleted_checks++;
    
    if(is_entry_deleted) {
        stats.num_deleted_nodes_visited++;
    } else {
        stats.num_valid_nodes_visited++;
    }
    
    q->push(searcher->entry_point, dist, is_entry_deleted);
    q->set_visited(searcher->entry_point);
    
    while (!q->is_empty()) {
        if (hop > hop_limit) {
            break;
        }
        
        auto start_queue = std::chrono::high_resolution_clock::now();
        std::pair<float, id_t> current_node_pair = q->get_next_id();
        auto end_queue = std::chrono::high_resolution_clock::now();
        stats.queue_operation_time_us += 
            std::chrono::duration_cast<std::chrono::microseconds>(end_queue - start_queue).count();
        stats.num_queue_operations++;
        
        id_t current_node_id = current_node_pair.second;
        float candidate_dist = -current_node_pair.first;
        bool flag_stop_search = candidate_dist > q->get_dist_bound();

        if (flag_stop_search) {
            break;
        }
        hop += 1;
        
        if(current_node_id >= searcher->max_elements) {
            break;
        }
        
        // Lock acquisition and neighbor access (combined measurement)
        // Note: We can't access node_locks directly, so we measure the combined time
        auto start_lock_neighbor = std::chrono::high_resolution_clock::now();
        
        // We'll use a wrapper approach: call getNeighbors which internally acquires the lock
        // The lock time will be included in neighbor access time
        auto [outs, sz, st] = searcher->getNeighbors(current_node_id);
        
        auto end_lock_neighbor = std::chrono::high_resolution_clock::now();
        uint64_t lock_neighbor_time = 
            std::chrono::duration_cast<std::chrono::microseconds>(end_lock_neighbor - start_lock_neighbor).count();
        
        // Estimate lock time as a small fraction (typically < 5% of neighbor access)
        // This is an approximation since we can't measure lock separately
        stats.lock_acquire_time_us += lock_neighbor_time * 0.05;  // Estimate 5% for lock
        stats.neighbor_access_time_us += lock_neighbor_time * 0.95;  // 95% for neighbor access
        stats.num_lock_acquires++;
        stats.num_neighbor_accesses++;
        
        for (int i = st; i < st + sz; ++i) {
            id_t candidate_id = outs[i];
            if(candidate_id >= searcher->max_elements) {
                continue;
            }
            if(i < st + sz - 1) {
                q->prefetch_visited_list(outs[i+1]);
            }
            
            // Visited check
            auto start_visited = std::chrono::high_resolution_clock::now();
            bool visited = q->is_visited(candidate_id);
            auto end_visited = std::chrono::high_resolution_clock::now();
            stats.visited_check_time_us += 
                std::chrono::duration_cast<std::chrono::microseconds>(end_visited - start_visited).count();
            stats.num_visited_checks++;
            
            if (!visited) {
                q->set_visited(candidate_id);
                
                // Distance computation
                start_dist = std::chrono::high_resolution_clock::now();
                float dist = searcher->getQueryDist(candidate_id, query_data);
                end_dist = std::chrono::high_resolution_clock::now();
                stats.distance_computation_time_us += 
                    std::chrono::duration_cast<std::chrono::microseconds>(end_dist - start_dist).count();
                stats.num_distance_computations++;
                
                // is_deleted check
                start_check = std::chrono::high_resolution_clock::now();
                bool is_candidate_deleted = searcher->is_deleted(candidate_id);
                end_check = std::chrono::high_resolution_clock::now();
                stats.is_deleted_check_time_us += 
                    std::chrono::duration_cast<std::chrono::microseconds>(end_check - start_check).count();
                stats.num_is_deleted_checks++;
                
                if(is_candidate_deleted) {
                    stats.num_deleted_nodes_visited++;
                } else {
                    stats.num_valid_nodes_visited++;
                }
                
                // Queue push
                start_queue = std::chrono::high_resolution_clock::now();
                q->push(candidate_id, dist, is_candidate_deleted);
                end_queue = std::chrono::high_resolution_clock::now();
                stats.queue_operation_time_us += 
                    std::chrono::duration_cast<std::chrono::microseconds>(end_queue - start_queue).count();
                stats.num_queue_operations++;
                
                ndc += 1;
            }
        }
    }
    q->releaseVisitedList();
    
    auto start_queue = std::chrono::high_resolution_clock::now();
    auto res = q->get_result(k);
    auto end_queue = std::chrono::high_resolution_clock::now();
    stats.queue_operation_time_us += 
        std::chrono::duration_cast<std::chrono::microseconds>(end_queue - start_queue).count();
    stats.num_queue_operations++;
    
    auto end_total = std::chrono::high_resolution_clock::now();
    stats.total_time_us = 
        std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
    
    stats.ndc = ndc;
    
    // Calculate recall and rderr
    std::unordered_set<id_t> gtset;
    for(int i = 0; i < k && i < test_gt_dim; ++i) {
        gtset.insert(gt[i]);
    }

    int acc = 0;
    for(size_t i = 0; i < res.size() && i < k; ++i) {
        if(gtset.find(res[i].second) != gtset.end()) {
            ++acc;
            gtset.erase(res[i].second);
        }
    }

    stats.recall = (float)acc / k;
    
    double rderr_sum = 0;
    for(size_t i = 0; i < res.size() && i < k; ++i) {
        float d0 = searcher->getDist(res[i].second, query_data);
        float d1 = searcher->getDist(gt[i], query_data);
        if(fabs(d1) >= 0.00001) {
            rderr_sum += d0 / d1;
        }
    }
    stats.rderr = rderr_sum / k;
    
    // Calculate other time (total - all measured components)
    uint64_t measured_time = stats.distance_computation_time_us + 
                             stats.is_deleted_check_time_us +
                             stats.lock_acquire_time_us +
                             stats.queue_operation_time_us +
                             stats.neighbor_access_time_us +
                             stats.visited_check_time_us;
    if(stats.total_time_us > measured_time) {
        stats.other_time_us = stats.total_time_us - measured_time;
    }
    
    return std::make_pair(res, stats);
}

// Generate random indices for deletion
std::vector<id_t> GenerateRandomDeletionIndices(size_t total_nodes, size_t num_to_delete, size_t seed = 42) {
    std::vector<id_t> indices(total_nodes);
    for(size_t i = 0; i < total_nodes; ++i) {
        indices[i] = i;
    }
    
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    std::vector<id_t> result(indices.begin(), indices.begin() + num_to_delete);
    std::sort(result.begin(), result.end());
    return result;
}

int main(int argc, char* argv[])
{
    int k = 100;
    int test_efs = 100;
    size_t efC_AKNN = 1500;
    size_t efC_delete = 500;
    size_t num_queries = 1000;  // Use fewer queries for detailed analysis
    std::unordered_map<std::string, std::string> paths;
    
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test_query_path")
            paths["test_query_path"] = argv[i + 1];
        if (arg == "--test_gt_path")
            paths["test_gt_path"] = argv[i + 1];
        if (arg == "--index_path")
            paths["index_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--result_path")
            paths["result_path"] = argv[i + 1];
        if (arg == "--K")
            k = std::stoi(argv[i + 1]);
        if (arg == "--efs")
            test_efs = std::stoi(argv[i + 1]);
        if (arg == "--efC_AKNN")
            efC_AKNN = std::stoi(argv[i + 1]);
        if (arg == "--efC_delete")
            efC_delete = std::stoi(argv[i + 1]);
        if (arg == "--num_queries")
            num_queries = std::stoi(argv[i + 1]);
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string index_path = paths["index_path"];
    std::string result_path = paths["result_path"];
    std::string metric_str = paths["metric"];

    std::cout << "=== Deletion Overhead Analysis ===\n";
    std::cout << "test_query_path: " << test_query_path << "\n";
    std::cout << "test_gt_path: " << test_gt_path << "\n";
    std::cout << "index_path: " << index_path << "\n";
    std::cout << "result_path: " << result_path << "\n";
    std::cout << "metric: " << metric_str << "\n";
    std::cout << "K: " << k << "\n";
    std::cout << "efs: " << test_efs << "\n";
    std::cout << "num_queries: " << num_queries << "\n";

    size_t test_number = 0;
    size_t test_gt_dim = 0, vecdim = 0;

    // Load test data
    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    
    num_queries = std::min(num_queries, test_number);
    
    std::cout << "Loaded test data: " << test_number << " queries, dimension: " << vecdim << "\n";
    std::cout << "Using " << num_queries << " queries for analysis\n";
    std::cout << "Test GT dimension: " << test_gt_dim << "\n";

    Metric metric;
    if(metric_str == "ip_float") {
        std::cout << "metric ip\n";
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        std::cout << "metric l2\n";
        metric = L2_float;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }

    // Load original index
    auto hnsw_ngfix_original = new HNSW_NGFix<float>(metric, index_path);
    std::cout << "\n=== Original Index Information ===\n";
    hnsw_ngfix_original->printGraphInfo();
    std::cout << "\n";
    
    size_t total_nodes = hnsw_ngfix_original->n;
    std::cout << "Total nodes in index: " << total_nodes << "\n";

    // Open result file
    std::ofstream output;
    output.open(result_path);
    output << "deletion_type,deletion_percentage,query_idx,"
           << "total_time_us,distance_time_us,is_deleted_time_us,lock_time_us,"
           << "queue_time_us,neighbor_time_us,visited_time_us,other_time_us,"
           << "num_distance,num_is_deleted,num_lock,num_queue,num_neighbor,num_visited,"
           << "num_deleted_visited,num_valid_visited,recall,ndc,rderr\n";
    output.flush();

    // Test deletion percentages: 1% and 2%
    std::vector<float> deletion_percentages = {0.01f, 0.02f};

    for(float del_pct : deletion_percentages) {
        std::cout << "\n========================================\n";
        std::cout << "=== Testing Deletion: " << (del_pct * 100) << "% ===\n";
        std::cout << "========================================\n";
        
        // ========== Lazy Deletion ==========
        std::cout << "\n--- Lazy Deletion ---\n";
        auto hnsw_lazy = new HNSW_NGFix<float>(metric, index_path);
        
        size_t num_to_delete = static_cast<size_t>(total_nodes * del_pct);
        std::vector<id_t> delete_indices = GenerateRandomDeletionIndices(
            total_nodes, num_to_delete, static_cast<size_t>(del_pct * 1000));
        
        for(id_t id : delete_indices) {
            if(id < total_nodes && id != hnsw_lazy->entry_point) {
                hnsw_lazy->DeletePointByFlag(id);
            }
        }
        
        std::cout << "Running " << num_queries << " queries with lazy deletion...\n";
        for(size_t q = 0; q < num_queries; ++q) {
            auto gt = test_gt + q * test_gt_dim;
            auto [results, stats] = searchKnnWithOverheadTracking(
                hnsw_lazy, test_query + q * vecdim, gt, k, test_efs, test_gt_dim);
            
            output << "lazy_deletion," << std::fixed << std::setprecision(2) << (del_pct * 100) << ","
                   << q << ","
                   << stats.total_time_us << ","
                   << stats.distance_computation_time_us << ","
                   << stats.is_deleted_check_time_us << ","
                   << stats.lock_acquire_time_us << ","
                   << stats.queue_operation_time_us << ","
                   << stats.neighbor_access_time_us << ","
                   << stats.visited_check_time_us << ","
                   << stats.other_time_us << ","
                   << stats.num_distance_computations << ","
                   << stats.num_is_deleted_checks << ","
                   << stats.num_lock_acquires << ","
                   << stats.num_queue_operations << ","
                   << stats.num_neighbor_accesses << ","
                   << stats.num_visited_checks << ","
                   << stats.num_deleted_nodes_visited << ","
                   << stats.num_valid_nodes_visited << ","
                   << std::fixed << std::setprecision(6) << stats.recall << ","
                   << stats.ndc << ","
                   << std::fixed << std::setprecision(6) << stats.rderr << "\n";
            
            if((q + 1) % 100 == 0) {
                std::cout << "  Processed " << (q + 1) << " queries...\n";
                output.flush();
            }
        }
        output.flush();
        delete hnsw_lazy;
        
        // ========== Real Deletion ==========
        std::cout << "\n--- Real Deletion ---\n";
        auto hnsw_real = new HNSW_NGFix<float>(metric, index_path);
        
        delete_indices = GenerateRandomDeletionIndices(
            total_nodes, num_to_delete, static_cast<size_t>(del_pct * 1000));
        
        for(id_t id : delete_indices) {
            if(id < total_nodes && id != hnsw_real->entry_point) {
                hnsw_real->DeletePointByFlag(id);
            }
        }
        
        std::cout << "Performing real deletion with NGFix repair...\n";
        hnsw_real->DeleteAllFlagPointsByNGFix(efC_delete, 32);
        
        std::cout << "Running " << num_queries << " queries with real deletion...\n";
        for(size_t q = 0; q < num_queries; ++q) {
            auto gt = test_gt + q * test_gt_dim;
            auto [results, stats] = searchKnnWithOverheadTracking(
                hnsw_real, test_query + q * vecdim, gt, k, test_efs, test_gt_dim);
            
            output << "real_deletion," << std::fixed << std::setprecision(2) << (del_pct * 100) << ","
                   << q << ","
                   << stats.total_time_us << ","
                   << stats.distance_computation_time_us << ","
                   << stats.is_deleted_check_time_us << ","
                   << stats.lock_acquire_time_us << ","
                   << stats.queue_operation_time_us << ","
                   << stats.neighbor_access_time_us << ","
                   << stats.visited_check_time_us << ","
                   << stats.other_time_us << ","
                   << stats.num_distance_computations << ","
                   << stats.num_is_deleted_checks << ","
                   << stats.num_lock_acquires << ","
                   << stats.num_queue_operations << ","
                   << stats.num_neighbor_accesses << ","
                   << stats.num_visited_checks << ","
                   << stats.num_deleted_nodes_visited << ","
                   << stats.num_valid_nodes_visited << ","
                   << std::fixed << std::setprecision(6) << stats.recall << ","
                   << stats.ndc << ","
                   << std::fixed << std::setprecision(6) << stats.rderr << "\n";
            
            if((q + 1) % 100 == 0) {
                std::cout << "  Processed " << (q + 1) << " queries...\n";
                output.flush();
            }
        }
        output.flush();
        delete hnsw_real;
    }

    output.close();
    
    delete []test_query;
    delete []test_gt;
    delete hnsw_ngfix_original;
    
    std::cout << "\n=== Analysis completed. Results saved to: " << result_path << " ===\n";
    
    return 0;
}
