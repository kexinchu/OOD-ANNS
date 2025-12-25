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

using namespace ours;
namespace fs = std::filesystem;

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
    // Configuration
    std::string base_index_path = "/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M16_efC500_MEX48.index";
    std::string train_gt_path;
    std::string output_index_dir = "/workspace/OOD-ANNS/Ours/data/t2i-10M";
    std::string metric_str = "ip_float";
    size_t k = 100;
    size_t ef_search = 100;
    float recall_threshold = 0.92f;  // 92% recall threshold
    size_t max_queries = 0;  // 0 means process all
    
    // Parse command line arguments
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base_index_path" && i + 1 < argc)
            base_index_path = argv[i + 1];
        if (arg == "--train_gt_path" && i + 1 < argc)
            train_gt_path = argv[i + 1];
        if (arg == "--output_index_dir" && i + 1 < argc)
            output_index_dir = argv[i + 1];
        if (arg == "--metric" && i + 1 < argc)
            metric_str = argv[i + 1];
        if (arg == "--K" && i + 1 < argc)
            k = std::stoi(argv[i + 1]);
        if (arg == "--efSearch" && i + 1 < argc)
            ef_search = std::stoi(argv[i + 1]);
        if (arg == "--recall_threshold" && i + 1 < argc)
            recall_threshold = std::stof(argv[i + 1]);
        if (arg == "--max_queries" && i + 1 < argc)
            max_queries = std::stoi(argv[i + 1]);
    }
    
    // Try to find train.gt.bin if not provided
    if(train_gt_path.empty()) {
        // Try common locations
        std::vector<std::string> possible_paths = {
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train.gt.bin",
            "/workspace/RoarGraph/data/t2i-10M/train.gt.bin",
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train_gt.bin",
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train.gt.ibin",
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train_gt.ibin"
        };
        
        for(const auto& path : possible_paths) {
            if(fs::exists(path)) {
                train_gt_path = path;
                std::cout << "Found train GT file at: " << train_gt_path << std::endl;
                break;
            }
        }
        
        if(train_gt_path.empty()) {
            std::cerr << "Error: train_gt_path not provided and not found in common locations." << std::endl;
            std::cerr << "Please provide --train_gt_path argument." << std::endl;
            return 1;
        }
    }
    
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Base index path: " << base_index_path << std::endl;
    std::cout << "Train GT path: " << train_gt_path << std::endl;
    std::cout << "Output index directory: " << output_index_dir << std::endl;
    std::cout << "Metric: " << metric_str << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "efSearch: " << ef_search << std::endl;
    std::cout << "Recall threshold: " << recall_threshold << std::endl;
    if(max_queries > 0) {
        std::cout << "Max queries to process: " << max_queries << std::endl;
    } else {
        std::cout << "Processing all queries" << std::endl;
    }
    std::cout << std::endl;
    
    // Check if base index exists
    if(!fs::exists(base_index_path)) {
        std::cerr << "Error: Base index file not found: " << base_index_path << std::endl;
        return 1;
    }
    
    // Check if train GT file exists
    if(!fs::exists(train_gt_path)) {
        std::cerr << "Error: Train GT file not found: " << train_gt_path << std::endl;
        return 1;
    }
    
    // Create output directory if it doesn't exist
    if(!fs::exists(output_index_dir)) {
        fs::create_directories(output_index_dir);
        std::cout << "Created output directory: " << output_index_dir << std::endl;
    }
    
    // Load train GT data
    size_t train_number = 0, train_gt_dim = 0;
    int* train_gt = LoadData<int>(train_gt_path, train_number, train_gt_dim);
    std::cout << "Loaded " << train_number << " training GT entries" << std::endl;
    std::cout << "GT dimension: " << train_gt_dim << std::endl;
    std::cout << std::endl;
    
    // Determine metric
    Metric metric;
    if(metric_str == "ip_float") {
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        metric = L2_float;
    } else {
        std::cerr << "Error: Unsupported metric type: " << metric_str << std::endl;
        delete[] train_gt;
        return 1;
    }
    
    // Load base index
    std::cout << "=== Loading Base Index ===" << std::endl;
    HNSW_Ours<float>* index = nullptr;
    try {
        index = new HNSW_Ours<float>(metric, base_index_path);
        index->printGraphInfo();
        std::cout << std::endl;
    } catch(const std::exception& e) {
        std::cerr << "Error loading index: " << e.what() << std::endl;
        delete[] train_gt;
        return 1;
    }
    
    size_t vecdim = index->dim;
    std::cout << "Vector dimension: " << vecdim << std::endl;
    std::cout << std::endl;
    
    // We need query vectors to test recall
    // For runtime insert, we need to get query vectors from somewhere
    // Let's try to find train query file
    std::string train_query_path;
    std::vector<std::string> possible_query_paths = {
        "/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin",
        "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train_query.fbin",
        "/workspace/RoarGraph/data/t2i-10M/train_query.fbin",
        "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train_query.bin",
        "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train.fbin",
        "/workspace/RoarGraph/data/t2i-10M/train_query.bin"
    };
    
    for(const auto& path : possible_query_paths) {
        if(fs::exists(path)) {
            train_query_path = path;
            std::cout << "Found train query file at: " << train_query_path << std::endl;
            break;
        }
    }
    
    if(train_query_path.empty()) {
        std::cerr << "Warning: Train query file not found. Cannot test recall." << std::endl;
        std::cerr << "Proceeding with GT-based optimization only..." << std::endl;
    }
    
    float* train_query = nullptr;
    size_t train_query_number = 0, train_query_dim = 0;
    if(!train_query_path.empty()) {
        train_query = LoadData<float>(train_query_path, train_query_number, train_query_dim);
        std::cout << "Loaded " << train_query_number << " training queries" << std::endl;
        std::cout << "Query dimension: " << train_query_dim << std::endl;
        
        if(train_query_dim != vecdim) {
            std::cerr << "Error: Query dimension mismatch: " << train_query_dim << " vs " << vecdim << std::endl;
            delete[] train_gt;
            delete[] train_query;
            delete index;
            return 1;
        }
        
        if(train_query_number != train_number) {
            std::cout << "Warning: Query number (" << train_query_number 
                      << ") != GT number (" << train_number << ")" << std::endl;
            train_number = std::min(train_query_number, train_number);
            std::cout << "Using " << train_number << " queries" << std::endl;
        }
    }
    
    std::cout << std::endl;
    
    // Step 1: Runtime insert and hard query detection
    std::cout << "=== Step 1: Runtime Insert and Hard Query Detection ===" << std::endl;
    std::vector<size_t> hard_query_indices;
    size_t total_queries_processed = 0;
    
    // Limit number of queries if specified
    size_t num_queries_to_process = train_number;
    if(max_queries > 0 && max_queries < train_number) {
        num_queries_to_process = max_queries;
        std::cout << "Limiting to " << num_queries_to_process << " queries for testing" << std::endl;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(size_t i = 0; i < num_queries_to_process; ++i) {
        if(i % 100 == 0 && i > 0) {
            std::cout << "  Processing query " << i << "/" << num_queries_to_process 
                      << " (hard queries: " << hard_query_indices.size() << ")" << std::endl;
        }
        
        int* gt = train_gt + i * train_gt_dim;
        
        // If we have query vectors, test recall first
        bool is_hard = false;
        if(train_query != nullptr) {
            float* query_data = train_query + i * vecdim;
            
            // Query with efSearch=100
            size_t ndc = 0;
            auto results = index->searchKnn(query_data, k, ef_search, ndc);
            
            // Calculate recall
            float recall = CalculateRecall(results, gt, k);
            
            // If recall < 92%, it's a hard query
            if(recall < recall_threshold) {
                is_hard = true;
            }
        } else {
            // Without query vectors, we can't test recall
            // For now, we'll use all queries as potential hard queries
            // This is a fallback - in practice you'd need query vectors
            is_hard = true;
        }
        
        if(is_hard) {
            hard_query_indices.push_back(i);
            
            // Perform runtime insert: use this hard query to optimize the index
            if(train_query != nullptr) {
                float* query_data = train_query + i * vecdim;
                
                // Use NGFix optimization with this hard query
                try {
                    index->NGFixOptimized(query_data, gt, k, k);
                } catch(const std::exception& e) {
                    std::cerr << "Warning: NGFixOptimized failed for query " << i 
                              << ": " << e.what() << std::endl;
                }
            } else {
                // Without query vectors, we can't do NGFix optimization
                // This is a limitation - we need query vectors for optimization
                std::cerr << "Warning: Cannot optimize without query vectors for query " << i << std::endl;
            }
        }
        
        total_queries_processed++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Total queries processed: " << total_queries_processed << std::endl;
    std::cout << "Hard queries found: " << hard_query_indices.size() << std::endl;
    std::cout << "Hard query percentage: " << std::fixed << std::setprecision(2) 
              << (100.0 * hard_query_indices.size() / total_queries_processed) << "%" << std::endl;
    std::cout << "Processing time: " << duration_ms << " ms" << std::endl;
    std::cout << std::endl;
    
    // Step 2: Save optimized index
    std::cout << "=== Step 2: Saving Optimized Index ===" << std::endl;
    std::string output_index_path = output_index_dir + "/optimized_runtime_insert.index";
    
    try {
        index->StoreIndex(output_index_path);
        std::cout << "Optimized index saved to: " << output_index_path << std::endl;
    } catch(const std::exception& e) {
        std::cerr << "Error saving index: " << e.what() << std::endl;
    }
    
    // Save statistics
    std::string stats_path = output_index_dir + "/runtime_insert_stats.txt";
    std::ofstream stats_file(stats_path);
    stats_file << "Runtime Insert Statistics\n";
    stats_file << "========================\n\n";
    stats_file << "Base index: " << base_index_path << "\n";
    stats_file << "Train GT: " << train_gt_path << "\n";
    stats_file << "Total queries processed: " << total_queries_processed << "\n";
    stats_file << "Hard queries found: " << hard_query_indices.size() << "\n";
    stats_file << "Hard query percentage: " << std::fixed << std::setprecision(2) 
               << (100.0 * hard_query_indices.size() / total_queries_processed) << "%\n";
    stats_file << "Recall threshold: " << recall_threshold << "\n";
    stats_file << "K: " << k << "\n";
    stats_file << "efSearch: " << ef_search << "\n";
    stats_file << "Processing time: " << duration_ms << " ms\n";
    stats_file.close();
    
    std::cout << "Statistics saved to: " << stats_path << std::endl;
    std::cout << std::endl;
    
    // Print final graph info
    std::cout << "=== Final Index Info ===" << std::endl;
    index->printGraphInfo();
    std::cout << std::endl;
    
    // Cleanup
    delete[] train_gt;
    if(train_query != nullptr) {
        delete[] train_query;
    }
    delete index;
    
    std::cout << "=== Done ===" << std::endl;
    return 0;
}

