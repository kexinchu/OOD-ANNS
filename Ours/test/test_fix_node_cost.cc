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

using namespace ours;
namespace fs = std::filesystem;

// Test 1: Insert Node
// Both methods: search -> get topk-ANN -> insert node -> add edges (NO local connectivity fix)

// Method 1: NGFix insert method
void NGFixInsertMethod(HNSW_Ours<float>* index, float* data, id_t node_id, size_t k, size_t ef_search, size_t efC,
                       double& search_time, double& insert_time) {
    // Step 1: Search to get top-k neighbors (using efSearch)
    size_t ndc = 0;
    auto search_start = std::chrono::high_resolution_clock::now();
    auto search_results = index->searchKnn(data, k, ef_search, ndc);
    auto search_end = std::chrono::high_resolution_clock::now();
    search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count() / 1000.0;
    
    // Step 2: Insert node and add edges (no local connectivity fix)
    auto insert_start = std::chrono::high_resolution_clock::now();
    index->InsertPoint(node_id, efC, data);
    // Note: InsertPoint already adds edges based on search results, so we don't need to add edges separately
    auto insert_end = std::chrono::high_resolution_clock::now();
    insert_time = std::chrono::duration_cast<std::chrono::microseconds>(insert_end - insert_start).count() / 1000.0;
}

// Method 2: Ours insert method (same as NGFix, no local connectivity fix)
void OursInsertMethod(HNSW_Ours<float>* index, float* data, id_t node_id, size_t k, size_t ef_search, size_t efC,
                      double& search_time, double& insert_time) {
    // Same as NGFix method - no local connectivity fix
    NGFixInsertMethod(index, data, node_id, k, ef_search, efC, search_time, insert_time);
}

int main(int argc, char* argv[]) {
    // Configuration
    std::string base_index_path;
    std::string insert_data_path;  // Data for insertion
    std::string output_dir = "/workspace/OOD-ANNS/Ours/data/fix_node_cost_test";
    std::string metric_str = "ip_float";
    size_t num_insert_ops = 10000;  // Insert 10K nodes
    size_t k = 100;
    size_t efC = 200;
    std::vector<size_t> ef_search_values = {100, 200, 300, 400, 500, 1000, 1500, 2000};
    
    // Parse arguments
    for(int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--base_index_path" && i + 1 < argc)
            base_index_path = argv[i + 1];
        if(arg == "--insert_data_path" && i + 1 < argc)
            insert_data_path = argv[i + 1];
        if(arg == "--output_dir" && i + 1 < argc)
            output_dir = argv[i + 1];
        if(arg == "--metric" && i + 1 < argc)
            metric_str = argv[i + 1];
        if(arg == "--num_insert_ops" && i + 1 < argc)
            num_insert_ops = std::stoi(argv[i + 1]);
        if(arg == "--k" && i + 1 < argc)
            k = std::stoi(argv[i + 1]);
        if(arg == "--efC" && i + 1 < argc)
            efC = std::stoi(argv[i + 1]);
    }
    
    // Try to find 8M base index (for insert test)
    if(base_index_path.empty()) {
        std::vector<std::string> possible_paths = {
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSWBottom_M16_efC500_MEX48_8M.index",
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/t2i_10M_HNSW_NGFix_M16_efC500_MEX48_AKNN1500_8M.index",
            "/workspace/OOD-ANNS/Ours/data/comparison_10M/base.index"
        };
        for(const auto& path : possible_paths) {
            if(fs::exists(path)) {
                base_index_path = path;
                break;
            }
        }
    }
    
    // Try to find insert data
    if(insert_data_path.empty()) {
        std::vector<std::string> possible_paths = {
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.additional.2M.fbin",
            "/workspace/RoarGraph/data/t2i-10M/base.10M.fbin",
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.fbin"
        };
        for(const auto& path : possible_paths) {
            if(fs::exists(path)) {
                insert_data_path = path;
                break;
            }
        }
    }
    
    if(base_index_path.empty() || !fs::exists(base_index_path)) {
        std::cerr << "Error: Base index file not found" << std::endl;
        return 1;
    }
    
    if(insert_data_path.empty() || !fs::exists(insert_data_path)) {
        std::cerr << "Error: Insert data file not found" << std::endl;
        return 1;
    }
    
    std::cout << "=== Insert Node Cost Comparison Test ===" << std::endl;
    std::cout << "Base index: " << base_index_path << std::endl;
    std::cout << "Insert data: " << insert_data_path << std::endl;
    std::cout << "Number of insert operations: " << num_insert_ops << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "efC: " << efC << std::endl;
    std::cout << "Output dir: " << output_dir << std::endl;
    std::cout << "Metric: " << metric_str << std::endl;
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
    
    // Load base index
    std::cout << "=== Loading Base Index ===" << std::endl;
    auto base_index = new HNSW_Ours<float>(metric, base_index_path);
    base_index->printGraphInfo();
    std::cout << std::endl;
    
    // Load insert data
    std::cout << "=== Loading Insert Data ===" << std::endl;
    size_t insert_total = 0, insert_vecdim = 0;
    float* insert_data_full = LoadData<float>(insert_data_path, insert_total, insert_vecdim);
    std::cout << "Loaded " << insert_total << " vectors, dimension: " << insert_vecdim << std::endl;
    
    if(num_insert_ops > insert_total) {
        std::cerr << "Warning: Requested " << num_insert_ops 
                  << " insertions but only " << insert_total << " vectors available" << std::endl;
        num_insert_ops = insert_total;
    }
    
    // Randomly sample vectors for insertion
    std::vector<size_t> insert_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, insert_total - 1);
    
    std::unordered_set<size_t> selected_indices;
    while(selected_indices.size() < num_insert_ops) {
        selected_indices.insert(dis(gen));
    }
    insert_indices.assign(selected_indices.begin(), selected_indices.end());
    std::shuffle(insert_indices.begin(), insert_indices.end(), gen);
    
    std::cout << "Selected " << num_insert_ops << " random vectors for insertion" << std::endl;
    std::cout << std::endl;
    
    // Determine which nodes to delete to make space (store the list)
    std::vector<id_t> nodes_to_delete_for_space;
    if(base_index->n >= base_index->max_elements) {
        std::cout << "Index is full (" << base_index->n << "/" << base_index->max_elements 
                  << "), will delete " << num_insert_ops << " nodes to make space." << std::endl;
        
        std::random_device rd_space;
        std::mt19937 gen_space(rd_space());
        std::uniform_int_distribution<id_t> node_dis_space(0, base_index->n - 1);
        
        std::unordered_set<id_t> selected_for_space;
        while(selected_for_space.size() < num_insert_ops) {
            id_t node_id = node_dis_space(gen_space);
            if(!base_index->is_deleted(node_id)) {
                selected_for_space.insert(node_id);
            }
        }
        nodes_to_delete_for_space.assign(selected_for_space.begin(), selected_for_space.end());
        std::cout << "Selected " << nodes_to_delete_for_space.size() << " nodes to delete." << std::endl;
    }
    
    // Prepare insert results storage
    std::vector<std::vector<double>> insert_results;
    
    // Test each efSearch value
    for(size_t ef_search : ef_search_values) {
        std::cout << "=== Testing Insert with efSearch = " << ef_search << " ===" << std::endl;
        
        // Method 1: NGFix insert method
        std::cout << "Testing NGFix insert method..." << std::endl;
        auto index1 = new HNSW_Ours<float>(metric, base_index_path);
        
        // Apply deletions to make space
        for(id_t node_id : nodes_to_delete_for_space) {
            index1->set_deleted(node_id);
        }
        
        // Find available node IDs (deleted nodes or beyond current size)
        std::vector<id_t> available_ids;
        for(id_t i = 0; i < index1->max_elements && available_ids.size() < num_insert_ops; ++i) {
            if(i >= index1->n || index1->is_deleted(i)) {
                available_ids.push_back(i);
            }
        }
        
        if(available_ids.size() < num_insert_ops) {
            std::cout << "  Warning: Only " << available_ids.size() << " slots available, requested " << num_insert_ops << std::endl;
        }
        
        size_t max_inserts = std::min(num_insert_ops, available_ids.size());
        
        double ngfix_total_search = 0.0;
        double ngfix_total_insert = 0.0;
        size_t actual_inserts = 0;
        
        for(size_t i = 0; i < max_inserts && i < available_ids.size(); ++i) {
            if(i % 1000 == 0 && i > 0) {
                std::cout << "  Processing insert " << i << "/" << max_inserts << std::endl;
            }
            
            size_t data_idx = insert_indices[i];
            float* data = insert_data_full + data_idx * insert_vecdim;
            id_t node_id = available_ids[i];
            
            double search_time, insert_time;
            NGFixInsertMethod(index1, data, node_id, k, ef_search, efC, search_time, insert_time);
            
            ngfix_total_search += search_time;
            ngfix_total_insert += insert_time;
            actual_inserts++;
        }
        
        if(actual_inserts == 0) {
            std::cerr << "Error: No insertions performed" << std::endl;
            delete index1;
            insert_results.push_back({
                (double)ef_search,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            });
            continue;
        }
        
        double ngfix_avg_search = ngfix_total_search / actual_inserts;
        double ngfix_avg_insert = ngfix_total_insert / actual_inserts;
        double ngfix_avg_total = ngfix_avg_search + ngfix_avg_insert;
        
        std::cout << "NGFix insert method results:" << std::endl;
        std::cout << "  Average search time: " << std::fixed << std::setprecision(4) << ngfix_avg_search << " ms" << std::endl;
        std::cout << "  Average insert time: " << ngfix_avg_insert << " ms" << std::endl;
        std::cout << "  Average total time: " << ngfix_avg_total << " ms" << std::endl;
        std::cout << std::endl;
        
        delete index1;
        
        // Method 2: Ours insert method (same as NGFix for insert)
        std::cout << "Testing Ours insert method..." << std::endl;
        auto index2 = new HNSW_Ours<float>(metric, base_index_path);
        
        // Apply deletions to make space
        for(id_t node_id : nodes_to_delete_for_space) {
            index2->set_deleted(node_id);
        }
        
        // Find available node IDs (same as NGFix)
        std::vector<id_t> ours_available_ids;
        for(id_t i = 0; i < index2->max_elements && ours_available_ids.size() < num_insert_ops; ++i) {
            if(i >= index2->n || index2->is_deleted(i)) {
                ours_available_ids.push_back(i);
            }
        }
        
        size_t ours_max_inserts = std::min(num_insert_ops, ours_available_ids.size());
        
        double ours_total_search = 0.0;
        double ours_total_insert = 0.0;
        size_t ours_actual_inserts = 0;
        
        for(size_t i = 0; i < ours_max_inserts && i < ours_available_ids.size(); ++i) {
            if(i % 1000 == 0 && i > 0) {
                std::cout << "  Processing insert " << i << "/" << ours_max_inserts << std::endl;
            }
            
            size_t data_idx = insert_indices[i];
            float* data = insert_data_full + data_idx * insert_vecdim;
            id_t node_id = ours_available_ids[i];
            
            double search_time, insert_time;
            OursInsertMethod(index2, data, node_id, k, ef_search, efC, search_time, insert_time);
            
            ours_total_search += search_time;
            ours_total_insert += insert_time;
            ours_actual_inserts++;
        }
        
        if(ours_actual_inserts == 0) {
            std::cerr << "Error: No insertions performed" << std::endl;
            delete index2;
            insert_results.push_back({
                (double)ef_search,
                ngfix_avg_search, ngfix_avg_insert, ngfix_avg_total,
                0.0, 0.0, 0.0
            });
            continue;
        }
        
        double ours_avg_search = ours_total_search / ours_actual_inserts;
        double ours_avg_insert = ours_total_insert / ours_actual_inserts;
        double ours_avg_total = ours_avg_search + ours_avg_insert;
        
        std::cout << "Ours insert method results:" << std::endl;
        std::cout << "  Average search time: " << std::fixed << std::setprecision(4) << ours_avg_search << " ms" << std::endl;
        std::cout << "  Average insert time: " << ours_avg_insert << " ms" << std::endl;
        std::cout << "  Average total time: " << ours_avg_total << " ms" << std::endl;
        std::cout << std::endl;
        
        // Store results
        insert_results.push_back({
            (double)ef_search,
            ngfix_avg_search,
            ngfix_avg_insert,
            ngfix_avg_total,
            ours_avg_search,
            ours_avg_insert,
            ours_avg_total
        });
        
        delete index2;
    }
    
    // Save insert results to CSV
    std::string insert_csv_path = output_dir + "/insert_node_results.csv";
    std::ofstream insert_csv_file(insert_csv_path);
    insert_csv_file << "efSearch,NGFix_Search_ms,NGFix_Insert_ms,NGFix_Total_ms,"
                    << "Ours_Search_ms,Ours_Insert_ms,Ours_Total_ms\n";
    
    for(const auto& row : insert_results) {
        for(size_t i = 0; i < row.size(); ++i) {
            insert_csv_file << std::fixed << std::setprecision(4) << row[i];
            if(i < row.size() - 1) insert_csv_file << ",";
        }
        insert_csv_file << "\n";
    }
    insert_csv_file.close();
    
    std::cout << "=== Results Summary ===" << std::endl;
    std::cout << std::left << std::setw(10) << "efSearch"
              << std::setw(20) << "NGFix Total (ms)"
              << std::setw(20) << "Ours Total (ms)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for(const auto& row : insert_results) {
        std::cout << std::left << std::setw(10) << (int)row[0]
                  << std::setw(20) << std::fixed << std::setprecision(4) << row[3]
                  << std::setw(20) << row[6] << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Insert results saved to: " << insert_csv_path << std::endl;
    
    // Cleanup
    delete[] insert_data_full;
    delete base_index;
    
    std::cout << std::endl;
    std::cout << "=== Done ===" << std::endl;
    return 0;
}

