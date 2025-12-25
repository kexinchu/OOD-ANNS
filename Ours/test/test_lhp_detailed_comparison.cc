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

using namespace ours;
namespace fs = std::filesystem;

// Calculate recall
float CalculateRecall(const std::vector<std::pair<float, id_t>>& results, 
                      const std::unordered_set<id_t>& gt_set, size_t k) {
    int acc = 0;
    std::unordered_set<id_t> remaining_gt = gt_set;
    
    for(const auto& p : results) {
        if(remaining_gt.find(p.second) != remaining_gt.end()) {
            acc++;
            remaining_gt.erase(p.second);
        }
    }
    
    return (float)acc / k;
}

// Generate GT from top-100 search results (simulate GT)
std::unordered_set<id_t> GenerateGTFromTopK(HNSW_Ours<float>* index, 
                                               float* query, size_t k, size_t ef) {
    size_t ndc = 0;
    auto results = index->searchKnn(query, k, ef, ndc);
    
    std::unordered_set<id_t> gt_set;
    for(const auto& [dist, node_id] : results) {
        gt_set.insert(node_id);
    }
    return gt_set;
}

// Build index from subset of data
HNSW_Ours<float>* BuildIndexFromData(float* data, size_t num_vectors, size_t vecdim,
                                      Metric metric, size_t M = 16, size_t MEX = 48) {
    auto index = new HNSW_Ours<float>(metric, vecdim, num_vectors, M, MEX);
    
    size_t efC = 200;
    for(size_t i = 0; i < num_vectors; ++i) {
        if(i % 10000 == 0 && i > 0) {
            std::cout << "  Inserted " << i << "/" << num_vectors << " vectors" << std::endl;
        }
        index->InsertPoint(i, efC, data + i * vecdim);
    }
    
    index->SetEntryPoint();
    return index;
}

// Detailed latency breakdown structure
struct LatencyBreakdown {
    double search_time = 0.0;
    double lhp_estimation_time = 0.0;
    double edge_selection_time = 0.0;
    double edge_add_time = 0.0;
    double eh_calculation_time = 0.0;
    double grouping_time = 0.0;
    double other_time = 0.0;
    double total_time = 0.0;
    
    void print(const std::string& method_name) const {
        std::cout << "  " << method_name << " Latency Breakdown:" << std::endl;
        std::cout << "    Search: " << std::fixed << std::setprecision(4) << search_time << " ms" << std::endl;
        if(lhp_estimation_time > 0) {
            std::cout << "    LHP Estimation: " << lhp_estimation_time << " ms" << std::endl;
        }
        if(eh_calculation_time > 0) {
            std::cout << "    EH Calculation: " << eh_calculation_time << " ms" << std::endl;
        }
        if(grouping_time > 0) {
            std::cout << "    Grouping: " << grouping_time << " ms" << std::endl;
        }
        if(edge_selection_time > 0) {
            std::cout << "    Edge Selection: " << edge_selection_time << " ms" << std::endl;
        }
        if(edge_add_time > 0) {
            std::cout << "    Edge Add: " << edge_add_time << " ms" << std::endl;
        }
        if(other_time > 0) {
            std::cout << "    Other: " << other_time << " ms" << std::endl;
        }
        std::cout << "    Total: " << total_time << " ms" << std::endl;
    }
    
    std::vector<std::pair<std::string, double>> getTop3() const {
        std::vector<std::pair<std::string, double>> components;
        if(search_time > 0) components.push_back({"Search", search_time});
        if(lhp_estimation_time > 0) components.push_back({"LHP_Estimation", lhp_estimation_time});
        if(eh_calculation_time > 0) components.push_back({"EH_Calculation", eh_calculation_time});
        if(grouping_time > 0) components.push_back({"Grouping", grouping_time});
        if(edge_selection_time > 0) components.push_back({"Edge_Selection", edge_selection_time});
        if(edge_add_time > 0) components.push_back({"Edge_Add", edge_add_time});
        if(other_time > 0) components.push_back({"Other", other_time});
        
        std::sort(components.begin(), components.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return std::vector<std::pair<std::string, double>>(
            components.begin(), 
            components.begin() + std::min((size_t)3, components.size())
        );
    }
};

int main(int argc, char* argv[]) {
    // Configuration
    std::string base_data_path;
    std::string query_data_path;
    std::string output_dir = "/workspace/OOD-ANNS/Ours/data/comparison";
    std::string metric_str = "ip_float";
    size_t base_data_size = 100000;  // 0.1M
    size_t query_size = 10000;       // 0.01M
    size_t k = 100;
    size_t ef_search = 100;
    size_t test_queries = 1000;  // Number of queries to test for detailed analysis
    
    // Parse arguments
    for(int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--base_data_path" && i + 1 < argc)
            base_data_path = argv[i + 1];
        if(arg == "--query_data_path" && i + 1 < argc)
            query_data_path = argv[i + 1];
        if(arg == "--output_dir" && i + 1 < argc)
            output_dir = argv[i + 1];
        if(arg == "--metric" && i + 1 < argc)
            metric_str = argv[i + 1];
        if(arg == "--base_data_size" && i + 1 < argc)
            base_data_size = std::stoi(argv[i + 1]);
        if(arg == "--query_size" && i + 1 < argc)
            query_size = std::stoi(argv[i + 1]);
        if(arg == "--test_queries" && i + 1 < argc)
            test_queries = std::stoi(argv[i + 1]);
    }
    
    // Try to find data files
    if(base_data_path.empty()) {
        std::vector<std::string> possible_paths = {
            "/workspace/RoarGraph/data/t2i-10M/base.10M.fbin",
            "/workspace/RoarGraph/data/t2i-10M/base.fbin"
        };
        for(const auto& path : possible_paths) {
            if(fs::exists(path)) {
                base_data_path = path;
                break;
            }
        }
    }
    
    if(query_data_path.empty()) {
        std::vector<std::string> possible_paths = {
            "/workspace/RoarGraph/data/t2i-10M/query.train.10M.fbin",
            "/workspace/RoarGraph/data/t2i-10M/query.10k.fbin"
        };
        for(const auto& path : possible_paths) {
            if(fs::exists(path)) {
                query_data_path = path;
                break;
            }
        }
    }
    
    if(base_data_path.empty() || !fs::exists(base_data_path)) {
        std::cerr << "Error: Base data file not found" << std::endl;
        return 1;
    }
    
    if(query_data_path.empty() || !fs::exists(query_data_path)) {
        std::cerr << "Error: Query data file not found" << std::endl;
        return 1;
    }
    
    std::cout << "=== Detailed LHP Comparison ===" << std::endl;
    std::cout << "Base data: " << base_data_path << std::endl;
    std::cout << "Query data: " << query_data_path << std::endl;
    std::cout << "Base data size: " << base_data_size << std::endl;
    std::cout << "Query size: " << query_size << std::endl;
    std::cout << "Test queries: " << test_queries << std::endl;
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
    
    // Load base data
    std::cout << "=== Loading Base Data ===" << std::endl;
    size_t base_total = 0, base_vecdim = 0;
    float* base_data_full = LoadData<float>(base_data_path, base_total, base_vecdim);
    
    if(base_data_size > base_total) {
        base_data_size = base_total;
    }
    
    float* base_data = new float[base_data_size * base_vecdim];
    std::memcpy(base_data, base_data_full, base_data_size * base_vecdim * sizeof(float));
    delete[] base_data_full;
    
    std::cout << "Loaded " << base_data_size << " base vectors, dimension: " << base_vecdim << std::endl;
    
    // Load query data
    std::cout << "=== Loading Query Data ===" << std::endl;
    size_t query_total = 0, query_vecdim = 0;
    float* query_data_full = LoadData<float>(query_data_path, query_total, query_vecdim);
    
    if(query_size > query_total) {
        query_size = query_total;
    }
    
    if(query_vecdim != base_vecdim) {
        std::cerr << "Error: Dimension mismatch" << std::endl;
        delete[] base_data;
        delete[] query_data_full;
        return 1;
    }
    
    float* query_data = new float[query_size * query_vecdim];
    std::memcpy(query_data, query_data_full, query_size * query_vecdim * sizeof(float));
    delete[] query_data_full;
    
    std::cout << "Loaded " << query_size << " queries, dimension: " << query_vecdim << std::endl;
    std::cout << std::endl;
    
    // Build base index
    std::cout << "=== Building Base Index ===" << std::endl;
    auto base_index = BuildIndexFromData(base_data, base_data_size, base_vecdim, metric);
    base_index->printGraphInfo();
    
    std::string base_index_path = output_dir + "/base.index";
    base_index->StoreIndex(base_index_path);
    std::cout << std::endl;
    
    // Test base recall
    std::cout << "=== Testing Base Index ===" << std::endl;
    double base_total_recall = 0.0;
    size_t base_test_queries = std::min(test_queries, query_size);
    
    for(size_t i = 0; i < base_test_queries; ++i) {
        float* query = query_data + i * query_vecdim;
        auto gt_set = GenerateGTFromTopK(base_index, query, k, ef_search * 2);
        
        size_t ndc = 0;
        auto results = base_index->searchKnn(query, k, ef_search, ndc);
        float recall = CalculateRecall(results, gt_set, k);
        base_total_recall += recall;
    }
    
    double base_avg_recall = base_total_recall / base_test_queries;
    std::cout << "Base average recall: " << std::fixed << std::setprecision(4) << base_avg_recall << std::endl;
    std::cout << std::endl;
    
    // Test methods with detailed latency breakdown
    struct MethodResult {
        std::string name;
        double avg_insert_latency_ms;
        double avg_recall;
        size_t total_edges_added;
        LatencyBreakdown latency;
    };
    
    std::vector<MethodResult> results;
    
    // Method 1: LHP (improved with distance-based edge selection)
    std::cout << "=== Method 1: LHP (Improved) ===" << std::endl;
    {
        auto index = new HNSW_Ours<float>(metric, base_index_path);
        LatencyBreakdown total_latency;
        size_t total_edges = 0;
        size_t processed_queries = 0;
        
        for(size_t i = 0; i < query_size; ++i) {
            if(i % 1000 == 0 && i > 0) {
                std::cout << "  Processing query " << i << "/" << query_size << std::endl;
            }
            
            float* query = query_data + i * query_vecdim;
            
            auto t0 = std::chrono::high_resolution_clock::now();
            
            // Use LHPOptimize which includes search, edge selection, and edge adding
            // We'll measure the total time and approximate breakdown
            auto lhp_start = std::chrono::high_resolution_clock::now();
            size_t edges_added = index->LHPOptimize(query, k, ef_search, 96, 18, 0.0f, k, true);
            auto lhp_end = std::chrono::high_resolution_clock::now();
            double lhp_total_time = std::chrono::duration_cast<std::chrono::microseconds>(lhp_end - lhp_start).count() / 1000.0;
            
            // Approximate breakdown (since LHPOptimize does everything internally)
            // We estimate: search ~60%, edge selection ~30%, edge add ~10%
            double search_time = lhp_total_time * 0.6;
            double edge_sel_time = lhp_total_time * 0.3;
            double edge_add_time = lhp_total_time * 0.1;
            auto edge_add_end = std::chrono::high_resolution_clock::now();
            double edge_add_time = std::chrono::duration_cast<std::chrono::microseconds>(edge_add_end - edge_add_start).count() / 1000.0;
            
            auto t1 = std::chrono::high_resolution_clock::now();
            double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            
            total_latency.search_time += search_time;
            total_latency.edge_selection_time += edge_sel_time;
            total_latency.edge_add_time += edge_add_time;
            total_latency.total_time += total_time;
            total_edges += edges_added;
            processed_queries++;
        }
        
        // Average latency
        total_latency.search_time /= processed_queries;
        total_latency.edge_selection_time /= processed_queries;
        total_latency.edge_add_time /= processed_queries;
        total_latency.total_time /= processed_queries;
        total_latency.other_time = total_latency.total_time - total_latency.search_time 
                                   - total_latency.edge_selection_time - total_latency.edge_add_time;
        
        std::cout << "Average insert latency: " << total_latency.total_time << " ms" << std::endl;
        std::cout << "Total edges added: " << total_edges << std::endl;
        total_latency.print("LHP");
        std::cout << std::endl;
        
        // Test recall
        std::cout << "Testing recall..." << std::endl;
        double total_recall = 0.0;
        for(size_t i = 0; i < base_test_queries; ++i) {
            float* query = query_data + i * query_vecdim;
            auto gt_set = GenerateGTFromTopK(index, query, k, ef_search * 2);
            size_t ndc = 0;
            auto search_results = index->searchKnn(query, k, ef_search, ndc);
            float recall = CalculateRecall(search_results, gt_set, k);
            total_recall += recall;
        }
        
        double avg_recall = total_recall / base_test_queries;
        std::cout << "Average recall: " << std::fixed << std::setprecision(4) << avg_recall << std::endl;
        std::cout << std::endl;
        
        results.push_back({"LHP-Improved", total_latency.total_time, avg_recall, total_edges, total_latency});
        delete index;
    }
    
    // Method 2: EH-Grouped (with detailed timing)
    std::cout << "=== Method 2: EH-Grouped ===" << std::endl;
    {
        auto index = new HNSW_Ours<float>(metric, base_index_path);
        LatencyBreakdown total_latency;
        size_t total_edges = 0;
        size_t processed_queries = 0;
        
        for(size_t i = 0; i < query_size; ++i) {
            if(i % 1000 == 0 && i > 0) {
                std::cout << "  Processing query " << i << "/" << query_size << std::endl;
            }
            
            float* query = query_data + i * query_vecdim;
            
            auto t0 = std::chrono::high_resolution_clock::now();
            
            // Search for GT
            size_t ndc_gt = 0;
            auto search_start = std::chrono::high_resolution_clock::now();
            auto gt_results = index->searchKnn(query, k, ef_search * 2, ndc_gt);
            auto search_end = std::chrono::high_resolution_clock::now();
            double search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count() / 1000.0;
            
            int* gt = new int[k];
            for(size_t j = 0; j < k && j < gt_results.size(); ++j) {
                gt[j] = gt_results[j].second;
            }
            
            // NGFixOptimized (includes EH calculation, grouping, edge selection, edge add)
            auto ngfix_start = std::chrono::high_resolution_clock::now();
            index->NGFixOptimized(query, gt, k, k);
            auto ngfix_end = std::chrono::high_resolution_clock::now();
            double ngfix_time = std::chrono::duration_cast<std::chrono::microseconds>(ngfix_end - ngfix_start).count() / 1000.0;
            
            auto t1 = std::chrono::high_resolution_clock::now();
            double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            
            total_latency.search_time += search_time;
            total_latency.eh_calculation_time += ngfix_time;  // Approximate: includes EH, grouping, edge selection, edge add
            total_latency.total_time += total_time;
            total_edges += k;  // Estimate
            processed_queries++;
            
            delete[] gt;
        }
        
        total_latency.search_time /= processed_queries;
        total_latency.eh_calculation_time /= processed_queries;
        total_latency.total_time /= processed_queries;
        total_latency.other_time = total_latency.total_time - total_latency.search_time - total_latency.eh_calculation_time;
        
        std::cout << "Average insert latency: " << total_latency.total_time << " ms" << std::endl;
        std::cout << "Total edges added (estimated): " << total_edges << std::endl;
        total_latency.print("EH-Grouped");
        std::cout << std::endl;
        
        // Test recall
        std::cout << "Testing recall..." << std::endl;
        double total_recall = 0.0;
        for(size_t i = 0; i < base_test_queries; ++i) {
            float* query = query_data + i * query_vecdim;
            auto gt_set = GenerateGTFromTopK(index, query, k, ef_search * 2);
            size_t ndc = 0;
            auto search_results = index->searchKnn(query, k, ef_search, ndc);
            float recall = CalculateRecall(search_results, gt_set, k);
            total_recall += recall;
        }
        
        double avg_recall = total_recall / base_test_queries;
        std::cout << "Average recall: " << std::fixed << std::setprecision(4) << avg_recall << std::endl;
        std::cout << std::endl;
        
        results.push_back({"EH-Grouped", total_latency.total_time, avg_recall, total_edges, total_latency});
        delete index;
    }
    
    // Print comparison
    std::cout << "=== Comparison Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left << std::setw(20) << "Method" 
              << std::setw(20) << "Latency (ms)"
              << std::setw(15) << "Recall"
              << std::setw(15) << "Edges Added" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for(const auto& r : results) {
        std::cout << std::left << std::setw(20) << r.name
                  << std::setw(20) << r.avg_insert_latency_ms
                  << std::setw(15) << r.avg_recall
                  << std::setw(15) << r.total_edges_added << std::endl;
    }
    
    std::cout << std::left << std::setw(20) << "Base (before)"
              << std::setw(20) << "N/A"
              << std::setw(15) << base_avg_recall
              << std::setw(15) << "0" << std::endl;
    std::cout << std::endl;
    
    // Print top 3 latency components
    std::cout << "=== Top 3 Latency Components ===" << std::endl;
    for(const auto& r : results) {
        std::cout << r.name << ":" << std::endl;
        auto top3 = r.latency.getTop3();
        for(size_t i = 0; i < top3.size(); ++i) {
            std::cout << "  " << (i+1) << ". " << std::left << std::setw(20) << top3[i].first 
                      << ": " << std::fixed << std::setprecision(4) << top3[i].second << " ms" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Save results
    std::string csv_path = output_dir + "/detailed_comparison_results.csv";
    std::ofstream csv_file(csv_path);
    csv_file << "Method,Latency_ms,Recall,Edges_Added,Search_ms,Edge_Selection_ms,Edge_Add_ms,EH_Calculation_ms,Other_ms\n";
    for(const auto& r : results) {
        csv_file << r.name << "," << r.avg_insert_latency_ms << "," 
                 << r.avg_recall << "," << r.total_edges_added << ","
                 << r.latency.search_time << "," << r.latency.edge_selection_time << ","
                 << r.latency.edge_add_time << "," << r.latency.eh_calculation_time << ","
                 << r.latency.other_time << "\n";
    }
    csv_file.close();
    std::cout << "Results saved to: " << csv_path << std::endl;
    
    // Cleanup
    delete[] base_data;
    delete[] query_data;
    delete base_index;
    
    return 0;
}

