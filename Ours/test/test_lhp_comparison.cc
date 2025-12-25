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
    }
    
    // Try to find data files
    if(base_data_path.empty()) {
        std::vector<std::string> possible_paths = {
            "/workspace/RoarGraph/data/t2i-10M/base.fbin",
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/base.fbin"
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
            "/workspace/RoarGraph/data/t2i-10M/query.10k.fbin",
            "/workspace/OOD-ANNS/NGFix/data/t2i-10M/train_query.fbin"
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
    
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Base data: " << base_data_path << std::endl;
    std::cout << "Query data: " << query_data_path << std::endl;
    std::cout << "Base data size: " << base_data_size << std::endl;
    std::cout << "Query size: " << query_size << std::endl;
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
    
    // Load base data
    std::cout << "=== Loading Base Data ===" << std::endl;
    size_t base_total = 0, base_vecdim = 0;
    float* base_data_full = LoadData<float>(base_data_path, base_total, base_vecdim);
    
    if(base_data_size > base_total) {
        std::cerr << "Warning: Requested " << base_data_size 
                  << " vectors but only " << base_total << " available" << std::endl;
        base_data_size = base_total;
    }
    
    // Sample base_data_size vectors (use first N for simplicity)
    float* base_data = new float[base_data_size * base_vecdim];
    std::memcpy(base_data, base_data_full, base_data_size * base_vecdim * sizeof(float));
    delete[] base_data_full;
    
    std::cout << "Loaded " << base_data_size << " base vectors, dimension: " << base_vecdim << std::endl;
    std::cout << std::endl;
    
    // Load query data
    std::cout << "=== Loading Query Data ===" << std::endl;
    size_t query_total = 0, query_vecdim = 0;
    float* query_data_full = LoadData<float>(query_data_path, query_total, query_vecdim);
    
    if(query_size > query_total) {
        std::cerr << "Warning: Requested " << query_size 
                  << " queries but only " << query_total << " available" << std::endl;
        query_size = query_total;
    }
    
    if(query_vecdim != base_vecdim) {
        std::cerr << "Error: Dimension mismatch: base=" << base_vecdim 
                  << ", query=" << query_vecdim << std::endl;
        delete[] base_data;
        delete[] query_data_full;
        return 1;
    }
    
    // Sample query_size queries
    float* query_data = new float[query_size * query_vecdim];
    std::memcpy(query_data, query_data_full, query_size * query_vecdim * sizeof(float));
    delete[] query_data_full;
    
    std::cout << "Loaded " << query_size << " queries, dimension: " << query_vecdim << std::endl;
    std::cout << std::endl;
    
    // Step 1: Build base index
    std::cout << "=== Step 1: Building Base Index ===" << std::endl;
    auto base_index = BuildIndexFromData(base_data, base_data_size, base_vecdim, metric);
    base_index->printGraphInfo();
    std::cout << std::endl;
    
    // Save base index
    std::string base_index_path = output_dir + "/base.index";
    base_index->StoreIndex(base_index_path);
    std::cout << "Base index saved to: " << base_index_path << std::endl;
    std::cout << std::endl;
    
    // Step 2: Test base index recall (before optimization)
    std::cout << "=== Step 2: Testing Base Index (Before Optimization) ===" << std::endl;
    double base_total_recall = 0.0;
    size_t base_test_queries = std::min(query_size, (size_t)1000);  // Test on subset
    
    for(size_t i = 0; i < base_test_queries; ++i) {
        if(i % 100 == 0 && i > 0) {
            std::cout << "  Testing query " << i << "/" << base_test_queries << std::endl;
        }
        
        float* query = query_data + i * query_vecdim;
        auto gt_set = GenerateGTFromTopK(base_index, query, k, ef_search * 2);
        
        size_t ndc = 0;
        auto results = base_index->searchKnn(query, k, ef_search, ndc);
        float recall = CalculateRecall(results, gt_set, k);
        base_total_recall += recall;
    }
    
    double base_avg_recall = base_total_recall / base_test_queries;
    std::cout << "Base index average recall: " << std::fixed << std::setprecision(4) 
              << base_avg_recall << std::endl;
    std::cout << std::endl;
    
    // Step 3: Test three methods
    struct MethodResult {
        std::string name;
        double avg_insert_latency_ms;
        double avg_search_latency_ms;
        double avg_ndc;
        double avg_recall;
        size_t total_edges_added;
    };
    
    std::vector<MethodResult> results;
    
    // Method 1: LHP
    std::cout << "=== Method 1: LHP Optimization ===" << std::endl;
    {
        auto index = new HNSW_Ours<float>(metric, base_index_path);
        double total_insert_time = 0.0;
        double total_search_time = 0.0;
        size_t total_ndc = 0;
        size_t total_edges = 0;
        size_t processed_queries = 0;
        
        for(size_t i = 0; i < query_size; ++i) {
            if(i % 100 == 0 && i > 0) {
                std::cout << "  Processing query " << i << "/" << query_size << std::endl;
            }
            
            float* query = query_data + i * query_vecdim;
            
            // Measure search latency and NDC
            size_t ndc = 0;
            auto search_start = std::chrono::high_resolution_clock::now();
            auto search_results = index->searchKnn(query, k, ef_search, ndc);
            auto search_end = std::chrono::high_resolution_clock::now();
            double search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count() / 1000.0;
            
            // Measure total optimization latency
            auto start = std::chrono::high_resolution_clock::now();
            // Force add edges for fair comparison with EH methods
            // Use k edges (same as EH methods) instead of kappa
            size_t edges = index->LHPOptimize(query, k, ef_search, 96, 18, 0.0f, k, true);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            total_insert_time += latency_ms;
            total_search_time += search_time;
            total_ndc += ndc;
            total_edges += edges;
            processed_queries++;
        }
        
        double avg_latency = total_insert_time / processed_queries;
        double avg_search_latency = total_search_time / processed_queries;
        double avg_ndc = (double)total_ndc / processed_queries;
        std::cout << "Average insert latency: " << avg_latency << " ms" << std::endl;
        std::cout << "Average search latency: " << avg_search_latency << " ms" << std::endl;
        std::cout << "Average NDC (nodes accessed): " << std::fixed << std::setprecision(2) << avg_ndc << std::endl;
        std::cout << "Total edges added: " << total_edges << std::endl;
        std::cout << std::endl;
        
        // Test recall after LHP optimization
        std::cout << "Testing recall after LHP optimization..." << std::endl;
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
        
        // Save index
        std::string index_path = output_dir + "/lhp_optimized.index";
        index->StoreIndex(index_path);
        std::cout << "LHP optimized index saved to: " << index_path << std::endl;
        std::cout << std::endl;
        
        results.push_back({"LHP", avg_latency, avg_search_latency, avg_ndc, avg_recall, total_edges});
        delete index;
    }
    
    // Method 2: EH-Grouped (NGFixOptimized)
    std::cout << "=== Method 2: EH-Grouped Optimization ===" << std::endl;
    {
        auto index = new HNSW_Ours<float>(metric, base_index_path);
        double total_insert_time = 0.0;
        double total_search_time = 0.0;
        size_t total_ndc = 0;
        size_t total_edges = 0;
        size_t processed_queries = 0;
        
        for(size_t i = 0; i < query_size; ++i) {
            if(i % 100 == 0 && i > 0) {
                std::cout << "  Processing query " << i << "/" << query_size << std::endl;
            }
            
            float* query = query_data + i * query_vecdim;
            
            // Generate GT from top-k search (measure search latency and NDC)
            size_t ndc_gt = 0;
            auto search_start = std::chrono::high_resolution_clock::now();
            auto gt_results = index->searchKnn(query, k, ef_search * 2, ndc_gt);
            auto search_end = std::chrono::high_resolution_clock::now();
            double search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count() / 1000.0;
            
            int* gt = new int[k];
            for(size_t j = 0; j < k && j < gt_results.size(); ++j) {
                gt[j] = gt_results[j].second;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            index->NGFixOptimized(query, gt, k, k);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            total_insert_time += latency_ms;
            total_search_time += search_time;
            total_ndc += ndc_gt;
            processed_queries++;
            
            // Count edges (approximate)
            total_edges += k;  // Rough estimate
            
            delete[] gt;
        }
        
        double avg_latency = total_insert_time / processed_queries;
        double avg_search_latency = total_search_time / processed_queries;
        double avg_ndc = (double)total_ndc / processed_queries;
        std::cout << "Average insert latency: " << avg_latency << " ms" << std::endl;
        std::cout << "Average search latency: " << avg_search_latency << " ms" << std::endl;
        std::cout << "Average NDC (nodes accessed): " << std::fixed << std::setprecision(2) << avg_ndc << std::endl;
        std::cout << "Total edges added (estimated): " << total_edges << std::endl;
        std::cout << std::endl;
        
        // Test recall
        std::cout << "Testing recall after EH-Grouped optimization..." << std::endl;
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
        
        std::string index_path = output_dir + "/eh_grouped_optimized.index";
        index->StoreIndex(index_path);
        std::cout << "EH-Grouped optimized index saved to: " << index_path << std::endl;
        std::cout << std::endl;
        
        results.push_back({"EH-Grouped", avg_latency, avg_search_latency, avg_ndc, avg_recall, total_edges});
        delete index;
    }
    
    // Method 3: NGFix EH (original)
    std::cout << "=== Method 3: NGFix EH (Original) ===" << std::endl;
    {
        auto index = new HNSW_Ours<float>(metric, base_index_path);
        double total_insert_time = 0.0;
        double total_search_time = 0.0;
        size_t total_ndc = 0;
        size_t total_edges = 0;
        size_t processed_queries = 0;
        
        for(size_t i = 0; i < query_size; ++i) {
            if(i % 100 == 0 && i > 0) {
                std::cout << "  Processing query " << i << "/" << query_size << std::endl;
            }
            
            float* query = query_data + i * query_vecdim;
            
            // Generate GT (measure search latency and NDC)
            size_t ndc_gt = 0;
            auto search_start = std::chrono::high_resolution_clock::now();
            auto gt_results = index->searchKnn(query, k, ef_search * 2, ndc_gt);
            auto search_end = std::chrono::high_resolution_clock::now();
            double search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count() / 1000.0;
            
            int* gt = new int[k];
            for(size_t j = 0; j < k && j < gt_results.size(); ++j) {
                gt[j] = gt_results[j].second;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            index->NGFix(query, gt, k, k);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            total_insert_time += latency_ms;
            total_search_time += search_time;
            total_ndc += ndc_gt;
            processed_queries++;
            total_edges += k;  // Rough estimate
            
            delete[] gt;
        }
        
        double avg_latency = total_insert_time / processed_queries;
        double avg_search_latency = total_search_time / processed_queries;
        double avg_ndc = (double)total_ndc / processed_queries;
        std::cout << "Average insert latency: " << avg_latency << " ms" << std::endl;
        std::cout << "Average search latency: " << avg_search_latency << " ms" << std::endl;
        std::cout << "Average NDC (nodes accessed): " << std::fixed << std::setprecision(2) << avg_ndc << std::endl;
        std::cout << "Total edges added (estimated): " << total_edges << std::endl;
        std::cout << std::endl;
        
        // Test recall
        std::cout << "Testing recall after NGFix EH optimization..." << std::endl;
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
        
        std::string index_path = output_dir + "/ngfix_eh_optimized.index";
        index->StoreIndex(index_path);
        std::cout << "NGFix EH optimized index saved to: " << index_path << std::endl;
        std::cout << std::endl;
        
        results.push_back({"NGFix-EH", avg_latency, avg_search_latency, avg_ndc, avg_recall, total_edges});
        delete index;
    }
    
    // Step 4: Generate comparison report
    std::cout << "=== Comparison Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left << std::setw(15) << "Method" 
              << std::setw(18) << "Insert Latency"
              << std::setw(18) << "Search Latency"
              << std::setw(15) << "NDC"
              << std::setw(15) << "Recall"
              << std::setw(15) << "Edges Added" << std::endl;
    std::cout << std::string(96, '-') << std::endl;
    
    for(const auto& r : results) {
        std::cout << std::left << std::setw(15) << r.name
                  << std::setw(18) << r.avg_insert_latency_ms
                  << std::setw(18) << r.avg_search_latency_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << r.avg_ndc
                  << std::setw(15) << std::setprecision(4) << r.avg_recall
                  << std::setw(15) << r.total_edges_added << std::endl;
    }
    
    std::cout << std::left << std::setw(15) << "Base (before)"
              << std::setw(18) << "N/A"
              << std::setw(18) << "N/A"
              << std::setw(15) << "N/A"
              << std::setw(15) << base_avg_recall
              << std::setw(15) << "0" << std::endl;
    std::cout << std::endl;
    
    // Save results to CSV
    std::string csv_path = output_dir + "/comparison_results.csv";
    std::ofstream csv_file(csv_path);
    csv_file << "Method,Insert_Latency_ms,Search_Latency_ms,NDC,Recall,Edges_Added,Base_Recall\n";
    csv_file << "Base,N/A,N/A,N/A," << base_avg_recall << ",0," << base_avg_recall << "\n";
    for(const auto& r : results) {
        csv_file << r.name << "," << r.avg_insert_latency_ms << "," 
                 << r.avg_search_latency_ms << "," << r.avg_ndc << ","
                 << r.avg_recall << "," << r.total_edges_added << "," << base_avg_recall << "\n";
    }
    csv_file.close();
    std::cout << "Results saved to: " << csv_path << std::endl;
    
    // Cleanup
    delete[] base_data;
    delete[] query_data;
    delete base_index;
    
    std::cout << std::endl;
    std::cout << "=== Done ===" << std::endl;
    return 0;
}

