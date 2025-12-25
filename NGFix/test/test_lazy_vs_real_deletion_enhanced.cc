#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include "ngfixlib/utils/search_list.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
using namespace ngfixlib;

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

struct EnhancedSearchResult {
    float recall;
    size_t ndc;
    int64_t latency;
    double rderr;
    size_t max_candidate_set_size;
};

template<typename T>
EnhancedSearchResult TestSingleQueryEnhanced(T* query_data, int* gt, size_t k, size_t efs, HNSW_NGFix<T>* searcher) {
    size_t ndc = 0;
    size_t max_candidate_set_size = 0;

    const unsigned long Converter = 1000 * 1000;
    struct timeval val;
    int ret = gettimeofday(&val, NULL);

    // Use the enhanced searchKnnWithMetrics function
    auto aknns = searcher->searchKnnWithMetrics(query_data, k, efs, ndc, max_candidate_set_size);
    
    struct timeval newVal;
    ret = gettimeofday(&newVal, NULL);
    int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

    std::unordered_set<id_t> gtset;
    for(int i = 0; i < k; ++i) {
        gtset.insert(gt[i]);
    }

    int acc = 0;
    for(int i = 0; i < k && i < (int)aknns.size(); ++i) {
        if(gtset.find(aknns[i].second) != gtset.end()) {
            ++acc;
            gtset.erase(aknns[i].second);
        }
    }

    float recall = (float)acc/k;
    double rderr = 0;

    for(int i = 0; i < k && i < (int)aknns.size(); ++i){
        float d0 = searcher->getDist(aknns[i].second, query_data);
        float d1 = searcher->getDist(gt[i], query_data);
        if(fabs(d1) < 0.00001) {continue; }
        rderr += d0/d1;
    }

    return EnhancedSearchResult{recall, ndc, diff, rderr/k, max_candidate_set_size};
}

template<typename T>
void TestQueriesAtEfsEnhanced(std::ofstream& output, T* test_query, int* test_gt, 
                      size_t test_number, size_t k, size_t test_gt_d, 
                      size_t vecdim, HNSW_NGFix<T>* searcher, 
                      int efs, const std::string& stage_name, float deletion_percentage = 0.0f) {
    // Warmup phase
    std::cout << "Warmup phase: Executing " << test_number << " queries (not counted)...\n";
    for(int i = 0; i < test_number; ++i){
        auto gt = test_gt + i*test_gt_d;
        size_t ndc_dummy = 0;
        searcher->searchKnn(test_query+1ll*i*vecdim, k, efs, ndc_dummy);
    }
    std::cout << "Warmup phase completed.\n";
    
    // Actual test phase
    std::vector<EnhancedSearchResult> results(test_number);
    
    auto start_time = getCurrentTimestamp();
    auto start_chrono = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < test_number; ++i){
        auto gt = test_gt + i*test_gt_d;
        auto res = TestSingleQueryEnhanced<T>(test_query+1ll*i*vecdim, gt, k, efs, searcher);
        results[i] = res;
    }
    
    auto end_chrono = std::chrono::high_resolution_clock::now();
    auto end_time = getCurrentTimestamp();
    
    double avg_recall = 0, avg_ndc = 0, avg_latency = 0, avg_rderr = 0;
    size_t max_max_candidate_set_size = 0;
    for(int i = 0; i < test_number; ++i){
        avg_recall += results[i].recall;
        avg_ndc += results[i].ndc;
        avg_rderr += results[i].rderr;
        avg_latency += results[i].latency;
        if(results[i].max_candidate_set_size > max_max_candidate_set_size) {
            max_max_candidate_set_size = results[i].max_candidate_set_size;
        }
    }
    avg_rderr /= test_number;
    avg_recall /= test_number;
    avg_latency /= test_number;
    avg_latency /= 1000;
    avg_ndc /= test_number;
    
    // Output: deletion_type, deletion_percentage, efs, recall, ndc, latency_ms, rderr, max_candidate_set_size, start_timestamp, end_timestamp
    output << stage_name << ","
           << std::fixed << std::setprecision(2) << deletion_percentage << ","
           << efs << ","
           << std::fixed << std::setprecision(6) << avg_recall << ","
           << std::fixed << std::setprecision(2) << avg_ndc << ","
           << std::fixed << std::setprecision(4) << avg_latency << ","
           << std::fixed << std::setprecision(6) << avg_rderr << ","
           << max_max_candidate_set_size << ","
           << start_time << ","
           << end_time << "\n";
    output.flush();
    
    std::cout << "Stage: " << stage_name 
              << ", efs: " << efs
              << ", Recall: " << avg_recall
              << ", NDC: " << avg_ndc
              << ", Latency: " << avg_latency << " ms"
              << ", Max Candidate Set Size: " << max_max_candidate_set_size
              << ", Start: " << start_time
              << ", End: " << end_time << "\n";
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
    std::vector<int> test_efs_list = {100, 200, 300, 400, 500};
    size_t efC_AKNN = 1500;
    size_t efC_delete = 500;
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
        if (arg == "--efC_AKNN")
            efC_AKNN = std::stoi(argv[i + 1]);
        if (arg == "--efC_delete")
            efC_delete = std::stoi(argv[i + 1]);
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string index_path = paths["index_path"];
    std::string result_path = paths["result_path"];
    std::string metric_str = paths["metric"];

    std::cout << "=== Enhanced Lazy Deletion vs Real Deletion Comparison Test ===\n";
    std::cout << "test_query_path: " << test_query_path << "\n";
    std::cout << "test_gt_path: " << test_gt_path << "\n";
    std::cout << "index_path: " << index_path << "\n";
    std::cout << "result_path: " << result_path << "\n";
    std::cout << "metric: " << metric_str << "\n";
    std::cout << "K: " << k << "\n";
    std::cout << "efs list: ";
    for(auto efs : test_efs_list) {
        std::cout << efs << " ";
    }
    std::cout << "\n";

    size_t test_number = 0;
    size_t test_gt_dim = 0, vecdim = 0;

    // Load test data
    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    
    std::cout << "Loaded test data: " << test_number << " queries, dimension: " << vecdim << "\n";
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
    output << "deletion_type,deletion_percentage,efs,recall,ndc,latency_ms,rderr,max_candidate_set_size,start_timestamp,end_timestamp\n";
    output.flush();

    // Deletion percentages to test
    std::vector<float> deletion_percentages = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.10f, 0.15f, 0.20f};

    // ========== BASELINE: Lazy Deletion ==========
    std::cout << "\n========================================\n";
    std::cout << "=== BASELINE: Lazy Deletion Tests ===\n";
    std::cout << "========================================\n";
    
    for(float del_pct : deletion_percentages) {
        std::cout << "\n--- Testing Lazy Deletion: " << (del_pct * 100) << "% ---\n";
        
        // Load fresh index for this test
        auto hnsw_lazy = new HNSW_NGFix<float>(metric, index_path);
        
        // Calculate number of nodes to delete
        size_t num_to_delete = static_cast<size_t>(total_nodes * del_pct);
        std::cout << "Deleting " << num_to_delete << " nodes (lazy deletion)\n";
        
        // Generate random deletion indices
        std::vector<id_t> delete_indices = GenerateRandomDeletionIndices(total_nodes, num_to_delete, 
                                                                         static_cast<size_t>(del_pct * 1000));
        
        // Perform lazy deletion (just mark as deleted)
        auto lazy_start = std::chrono::high_resolution_clock::now();
        for(id_t id : delete_indices) {
            if(id < total_nodes && id != hnsw_lazy->entry_point) {
                hnsw_lazy->DeletePointByFlag(id);
            }
        }
        auto lazy_end = std::chrono::high_resolution_clock::now();
        auto lazy_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            lazy_end - lazy_start).count();
        std::cout << "Lazy deletion completed in " << lazy_duration << " ms\n";
        
        // Test queries at different efSearch values
        for(int test_efs : test_efs_list) {
            std::stringstream stage_name;
            stage_name << "lazy_deletion_" << std::fixed << std::setprecision(0) << (del_pct * 100) << "pct";
            TestQueriesAtEfsEnhanced(output, test_query, test_gt, test_number, k, test_gt_dim, 
                            vecdim, hnsw_lazy, test_efs, stage_name.str(), del_pct * 100.0f);
        }
        
        delete hnsw_lazy;
    }

    // ========== IMPROVED METHOD: Real Deletion with Improved NGFix Repair ==========
    std::cout << "\n========================================\n";
    std::cout << "=== IMPROVED METHOD: Real Deletion + Improved NGFix Repair ===\n";
    std::cout << "========================================\n";
    
    for(float del_pct : deletion_percentages) {
        std::cout << "\n--- Testing Real Deletion: " << (del_pct * 100) << "% ---\n";
        
        // Load fresh index for this test
        auto hnsw_real = new HNSW_NGFix<float>(metric, index_path);
        
        // Calculate number of nodes to delete
        size_t num_to_delete = static_cast<size_t>(total_nodes * del_pct);
        std::cout << "Deleting " << num_to_delete << " nodes (real deletion with improved NGFix repair)\n";
        
        // Generate random deletion indices (same seed for consistency)
        std::vector<id_t> delete_indices = GenerateRandomDeletionIndices(total_nodes, num_to_delete, 
                                                                         static_cast<size_t>(del_pct * 1000));
        
        // Step 1: Mark nodes for deletion
        auto mark_start = std::chrono::high_resolution_clock::now();
        for(id_t id : delete_indices) {
            if(id < total_nodes && id != hnsw_real->entry_point) {
                hnsw_real->DeletePointByFlag(id);
            }
        }
        auto mark_end = std::chrono::high_resolution_clock::now();
        auto mark_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            mark_end - mark_start).count();
        std::cout << "Marking nodes for deletion completed in " << mark_duration << " ms\n";
        
        // Step 2: Real deletion with improved NGFix repair
        std::cout << "Performing real deletion and improved NGFix repair...\n";
        auto delete_start = std::chrono::high_resolution_clock::now();
        hnsw_real->DeleteAllFlagPointsByNGFix(efC_delete, 32);
        auto delete_end = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            delete_end - delete_start).count();
        std::cout << "Real deletion and improved NGFix repair completed in " << delete_duration << " ms\n";
        
        // Test queries at different efSearch values
        for(int test_efs : test_efs_list) {
            std::stringstream stage_name;
            stage_name << "real_deletion_" << std::fixed << std::setprecision(0) << (del_pct * 100) << "pct";
            TestQueriesAtEfsEnhanced(output, test_query, test_gt, test_number, k, test_gt_dim, 
                            vecdim, hnsw_real, test_efs, stage_name.str(), del_pct * 100.0f);
        }
        
        delete hnsw_real;
    }

    output.close();
    delete []test_query;
    delete []test_gt;
    delete hnsw_ngfix_original;
    
    std::cout << "\n=== Test completed. Results saved to: " << result_path << " ===\n";
    
    return 0;
}

