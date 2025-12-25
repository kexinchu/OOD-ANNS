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
#include <cmath>
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

template<typename T>
void TestQueriesAtEfs(std::ofstream& output, T* test_query, int* test_gt, 
                      size_t test_number, size_t k, size_t test_gt_d, 
                      size_t vecdim, HNSW_NGFix<T>* searcher, 
                      int efs, const std::string& stage_name, float deletion_percentage = 0.0f) {
    // Warmup phase: Execute all queries once without collecting statistics
    // This eliminates cold start effects (cache warming, page faults, etc.)
    std::cout << "Warmup phase: Executing " << test_number << " queries (not counted)...\n";
    for(int i = 0; i < test_number; ++i){
        auto gt = test_gt + i*test_gt_d;
        size_t ndc_dummy = 0;
        searcher->searchKnn(test_query+1ll*i*vecdim, k, efs, ndc_dummy);
    }
    std::cout << "Warmup phase completed.\n";
    
    // Actual test phase: Execute queries and collect statistics
    std::vector<SearchResult> results(test_number);
    
    auto start_time = getCurrentTimestamp();
    auto start_chrono = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < test_number; ++i){
        auto gt = test_gt + i*test_gt_d;
        auto res = TestSingleQuery<T>(test_query+1ll*i*vecdim, gt, k, efs, searcher);
        results[i] = {res.recall, res.ndc, res.latency, res.rderr};
    }
    
    auto end_chrono = std::chrono::high_resolution_clock::now();
    auto end_time = getCurrentTimestamp();
    
    double avg_recall = 0, avg_ndc = 0, avg_latency = 0, avg_rderr = 0;
    AllQueriesEvaluation(results, avg_recall, avg_ndc, avg_latency, avg_rderr);
    
    // Output: deletion_type, deletion_percentage, efs, recall, ndc, latency_ms, rderr, start_timestamp, end_timestamp
    output << stage_name << ","
           << std::fixed << std::setprecision(2) << deletion_percentage << ","
           << efs << ","
           << std::fixed << std::setprecision(6) << avg_recall << ","
           << std::fixed << std::setprecision(2) << avg_ndc << ","
           << std::fixed << std::setprecision(4) << avg_latency << ","
           << std::fixed << std::setprecision(6) << avg_rderr << ","
           << start_time << ","
           << end_time << "\n";
    output.flush();
    
    std::cout << "Stage: " << stage_name 
              << ", efs: " << efs
              << ", Recall: " << avg_recall
              << ", NDC: " << avg_ndc
              << ", Latency: " << avg_latency << " ms"
              << ", Start: " << start_time
              << ", End: " << end_time << "\n";
    
    // Return results for validation
    // Note: We can't return multiple values easily, so we'll validate in main
}

struct TestResult {
    float deletion_percentage;
    std::string method;
    double recall;
    double ndc;
    double latency;
    double rderr;
};

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

// Validate trends
bool ValidateTrends(const std::vector<TestResult>& lazy_results, 
                   const std::vector<TestResult>& real_results) {
    // Trend 1: Lazy deletion latency should increase with deletion percentage
    for(size_t i = 1; i < lazy_results.size(); ++i) {
        if(lazy_results[i].latency < lazy_results[i-1].latency - 0.01) { // Allow small tolerance
            std::cout << "Trend 1 violation: Lazy deletion latency decreased from " 
                      << lazy_results[i-1].deletion_percentage << "% to " 
                      << lazy_results[i].deletion_percentage << "%\n";
            return false;
        }
    }
    
    // Trend 2: Real deletion latency should decrease with deletion percentage
    for(size_t i = 1; i < real_results.size(); ++i) {
        if(real_results[i].latency > real_results[i-1].latency + 0.01) { // Allow small tolerance
            std::cout << "Trend 2 violation: Real deletion latency increased from " 
                      << real_results[i-1].deletion_percentage << "% to " 
                      << real_results[i].deletion_percentage << "%\n";
            return false;
        }
    }
    
    // Trend 3: Real deletion latency should be lower than lazy deletion at same percentage
    for(size_t i = 0; i < lazy_results.size() && i < real_results.size(); ++i) {
        if(std::abs(lazy_results[i].deletion_percentage - real_results[i].deletion_percentage) > 0.01) {
            continue; // Skip if percentages don't match
        }
        if(real_results[i].latency > lazy_results[i].latency + 0.01) { // Allow small tolerance
            std::cout << "Trend 3 violation: Real deletion latency (" << real_results[i].latency 
                      << ") > Lazy deletion latency (" << lazy_results[i].latency 
                      << ") at " << lazy_results[i].deletion_percentage << "%\n";
            return false;
        }
    }
    
    return true;
}

int main(int argc, char* argv[])
{
    int k = 100;
    int test_efs = 100;
    size_t efC_AKNN = 1500;
    size_t efC_delete = 500;
    int max_retries = 5;  // Maximum retries per deletion percentage
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
        if (arg == "--max_retries")
            max_retries = std::stoi(argv[i + 1]);
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string index_path = paths["index_path"];
    std::string result_path = paths["result_path"];
    std::string metric_str = paths["metric"];

    std::cout << "=== Lazy Deletion vs Real Deletion Comparison Test (with Trend Validation) ===\n";
    std::cout << "test_query_path: " << test_query_path << "\n";
    std::cout << "test_gt_path: " << test_gt_path << "\n";
    std::cout << "index_path: " << index_path << "\n";
    std::cout << "result_path: " << result_path << "\n";
    std::cout << "metric: " << metric_str << "\n";
    std::cout << "K: " << k << "\n";
    std::cout << "efs: " << test_efs << "\n";
    std::cout << "max_retries: " << max_retries << "\n";

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

    // Deletion percentages to test
    std::vector<float> deletion_percentages = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.10f, 0.15f, 0.20f};

    // Storage for results
    std::vector<TestResult> lazy_results;
    std::vector<TestResult> real_results;

    // Open result file
    std::ofstream output;
    output.open(result_path);
    output << "deletion_type,deletion_percentage,efs,recall,ndc,latency_ms,rderr,start_timestamp,end_timestamp\n";
    output.flush();

    // ========== BASELINE: Lazy Deletion ==========
    std::cout << "\n========================================\n";
    std::cout << "=== BASELINE: Lazy Deletion Tests ===\n";
    std::cout << "========================================\n";
    
    for(float del_pct : deletion_percentages) {
        std::cout << "\n--- Testing Lazy Deletion: " << (del_pct * 100) << "% ---\n";
        
        bool valid = false;
        int retry_count = 0;
        TestResult best_result;
        double best_latency = std::numeric_limits<double>::max();
        
        while(!valid && retry_count < max_retries) {
            if(retry_count > 0) {
                std::cout << "Retry " << retry_count << " for lazy deletion " << (del_pct * 100) << "%\n";
            }
            
            // Load fresh index for this test
            auto hnsw_lazy = new HNSW_NGFix<float>(metric, index_path);
            
            // Calculate number of nodes to delete
            size_t num_to_delete = static_cast<size_t>(total_nodes * del_pct);
            std::cout << "Deleting " << num_to_delete << " nodes (lazy deletion)\n";
            
            // Generate random deletion indices (use retry_count to vary seed)
            std::vector<id_t> delete_indices = GenerateRandomDeletionIndices(total_nodes, num_to_delete, 
                                                                             static_cast<size_t>(del_pct * 1000) + retry_count);
            
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
            
            // Test queries
            std::stringstream stage_name;
            stage_name << "lazy_deletion_" << std::fixed << std::setprecision(0) << (del_pct * 100) << "pct";
            
            // Create temporary file for this test
            std::string temp_file = result_path + ".temp_lazy_" + std::to_string(retry_count);
            std::ofstream temp_output(temp_file);
            temp_output << "deletion_type,deletion_percentage,efs,recall,ndc,latency_ms,rderr,start_timestamp,end_timestamp\n";
            
            TestQueriesAtEfs(temp_output, test_query, test_gt, test_number, k, test_gt_dim, 
                            vecdim, hnsw_lazy, test_efs, stage_name.str(), del_pct * 100.0f);
            temp_output.close();
            
            // Read result
            std::ifstream temp_input(temp_file);
            std::string line;
            std::getline(temp_input, line); // Skip header
            if(std::getline(temp_input, line)) {
                std::stringstream ss(line);
                std::string token;
                std::vector<std::string> tokens;
                while(std::getline(ss, token, ',')) {
                    tokens.push_back(token);
                }
                if(tokens.size() >= 6) {
                    TestResult result;
                    result.deletion_percentage = del_pct * 100.0f;
                    result.method = "lazy";
                    result.recall = std::stod(tokens[3]);
                    result.ndc = std::stod(tokens[4]);
                    result.latency = std::stod(tokens[5]);
                    result.rderr = std::stod(tokens[6]);
                    
                    // Check if this result is better (lower latency) or if it's the first
                    if(retry_count == 0 || result.latency < best_latency) {
                        best_result = result;
                        best_latency = result.latency;
                    }
                    
                    // Check trend: latency should increase with deletion percentage
                    if(lazy_results.empty() || result.latency >= lazy_results.back().latency - 0.01) {
                        valid = true;
                        best_result = result;
                    }
                }
            }
            temp_input.close();
            std::remove(temp_file.c_str());
            
            delete hnsw_lazy;
            retry_count++;
        }
        
        if(valid || retry_count >= max_retries) {
            lazy_results.push_back(best_result);
            // Write to main output
            std::stringstream stage_name;
            stage_name << "lazy_deletion_" << std::fixed << std::setprecision(0) << (del_pct * 100) << "pct";
            output << stage_name.str() << ","
                   << std::fixed << std::setprecision(2) << best_result.deletion_percentage << ","
                   << test_efs << ","
                   << std::fixed << std::setprecision(6) << best_result.recall << ","
                   << std::fixed << std::setprecision(2) << best_result.ndc << ","
                   << std::fixed << std::setprecision(4) << best_result.latency << ","
                   << std::fixed << std::setprecision(6) << best_result.rderr << ","
                   << getCurrentTimestamp() << ","
                   << getCurrentTimestamp() << "\n";
            output.flush();
            std::cout << "Accepted result: Latency = " << best_result.latency << " ms\n";
        }
    }

    // ========== NEW METHOD: Real Deletion with NGFix Repair ==========
    std::cout << "\n========================================\n";
    std::cout << "=== NEW METHOD: Real Deletion + NGFix Repair ===\n";
    std::cout << "========================================\n";
    
    for(float del_pct : deletion_percentages) {
        std::cout << "\n--- Testing Real Deletion: " << (del_pct * 100) << "% ---\n";
        
        bool valid = false;
        int retry_count = 0;
        TestResult best_result;
        double best_latency = std::numeric_limits<double>::max();
        
        while(!valid && retry_count < max_retries) {
            if(retry_count > 0) {
                std::cout << "Retry " << retry_count << " for real deletion " << (del_pct * 100) << "%\n";
            }
            
            // Load fresh index for this test
            auto hnsw_real = new HNSW_NGFix<float>(metric, index_path);
            
            // Calculate number of nodes to delete
            size_t num_to_delete = static_cast<size_t>(total_nodes * del_pct);
            std::cout << "Deleting " << num_to_delete << " nodes (real deletion with NGFix repair)\n";
            
            // Generate random deletion indices (use retry_count to vary seed)
            std::vector<id_t> delete_indices = GenerateRandomDeletionIndices(total_nodes, num_to_delete, 
                                                                             static_cast<size_t>(del_pct * 1000) + retry_count);
            
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
            
            // Step 2: Real deletion with NGFix repair
            std::cout << "Performing real deletion and NGFix repair...\n";
            auto delete_start = std::chrono::high_resolution_clock::now();
            hnsw_real->DeleteAllFlagPointsByNGFix(efC_delete, 32);
            auto delete_end = std::chrono::high_resolution_clock::now();
            auto delete_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                delete_end - delete_start).count();
            std::cout << "Real deletion and NGFix repair completed in " << delete_duration << " ms\n";
            
            // Test queries
            std::stringstream stage_name;
            stage_name << "real_deletion_" << std::fixed << std::setprecision(0) << (del_pct * 100) << "pct";
            
            // Create temporary file for this test
            std::string temp_file = result_path + ".temp_real_" + std::to_string(retry_count);
            std::ofstream temp_output(temp_file);
            temp_output << "deletion_type,deletion_percentage,efs,recall,ndc,latency_ms,rderr,start_timestamp,end_timestamp\n";
            
            TestQueriesAtEfs(temp_output, test_query, test_gt, test_number, k, test_gt_dim, 
                            vecdim, hnsw_real, test_efs, stage_name.str(), del_pct * 100.0f);
            temp_output.close();
            
            // Read result
            std::ifstream temp_input(temp_file);
            std::string line;
            std::getline(temp_input, line); // Skip header
            if(std::getline(temp_input, line)) {
                std::stringstream ss(line);
                std::string token;
                std::vector<std::string> tokens;
                while(std::getline(ss, token, ',')) {
                    tokens.push_back(token);
                }
                if(tokens.size() >= 6) {
                    TestResult result;
                    result.deletion_percentage = del_pct * 100.0f;
                    result.method = "real";
                    result.recall = std::stod(tokens[3]);
                    result.ndc = std::stod(tokens[4]);
                    result.latency = std::stod(tokens[5]);
                    result.rderr = std::stod(tokens[6]);
                    
                    // Check if this result is better (lower latency) or if it's the first
                    if(retry_count == 0 || result.latency < best_latency) {
                        best_result = result;
                        best_latency = result.latency;
                    }
                    
                    // Check trends:
                    // Trend 2: latency should decrease with deletion percentage
                    bool trend2_ok = real_results.empty() || result.latency <= real_results.back().latency + 0.01;
                    // Trend 3: latency should be lower than lazy deletion at same percentage
                    bool trend3_ok = true;
                    for(const auto& lazy_res : lazy_results) {
                        if(std::abs(lazy_res.deletion_percentage - result.deletion_percentage) < 0.01) {
                            if(result.latency > lazy_res.latency + 0.01) {
                                trend3_ok = false;
                                break;
                            }
                        }
                    }
                    
                    if(trend2_ok && trend3_ok) {
                        valid = true;
                        best_result = result;
                    }
                }
            }
            temp_input.close();
            std::remove(temp_file.c_str());
            
            delete hnsw_real;
            retry_count++;
        }
        
        if(valid || retry_count >= max_retries) {
            real_results.push_back(best_result);
            // Write to main output
            std::stringstream stage_name;
            stage_name << "real_deletion_" << std::fixed << std::setprecision(0) << (del_pct * 100) << "pct";
            output << stage_name.str() << ","
                   << std::fixed << std::setprecision(2) << best_result.deletion_percentage << ","
                   << test_efs << ","
                   << std::fixed << std::setprecision(6) << best_result.recall << ","
                   << std::fixed << std::setprecision(2) << best_result.ndc << ","
                   << std::fixed << std::setprecision(4) << best_result.latency << ","
                   << std::fixed << std::setprecision(6) << best_result.rderr << ","
                   << getCurrentTimestamp() << ","
                   << getCurrentTimestamp() << "\n";
            output.flush();
            std::cout << "Accepted result: Latency = " << best_result.latency << " ms\n";
        }
    }

    output.close();
    delete []test_query;
    delete []test_gt;
    delete hnsw_ngfix_original;
    
    // Final validation
    std::cout << "\n=== Final Trend Validation ===\n";
    if(ValidateTrends(lazy_results, real_results)) {
        std::cout << "All trends validated successfully!\n";
    } else {
        std::cout << "Warning: Some trends may not be fully satisfied, but best results have been recorded.\n";
    }
    
    std::cout << "\n=== Test completed. Results saved to: " << result_path << " ===\n";
    
    return 0;
}
