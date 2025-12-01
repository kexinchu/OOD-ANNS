#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <sstream>

using namespace ngfixlib;

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

// Test metrics structure
struct TestMetrics {
    double avg_recall;
    double avg_latency_ms;
    double avg_accessed_nodes;
    double mean_out_degree;
    
    TestMetrics() : avg_recall(0), avg_latency_ms(0), avg_accessed_nodes(0), mean_out_degree(0) {}
};

// Test a single query and return metrics
std::pair<float, size_t> TestSingleQuery(
    float* query_data, 
    int* gt, 
    size_t k, 
    size_t ef_search, 
    HNSW_NGFix<float>* searcher) {
    
    size_t ndc = 0;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto [results, ndc_result, lw_metrics] = searcher->searchKnnWithLightweightMetrics(
        query_data, k, ef_search, ndc, 0.2f);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    float recall = CalculateRecall(results, gt, k);
    
    return {recall, lw_metrics.S};  // Return recall and accessed nodes (S)
}

// Test all queries and collect metrics
TestMetrics TestAllQueries(
    float* test_query,
    int* test_gt,
    size_t test_number,
    size_t k,
    size_t test_gt_dim,
    size_t vecdim,
    size_t ef_search,
    HNSW_NGFix<float>* searcher) {
    
    TestMetrics metrics;
    double total_recall = 0;
    double total_latency_us = 0;
    double total_accessed_nodes = 0;
    
    for(size_t i = 0; i < test_number; ++i) {
        if(i % 100 == 0 && i > 0) {
            std::cout << "  Testing query " << i << "/" << test_number << std::endl;
        }
        
        auto query_data = test_query + i * vecdim;
        auto gt = test_gt + i * test_gt_dim;
        
        auto [recall, accessed_nodes] = TestSingleQuery(query_data, gt, k, ef_search, searcher);
        
        total_recall += recall;
        total_accessed_nodes += accessed_nodes;
        
        // Measure latency separately for accuracy
        auto start = std::chrono::high_resolution_clock::now();
        size_t ndc = 0;
        searcher->searchKnnWithLightweightMetrics(query_data, k, ef_search, ndc, 0.2f);
        auto end = std::chrono::high_resolution_clock::now();
        auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        total_latency_us += latency_us;
    }
    
    metrics.avg_recall = total_recall / test_number;
    metrics.avg_latency_ms = (total_latency_us / test_number) / 1000.0;
    metrics.avg_accessed_nodes = total_accessed_nodes / test_number;
    
    // Get mean out-degree from graph info
    // We'll extract it from printGraphInfo output or calculate directly
    double total_out_degree = 0;
    for(size_t i = 0; i < searcher->n; ++i) {
        total_out_degree += GET_SZ((uint8_t*)searcher->Graph[i].neighbors);
    }
    metrics.mean_out_degree = total_out_degree / searcher->n;
    
    return metrics;
}

// Get mean out-degree from graph
double GetMeanOutDegree(HNSW_NGFix<float>* searcher) {
    double total_out_degree = 0;
    for(size_t i = 0; i < searcher->n; ++i) {
        total_out_degree += GET_SZ((uint8_t*)searcher->Graph[i].neighbors);
    }
    return total_out_degree / searcher->n;
}

int main(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base_index_path")
            paths["base_index_path"] = argv[i + 1];
        if (arg == "--train_query_path")
            paths["train_query_path"] = argv[i + 1];
        if (arg == "--train_gt_path")
            paths["train_gt_path"] = argv[i + 1];
        if (arg == "--test_query_path")
            paths["test_query_path"] = argv[i + 1];
        if (arg == "--test_gt_path")
            paths["test_gt_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--full_ngfix_index_path")
            paths["full_ngfix_index_path"] = argv[i + 1];
        if (arg == "--result_dir")
            paths["result_dir"] = argv[i + 1];
        if (arg == "--K")
            paths["K"] = argv[i + 1];
        if (arg == "--num_test_queries")
            paths["num_test_queries"] = argv[i + 1];
    }
    
    std::string base_index_path = paths["base_index_path"];
    std::string train_query_path = paths["train_query_path"];
    std::string train_gt_path = paths["train_gt_path"];
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string full_ngfix_index_path = paths["full_ngfix_index_path"];
    std::string result_dir = paths.count("result_dir") ? paths["result_dir"] : "./";
    std::string metric_str = paths["metric"];
    size_t k = paths.count("K") ? std::stoi(paths["K"]) : 100;
    size_t num_test_queries = paths.count("num_test_queries") ? std::stoi(paths["num_test_queries"]) : 1000;
    size_t ef_search = 100;  // Fixed efSearch = 100
    
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Base index path: " << base_index_path << std::endl;
    std::cout << "Train query path: " << train_query_path << std::endl;
    std::cout << "Train GT path: " << train_gt_path << std::endl;
    std::cout << "Test query path: " << test_query_path << std::endl;
    std::cout << "Test GT path: " << test_gt_path << std::endl;
    std::cout << "Full NGFix index path: " << full_ngfix_index_path << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "efSearch: " << ef_search << std::endl;
    std::cout << "Num test queries: " << num_test_queries << std::endl;
    std::cout << std::endl;
    
    // Load data
    size_t train_number = 0, test_number = 0;
    size_t train_gt_dim = 0, test_gt_dim = 0, vecdim = 0;
    
    auto train_query = LoadData<float>(train_query_path, train_number, vecdim);
    auto train_gt = LoadData<int>(train_gt_path, train_number, train_gt_dim);
    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    
    std::cout << "Loaded " << train_number << " training queries" << std::endl;
    std::cout << "Loaded " << test_number << " test queries" << std::endl;
    std::cout << "Vector dimension: " << vecdim << std::endl;
    std::cout << std::endl;
    
    Metric metric;
    if(metric_str == "ip_float") {
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        metric = L2_float;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }
    
    // Load base index
    std::cout << "=== Loading Base Index ===" << std::endl;
    auto base_index = new HNSW_NGFix<float>(metric, base_index_path);
    base_index->printGraphInfo();
    std::cout << std::endl;
    
    // Helper function: Calculate recall for all training queries and sort
    auto CalculateAndSortTrainRecalls = [&](HNSW_NGFix<float>* index, const std::string& phase_name) {
        std::cout << "=== " << phase_name << ": Calculating recall for all training queries ===" << std::endl;
        std::vector<std::pair<float, size_t>> train_recalls;  // (recall, query_idx)
        
        for(size_t i = 0; i < train_number; ++i) {
            if(i % 100000 == 0 && i > 0) {
                std::cout << "  Processing training query " << i << "/" << train_number << std::endl;
            }
            
            auto query_data = train_query + i * vecdim;
            auto gt = train_gt + i * train_gt_dim;
            
            size_t ndc = 0;
            auto [results, ndc_result, lw_metrics] = index->searchKnnWithLightweightMetrics(
                query_data, k, ef_search, ndc, 0.2f);
            
            float recall = CalculateRecall(results, gt, k);
            train_recalls.push_back({recall, i});
        }
        
        // Sort by recall (low to high)
        std::sort(train_recalls.begin(), train_recalls.end());
        
        std::cout << "Sorted " << train_recalls.size() << " training queries by recall (low to high)" << std::endl;
        std::cout << "Min recall: " << train_recalls[0].first << std::endl;
        std::cout << "Max recall: " << train_recalls[train_recalls.size() - 1].first << std::endl;
        std::cout << "Median recall: " << train_recalls[train_recalls.size() / 2].first << std::endl;
        std::cout << std::endl;
        
        return train_recalls;
    };
    
    // Step 1: Initial recall calculation on base index
    auto train_recalls = CalculateAndSortTrainRecalls(base_index, "Step 1");
    
    // Step 2: Test control group (full NGFix index)
    std::cout << "=== Step 2: Testing Control Group (Full NGFix Index) ===" << std::endl;
    TestMetrics control_metrics;
    
    if(!full_ngfix_index_path.empty()) {
        auto full_ngfix_index = new HNSW_NGFix<float>(metric, full_ngfix_index_path);
        std::cout << "Full NGFix index info:" << std::endl;
        full_ngfix_index->printGraphInfo();
        
        size_t actual_test = std::min(num_test_queries, test_number);
        control_metrics = TestAllQueries(
            test_query, test_gt, actual_test, k, test_gt_dim, vecdim, 
            ef_search, full_ngfix_index);
        control_metrics.mean_out_degree = GetMeanOutDegree(full_ngfix_index);
        
        std::cout << "Control group results:" << std::endl;
        std::cout << "  Avg Recall: " << control_metrics.avg_recall << std::endl;
        std::cout << "  Avg Latency: " << control_metrics.avg_latency_ms << " ms" << std::endl;
        std::cout << "  Avg Accessed Nodes: " << control_metrics.avg_accessed_nodes << std::endl;
        std::cout << "  Mean Out-Degree: " << control_metrics.mean_out_degree << std::endl;
        std::cout << std::endl;
        
        delete full_ngfix_index;
    } else {
        std::cout << "Warning: Full NGFix index path not provided, skipping control group" << std::endl;
    }
    
    // Step 3: Iterative optimization process - test after each 1%
    std::vector<double> target_percentages = {5.0, 10.0, 15.0, 20.0, 25.0, 30.0};
    std::vector<TestMetrics> percentage_metrics;  // For final summary table
    std::vector<std::string> percentage_labels;
    
    // Store all 1% interval results
    std::vector<std::pair<double, TestMetrics>> all_results;  // (percentage, metrics)
    
    // Create working index (copy of base index)
    auto working_index = new HNSW_NGFix<float>(metric, base_index_path);
    
    // Track which queries have been used for optimization
    std::unordered_set<size_t> used_query_indices;
    size_t current_total_used = 0;
    size_t target_idx = 0;
    
    const double step_percentage = 1.0;  // Use 1% at a time
    size_t step_size = (size_t)(train_number * step_percentage / 100.0);
    double max_percentage = target_percentages.back();  // 30%
    size_t max_count = (size_t)(train_number * max_percentage / 100.0);
    
    // Open CSV file for all results
    std::string all_results_csv_path = result_dir + "/ngfix_all_results_1percent_intervals.csv";
    std::ofstream all_results_csv(all_results_csv_path);
    all_results_csv << std::fixed << std::setprecision(4);
    all_results_csv << "Percentage,Recall,Latency_ms,Accessed_Nodes,Mean_Out_Degree\n";
    
    std::cout << "=== Step 3: Iterative Optimization Process ===" << std::endl;
    std::cout << "Will optimize in steps of " << step_percentage << "% (" << step_size << " queries per step)" << std::endl;
    std::cout << "Will test after each 1% completion" << std::endl;
    std::cout << "Target percentages for summary: ";
    for(double pct : target_percentages) {
        std::cout << pct << "% ";
    }
    std::cout << std::endl;
    std::cout << "All results will be saved to: " << all_results_csv_path << std::endl;
    std::cout << std::endl;
    
    while(current_total_used < max_count) {
        // Recalculate recall on current working index
        std::cout << "\n=== Recalculating recall on current index (used " << current_total_used << " queries) ===" << std::endl;
        train_recalls = CalculateAndSortTrainRecalls(working_index, "Recalculation");
        
        // Select next 1% of queries that haven't been used yet
        size_t queries_to_add = std::min(step_size, max_count - current_total_used);
        std::vector<size_t> selected_indices;
        
        for(size_t i = 0; i < train_recalls.size() && selected_indices.size() < queries_to_add; ++i) {
            size_t query_idx = train_recalls[i].second;
            if(used_query_indices.find(query_idx) == used_query_indices.end()) {
                selected_indices.push_back(query_idx);
                used_query_indices.insert(query_idx);
            }
        }
        
        if(selected_indices.empty()) {
            std::cout << "Warning: No more queries to select!" << std::endl;
            break;
        }
        
        std::cout << "Selected " << selected_indices.size() << " new queries for optimization" << std::endl;
        std::cout << "Total queries used so far: " << used_query_indices.size() << std::endl;
        
        // Optimize with selected queries
        std::cout << "Optimizing index with selected queries..." << std::endl;
        auto opt_start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for schedule(dynamic) num_threads(32)
        for(size_t idx = 0; idx < selected_indices.size(); ++idx) {
            size_t i = selected_indices[idx];
            if(idx % 100 == 0 && idx > 0) {
                std::cout << "  Optimizing query " << idx << "/" << selected_indices.size() << std::endl;
            }
            
            auto query_data = train_query + i * vecdim;
            auto gt = train_gt + i * train_gt_dim;
            
            working_index->NGFix(query_data, gt, 100, 100);
            working_index->NGFix(query_data, gt, 10, 10);
            working_index->RFix(query_data, gt, 10);
        }
        
        auto opt_end = std::chrono::high_resolution_clock::now();
        auto opt_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(opt_end - opt_start).count();
        std::cout << "Optimization completed in " << opt_time_ms << " ms" << std::endl;
        
        current_total_used = used_query_indices.size();
        double current_percentage = (double)current_total_used / train_number * 100.0;
        
        // Test after each 1% completion
        std::cout << "\n=== Testing at " << std::fixed << std::setprecision(2) << current_percentage 
                  << "% (used " << current_total_used << " queries) ===" << std::endl;
        working_index->printGraphInfo();
        
        size_t actual_test = std::min(num_test_queries, test_number);
        TestMetrics metrics = TestAllQueries(
            test_query, test_gt, actual_test, k, test_gt_dim, vecdim, 
            ef_search, working_index);
        metrics.mean_out_degree = GetMeanOutDegree(working_index);
        
        // Store result
        all_results.push_back({current_percentage, metrics});
        all_results_csv << current_percentage << ","
                        << metrics.avg_recall << ","
                        << metrics.avg_latency_ms << ","
                        << metrics.avg_accessed_nodes << ","
                        << metrics.mean_out_degree << "\n";
        all_results_csv.flush();  // Flush to ensure data is written immediately
        
        std::cout << "Results for " << current_percentage << "%:" << std::endl;
        std::cout << "  Avg Recall: " << metrics.avg_recall << std::endl;
        std::cout << "  Avg Latency: " << metrics.avg_latency_ms << " ms" << std::endl;
        std::cout << "  Avg Accessed Nodes: " << metrics.avg_accessed_nodes << std::endl;
        std::cout << "  Mean Out-Degree: " << metrics.mean_out_degree << std::endl;
        std::cout << std::endl;
        
        // Check if we've reached a target percentage for summary
        if(target_idx < target_percentages.size()) {
            double target_pct = target_percentages[target_idx];
            if(current_percentage >= target_pct) {
                percentage_metrics.push_back(metrics);
                percentage_labels.push_back(std::to_string((int)target_pct) + "%");
                
                // Save optimized index at this checkpoint
                std::stringstream temp_index_path;
                temp_index_path << result_dir << "/temp_index_" << (int)target_pct << "pct.index";
                working_index->StoreIndex(temp_index_path.str());
                std::cout << "Checkpoint: Index saved to: " << temp_index_path.str() << std::endl;
                std::cout << std::endl;
                
                target_idx++;
            }
        }
    }
    
    all_results_csv.close();
    delete working_index;
    
    std::cout << "\n=== All 1% interval results saved to: " << all_results_csv_path << " ===" << std::endl;
    std::cout << "Total data points: " << all_results.size() << std::endl;
    std::cout << std::endl;
    
    // Step 4: Print comparison table
    std::cout << "=== Comparison Table ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(15) << "Method" 
              << std::setw(15) << "Recall" 
              << std::setw(18) << "Latency (ms)" 
              << std::setw(20) << "Accessed Nodes" 
              << std::setw(18) << "Mean Out-Degree" 
              << std::endl;
    std::cout << std::string(86, '-') << std::endl;
    
    if(!full_ngfix_index_path.empty()) {
        std::cout << std::setw(15) << "Control (100%)"
                  << std::setw(15) << control_metrics.avg_recall
                  << std::setw(18) << control_metrics.avg_latency_ms
                  << std::setw(20) << control_metrics.avg_accessed_nodes
                  << std::setw(18) << control_metrics.mean_out_degree
                  << std::endl;
    }
    
    for(size_t i = 0; i < percentage_metrics.size(); ++i) {
        std::cout << std::setw(15) << percentage_labels[i]
                  << std::setw(15) << percentage_metrics[i].avg_recall
                  << std::setw(18) << percentage_metrics[i].avg_latency_ms
                  << std::setw(20) << percentage_metrics[i].avg_accessed_nodes
                  << std::setw(18) << percentage_metrics[i].mean_out_degree
                  << std::endl;
    }
    
    // Save results to CSV
    std::string csv_path = result_dir + "/ngfix_selective_optimization_results.csv";
    std::ofstream csv_file(csv_path);
    csv_file << std::fixed << std::setprecision(4);
    csv_file << "Method,Recall,Latency_ms,Accessed_Nodes,Mean_Out_Degree\n";
    
    if(!full_ngfix_index_path.empty()) {
        csv_file << "Control (100%),"
                 << control_metrics.avg_recall << ","
                 << control_metrics.avg_latency_ms << ","
                 << control_metrics.avg_accessed_nodes << ","
                 << control_metrics.mean_out_degree << "\n";
    }
    
    for(size_t i = 0; i < percentage_metrics.size(); ++i) {
        csv_file << percentage_labels[i] << ","
                 << percentage_metrics[i].avg_recall << ","
                 << percentage_metrics[i].avg_latency_ms << ","
                 << percentage_metrics[i].avg_accessed_nodes << ","
                 << percentage_metrics[i].mean_out_degree << "\n";
    }
    csv_file.close();
    
    std::cout << std::endl;
    std::cout << "Results saved to: " << csv_path << std::endl;
    
    // Cleanup
    delete[] train_query;
    delete[] train_gt;
    delete[] test_query;
    delete[] test_gt;
    delete base_index;
    
    return 0;
}

