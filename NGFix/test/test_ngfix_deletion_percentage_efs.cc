#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
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
                      int efs, const std::string& stage_name) {
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
    
    // Output: deletion_percentage, efs, recall, ndc, latency_ms, rderr, start_timestamp, end_timestamp
    output << stage_name << ","
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
}

int main(int argc, char* argv[])
{
    int k = 100;
    std::vector<int> test_efs_list = {100, 200, 300, 400, 500, 1000, 1500, 2000};
    std::vector<float> deletion_percentages = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.10f, 0.15f};
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
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string index_path = paths["index_path"];
    std::string result_path = paths["result_path"];
    std::string metric_str = paths["metric"];

    std::cout << "=== NGFix Deletion Percentage Test ===\n";
    std::cout << "test_query_path: " << test_query_path << "\n";
    std::cout << "test_gt_path: " << test_gt_path << "\n";
    std::cout << "index_path: " << index_path << "\n";
    std::cout << "result_path: " << result_path << "\n";
    std::cout << "metric: " << metric_str << "\n";
    std::cout << "K: " << k << "\n";
    std::cout << "Test efSearch values: ";
    for(auto efs : test_efs_list) {
        std::cout << efs << " ";
    }
    std::cout << "\n";
    std::cout << "Test deletion percentages: ";
    for(auto pct : deletion_percentages) {
        std::cout << (pct * 100) << "% ";
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

    // Open result file
    std::ofstream output;
    output.open(result_path);
    output << "deletion_percentage,efs,recall,ndc,latency_ms,rderr,start_timestamp,end_timestamp\n";
    output.flush();

    // Load base index once
    auto hnsw_ngfix_base = new HNSW_NGFix<float>(metric, index_path);
    std::cout << "\n=== Initial Index Information ===\n";
    hnsw_ngfix_base->printGraphInfo();
    std::cout << "\n";

    // Test each deletion percentage
    for(float deletion_pct : deletion_percentages) {
        std::cout << "\n========================================\n";
        std::cout << "Testing deletion percentage: " << (deletion_pct * 100) << "%\n";
        std::cout << "========================================\n";
        
        // Create a copy of the index for this test
        // We'll load from base and apply deletion
        auto hnsw_ngfix = new HNSW_NGFix<float>(metric, index_path);
        
        // Apply deletion
        std::cout << "\n=== Deleting " << (deletion_pct * 100) << "% of addition edges ===\n";
        auto delete_start = getCurrentTimestamp();
        hnsw_ngfix->PartialRemoveEdges(deletion_pct);
        auto delete_end = getCurrentTimestamp();
        std::cout << "Deletion completed. Start: " << delete_start << ", End: " << delete_end << "\n";
        
        std::cout << "\n=== Index Information After " << (deletion_pct * 100) << "% Deletion ===\n";
        hnsw_ngfix->printGraphInfo();
        std::cout << "\n";

        // Test at each efSearch value
        for(int efs : test_efs_list) {
            std::stringstream stage_name;
            stage_name << "deletion_" << std::fixed << std::setprecision(0) << (deletion_pct * 100) << "pct";
            
            std::cout << "\n--- Testing at efSearch = " << efs << " ---\n";
            TestQueriesAtEfs(output, test_query, test_gt, test_number, k, test_gt_dim, 
                           vecdim, hnsw_ngfix, efs, stage_name.str());
        }
        
        // Clean up this index copy
        delete hnsw_ngfix;
    }
    
    output.close();
    delete []test_query;
    delete []test_gt;
    delete hnsw_ngfix_base;
    
    std::cout << "\n=== Test completed. Results saved to: " << result_path << " ===\n";
    
    return 0;
}




