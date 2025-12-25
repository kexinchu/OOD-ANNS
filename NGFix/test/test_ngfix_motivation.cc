#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
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
    
    // Output: stage, efs, recall, ndc, latency_ms, rderr, start_timestamp, end_timestamp
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
    int test_efs = 200;  // Default efsearch value for testing
    size_t efC_AKNN = 1500;
    std::unordered_map<std::string, std::string> paths;
    
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test_query_path")
            paths["test_query_path"] = argv[i + 1];
        if (arg == "--test_gt_path")
            paths["test_gt_path"] = argv[i + 1];
        if (arg == "--train_query_path")
            paths["train_query_path"] = argv[i + 1];
        if (arg == "--train_gt_path")
            paths["train_gt_path"] = argv[i + 1];
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
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string train_query_path = paths["train_query_path"];
    std::string train_gt_path = paths["train_gt_path"];
    std::string index_path = paths["index_path"];
    std::string result_path = paths["result_path"];
    std::string metric_str = paths["metric"];

    std::cout << "=== NGFix Motivation Test ===\n";
    std::cout << "test_query_path: " << test_query_path << "\n";
    std::cout << "test_gt_path: " << test_gt_path << "\n";
    std::cout << "train_query_path: " << train_query_path << "\n";
    std::cout << "train_gt_path: " << train_gt_path << "\n";
    std::cout << "index_path: " << index_path << "\n";
    std::cout << "result_path: " << result_path << "\n";
    std::cout << "metric: " << metric_str << "\n";
    std::cout << "K: " << k << "\n";
    std::cout << "efs: " << test_efs << "\n";

    size_t test_number = 0, train_number = 0;
    size_t test_gt_dim = 0, train_gt_dim = 0, vecdim = 0;

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

    // Load existing index
    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, index_path);
    std::cout << "\n=== Initial Index Information ===\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    // Open result file
    std::ofstream output;
    output.open(result_path);
    output << "stage,efs,recall,ndc,latency_ms,rderr,start_timestamp,end_timestamp\n";
    output.flush();

    // Step 1: Delete 20% of addition edges
    std::cout << "\n=== Step 1: Deleting 20% of addition edges ===\n";
    auto delete_start = getCurrentTimestamp();
    hnsw_ngfix->PartialRemoveEdges(0.2f);
    auto delete_end = getCurrentTimestamp();
    std::cout << "Deletion completed. Start: " << delete_start << ", End: " << delete_end << "\n";
    
    std::cout << "\n=== Index Information After Deletion ===\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    // Step 2: Test with 10K test queries after deletion
    std::cout << "\n=== Step 2: Testing with 10K queries after deletion ===\n";
    TestQueriesAtEfs(output, test_query, test_gt, test_number, k, test_gt_dim, 
                     vecdim, hnsw_ngfix, test_efs, "after_deletion_20pct");

    // Step 3: Rebuild index incrementally with 10M train_query
    std::cout << "\n=== Step 3: Rebuilding index incrementally ===\n";
    
    auto train_query_in = getVectorsHead(train_query_path, train_number, vecdim);
    std::ifstream* train_gt_in = nullptr;
    bool use_train_gt = !train_gt_path.empty();
    
    if (use_train_gt) {
        size_t dummy_n, dummy_d;
        train_gt_in = new std::ifstream();
        train_gt_in->open(train_gt_path, std::ios::in | std::ios::binary);
        train_gt_in->read((char*)&dummy_n, 4);
        train_gt_in->read((char*)&dummy_d, 4);
        train_gt_dim = dummy_d;
        std::cout << "Loaded train_gt: " << dummy_n << " vectors, dimension: " << train_gt_dim << "\n";
    }
    
    std::cout << "Total train queries: " << train_number << "\n";
    std::cout << "Will test at: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%\n";
    
    size_t checkpoint_interval = train_number / 10;  // 10% intervals
    auto rebuild_start = std::chrono::high_resolution_clock::now();
    
    // Process training queries in batches
    // Note: We need to read sequentially to maintain order, so we process sequentially
    // but use parallel processing within NGFix operations
    for(int checkpoint_num = 1; checkpoint_num <= 10; ++checkpoint_num) {
        size_t batch_start = (checkpoint_num - 1) * checkpoint_interval;
        size_t batch_end = (checkpoint_num == 10) ? train_number : checkpoint_num * checkpoint_interval;
        size_t batch_size = batch_end - batch_start;
        
        std::cout << "\n=== Processing batch " << checkpoint_num << " (" 
                  << batch_start << " to " << batch_end << ", " << batch_size << " queries) ===\n";
        
        // Process this batch sequentially to maintain file reading order
        for(size_t i = batch_start; i < batch_end; ++i) {
            if(i % 100000 == 0) {
                std::cout << "Processing train query " << i << " / " << train_number << "\n";
            }
            
            auto train_query = getNextVector<float>(train_query_in, vecdim);
            int* gt = nullptr;
            
            if (use_train_gt && train_gt_in) {
                gt = new int[train_gt_dim];
                train_gt_in->read((char*)gt, train_gt_dim * sizeof(int));
            } else {
                gt = new int[500];
                hnsw_ngfix->AKNNGroundTruth(train_query, gt, 500, efC_AKNN);
            }
            
            hnsw_ngfix->NGFix(train_query, gt, 100, 100);
            hnsw_ngfix->NGFix(train_query, gt, 10, 10);
            hnsw_ngfix->RFix(train_query, gt, 10);
            
            delete []train_query;
            delete []gt;
        }
        
        // Test after this batch
        int percentage = checkpoint_num * 10;
        std::cout << "\n=== Checkpoint: " << percentage << "% complete (" 
                  << batch_end << " / " << train_number << ") ===\n";
        
        std::stringstream stage_name;
        stage_name << "rebuild_" << percentage << "pct";
        TestQueriesAtEfs(output, test_query, test_gt, test_number, k, 
                       test_gt_dim, vecdim, hnsw_ngfix, test_efs, 
                       stage_name.str());
    }
    
    auto rebuild_end = std::chrono::high_resolution_clock::now();
    auto rebuild_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
        rebuild_end - rebuild_start).count();
    
    std::cout << "\n=== Rebuild completed ===\n";
    std::cout << "Total rebuild time: " << rebuild_diff << " ms\n";
    
    std::cout << "\n=== Final Index Information ===\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";
    
    if (train_gt_in) {
        train_gt_in->close();
        delete train_gt_in;
    }
    
    output.close();
    delete []test_query;
    delete []test_gt;
    
    std::cout << "\n=== Test completed. Results saved to: " << result_path << " ===\n";
    
    return 0;
}

