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
#include <mutex>
using namespace ngfixlib;

static std::mutex gt_loader_lock;

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

// SearchResult and TestSingleQuery are already defined in result_evaluation.h

template<typename T>
void TestQueriesAtEfs(std::ofstream& output, T* test_query, int* test_gt, 
                      size_t test_number, size_t k, size_t test_gt_d, 
                      size_t vecdim, HNSW_NGFix<T>* searcher, 
                      int efs, const std::string& stage_name) {
    // Warmup phase
    std::cout << "Warmup phase: Executing " << test_number << " queries (not counted)...\n";
    for(int i = 0; i < test_number; ++i){
        auto gt = test_gt + i*test_gt_d;
        size_t ndc_dummy = 0;
        searcher->searchKnn(test_query+1ll*i*vecdim, k, efs, ndc_dummy);
    }
    std::cout << "Warmup phase completed.\n";
    
    // Actual test phase
    std::vector<SearchResult> results(test_number);
    
    auto start_time = getCurrentTimestamp();
    auto start_chrono = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < test_number; ++i){
        auto gt = test_gt + i*test_gt_d;
        auto res = TestSingleQuery<T>(test_query+1ll*i*vecdim, gt, k, efs, searcher);
        results[i] = res;
    }
    
    auto end_chrono = std::chrono::high_resolution_clock::now();
    auto end_time = getCurrentTimestamp();
    
    double avg_recall = 0, avg_ndc = 0, avg_latency = 0, avg_rderr = 0;
    for(int i = 0; i < test_number; ++i){
        avg_recall += results[i].recall;
        avg_ndc += results[i].ndc;
        avg_rderr += results[i].rderr;
        avg_latency += results[i].latency;
    }
    avg_rderr /= test_number;
    avg_recall /= test_number;
    avg_latency /= test_number;
    avg_latency /= 1000;
    avg_ndc /= test_number;
    
    // Output: stage_name, efs, recall, ndc, latency_ms, rderr, start_timestamp, end_timestamp
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
    size_t M = 16, MEX = 48, efC = 500, efC_AKNN = 1500, efC_delete = 500;
    size_t K = 100;
    size_t base_size = 10000000;  // 10M base data
    size_t additional_size = 3000000;  // 3M additional data
    std::unordered_map<std::string, std::string> paths;
    
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base_data_path")
            paths["base_data_path"] = argv[i + 1];
        if (arg == "--additional_data_path")
            paths["additional_data_path"] = argv[i + 1];
        if (arg == "--train_query_path")
            paths["train_query_path"] = argv[i + 1];
        if (arg == "--train_gt_path")
            paths["train_gt_path"] = argv[i + 1];
        if (arg == "--test_query_path")
            paths["test_query_path"] = argv[i + 1];
        if (arg == "--test_gt_path")
            paths["test_gt_path"] = argv[i + 1];
        if (arg == "--base_index_path")
            paths["base_index_path"] = argv[i + 1];
        if (arg == "--result_index_path")
            paths["result_index_path"] = argv[i + 1];
        if (arg == "--result_csv_path")
            paths["result_csv_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--MEX")
            MEX = std::stoi(argv[i + 1]);
        if (arg == "--efC")
            efC = std::stoi(argv[i + 1]);
        if (arg == "--efC_AKNN")
            efC_AKNN = std::stoi(argv[i + 1]);
        if (arg == "--efC_delete")
            efC_delete = std::stoi(argv[i + 1]);
        if (arg == "--K")
            K = std::stoi(argv[i + 1]);
        if (arg == "--base_size")
            base_size = std::stoi(argv[i + 1]);
        if (arg == "--additional_size")
            additional_size = std::stoi(argv[i + 1]);
    }

    std::string base_data_path = paths["base_data_path"];
    std::string additional_data_path = paths["additional_data_path"];
    std::string train_query_path = paths["train_query_path"];
    std::string train_gt_path = paths["train_gt_path"];
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string base_index_path = paths["base_index_path"];
    std::string result_index_path = paths["result_index_path"];
    std::string result_csv_path = paths["result_csv_path"];
    std::string metric_str = paths["metric"];

    std::cout << "=== Comprehensive Deletion Comparison Test ===" << "\n";
    std::cout << "Base data: " << base_size << " vectors\n";
    std::cout << "Additional data: " << additional_size << " vectors\n";
    std::cout << "Base index: " << base_index_path << "\n";
    std::cout << "Result index: " << result_index_path << "\n";
    std::cout << "Result CSV: " << result_csv_path << "\n";

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

    // Load test data
    size_t test_number = 0, test_gt_dim = 0, vecdim = 0;
    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    std::cout << "Loaded " << test_number << " test queries\n";

    // Open result CSV file
    std::ofstream result_csv(result_csv_path);
    result_csv << "stage,efs,recall,ndc,latency_ms,rderr,start_time,end_time\n";

    // Step 1: Load base index with 10M data (must be built separately)
    std::cout << "\n=== Step 1: Loading base index ===\n";
    if(base_index_path.empty()) {
        throw std::runtime_error("Error: base_index_path is required");
    }
    std::ifstream index_check(base_index_path);
    if(!index_check.good()) {
        throw std::runtime_error("Error: Base index file not found: " + base_index_path + ". Please build it first.");
    }
    index_check.close();
    
    HNSW_NGFix<float>* hnsw_ngfix = new HNSW_NGFix<float>(metric, base_index_path);
    std::cout << "Base index loaded\n";
    hnsw_ngfix->printGraphInfo();

    // Step 2: Insert 3M additional data
    std::cout << "\n=== Step 2: Inserting " << additional_size << " additional vectors ===\n";
    size_t additional_number = 0, additional_vecdim = 0;
    auto additional_data = LoadData<float>(additional_data_path, additional_number, additional_vecdim);
    if(additional_vecdim != vecdim) {
        throw std::runtime_error("Vector dimension mismatch!");
    }
    
    size_t total_size = base_size + additional_size;
    hnsw_ngfix->resize(total_size);
    
    size_t insert_start_id = base_size;
    size_t actual_additional_size = std::min(additional_size, additional_number);
    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(size_t i = 0; i < actual_additional_size; ++i) {
        if(i % 100000 == 0) {
            std::cout << "Inserting additional point " << i << "\n";
        }
        size_t insert_id = insert_start_id + i;
        hnsw_ngfix->InsertPoint(insert_id, efC, additional_data + i*vecdim);
    }
    std::cout << "Inserted " << actual_additional_size << " additional vectors\n";

    // Step 3: Apply NGFix + RFix with 10M train queries
    std::cout << "\n=== Step 3: Applying NGFix + RFix with train queries ===\n";
    size_t train_number = 0, train_gt_dim = 0, train_vecdim = 0;
    auto train_query_in = getVectorsHead(train_query_path, train_number, train_vecdim);
    std::ifstream* train_gt_in = nullptr;
    bool use_train_gt = !train_gt_path.empty();
    
    if(use_train_gt) {
        train_gt_in = new std::ifstream();
        train_gt_in->open(train_gt_path, std::ios::in | std::ios::binary);
        size_t dummy_n, dummy_d;
        train_gt_in->read((char*)&dummy_n, 4);
        train_gt_in->read((char*)&dummy_d, 4);
        train_gt_dim = dummy_d;
        std::cout << "Loaded train_gt: " << dummy_n << " queries, dimension: " << train_gt_dim << "\n";
    }

    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = 0; i < train_number; ++i) {
        if(i % 100000 == 0) {
            std::cout << "Processing train query " << i << "\n";
        }
        auto train_query = getNextVector<float>(train_query_in, train_vecdim);
        int* gt = nullptr;
        
        if(use_train_gt && train_gt_in) {
            gt = new int[train_gt_dim];
            gt_loader_lock.lock();
            train_gt_in->read((char*)gt, train_gt_dim * sizeof(int));
            gt_loader_lock.unlock();
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
    
    if(train_gt_in) {
        train_gt_in->close();
        delete train_gt_in;
    }
    
    std::cout << "NGFix + RFix completed\n";
    hnsw_ngfix->printGraphInfo();

    // Step 4: Store index to SSD
    std::cout << "\n=== Step 4: Storing index to SSD ===\n";
    hnsw_ngfix->StoreIndex(result_index_path);
    std::cout << "Index stored to: " << result_index_path << "\n";

    // Step 5: Test initial recall/NDC/latency with different efSearch values
    std::cout << "\n=== Step 5: Testing initial index ===\n";
    std::vector<int> efSearch_values = {100, 200, 300, 400, 500, 1000, 2000};
    for(int efs : efSearch_values) {
        TestQueriesAtEfs(result_csv, test_query, test_gt, test_number, K, test_gt_dim, 
                        vecdim, hnsw_ngfix, efs, "initial");
    }

    // Step 6: Lazy deletion test
    std::cout << "\n=== Step 6: Testing lazy deletion ===\n";
    // Mark additional data for deletion (lazy delete)
    for(size_t i = insert_start_id; i < insert_start_id + additional_size; ++i) {
        if(i < total_size && i != hnsw_ngfix->entry_point) {
            hnsw_ngfix->DeletePointByFlag(i);
        }
    }
    std::cout << "Marked " << additional_size << " nodes for lazy deletion\n";
    
    // Test with lazy deletion (nodes are marked but not removed)
    for(int efs : efSearch_values) {
        TestQueriesAtEfs(result_csv, test_query, test_gt, test_number, K, test_gt_dim, 
                        vecdim, hnsw_ngfix, efs, "lazy_deletion");
    }

    // Step 7: Real deletion test
    std::cout << "\n=== Step 7: Testing real deletion ===\n";
    // Reload index for real deletion test
    delete hnsw_ngfix;
    hnsw_ngfix = new HNSW_NGFix<float>(metric, result_index_path);
    
    // Mark for deletion again
    for(size_t i = insert_start_id; i < insert_start_id + additional_size; ++i) {
        if(i < total_size && i != hnsw_ngfix->entry_point) {
            hnsw_ngfix->DeletePointByFlag(i);
        }
    }
    
    // Perform real deletion with repair
    std::cout << "Performing real deletion with NGFix repair...\n";
    auto delete_start = std::chrono::high_resolution_clock::now();
    hnsw_ngfix->DeleteAllFlagPointsByNGFix(efC_delete, 32);
    auto delete_end = std::chrono::high_resolution_clock::now();
    auto delete_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        delete_end - delete_start).count();
    std::cout << "Real deletion completed in " << delete_duration << " ms\n";
    hnsw_ngfix->printGraphInfo();
    
    // Test with real deletion
    for(int efs : efSearch_values) {
        TestQueriesAtEfs(result_csv, test_query, test_gt, test_number, K, test_gt_dim, 
                        vecdim, hnsw_ngfix, efs, "real_deletion");
    }

    // Cleanup
    result_csv.close();
    delete[] test_query;
    delete[] test_gt;
    delete[] additional_data;
    delete hnsw_ngfix;
    
    std::cout << "\n=== Test completed successfully ===\n";
    std::cout << "Results saved to: " << result_csv_path << "\n";
    
    return 0;
}

