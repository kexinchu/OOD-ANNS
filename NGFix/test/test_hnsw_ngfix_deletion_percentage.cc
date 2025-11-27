#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <chrono>
using namespace ngfixlib;

int main(int argc, char* argv[])
{
    size_t efC = 500;
    size_t delete_start_id = 0;
    size_t delete_end_id = 0;
    std::unordered_map<std::string, std::string> paths;
    
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--index_path")
            paths["index_path"] = argv[i + 1];
        if (arg == "--lazy_delete_index_path")
            paths["lazy_delete_index_path"] = argv[i + 1];
        if (arg == "--total_delete_index_path")
            paths["total_delete_index_path"] = argv[i + 1];
        if (arg == "--result_log")
            paths["result_log"] = argv[i + 1];
        if (arg == "--efC")
            efC = std::stoi(argv[i + 1]);
        if (arg == "--delete_start_id")
            delete_start_id = std::stoull(argv[i + 1]);
        if (arg == "--delete_end_id")
            delete_end_id = std::stoull(argv[i + 1]);
    }
    
    std::string index_path = paths["index_path"];
    std::cout << "index_path: " << index_path << "\n";
    std::string lazy_delete_index_path = paths["lazy_delete_index_path"];
    std::cout << "lazy_delete_index_path: " << lazy_delete_index_path << "\n";
    std::string total_delete_index_path = paths["total_delete_index_path"];
    std::cout << "total_delete_index_path: " << total_delete_index_path << "\n";
    std::string result_log = paths["result_log"];
    std::cout << "result_log: " << result_log << "\n";
    std::string metric_str = paths["metric"];
    
    std::cout << "Delete range: [" << delete_start_id << ", " << delete_end_id << ")\n";
    std::cout << "efC: " << efC << "\n";

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

    // Open log file
    std::ofstream log_file(result_log);
    
    // Load index for lazy delete test
    auto hnsw_ngfix_lazy = new HNSW_NGFix<float>(metric, index_path);
    std::cout << "Original Index Information:\n";
    hnsw_ngfix_lazy->printGraphInfo();
    std::cout << "\n";
    
    // ========== Test Lazy Delete (DeletePointByFlag only) ==========
    std::cout << "=== Testing Lazy Delete ===\n";
    auto lazy_start = std::chrono::high_resolution_clock::now();
    
    // Perform lazy delete (only marking as deleted)
    for(size_t i = delete_start_id; i < delete_end_id; ++i) {
        if(i % 100000 == 0 && i > delete_start_id) {
            std::cout << "Lazy deleting point " << i << "\n";
        }
        hnsw_ngfix_lazy->DeletePointByFlag(i);
    }
    
    auto lazy_end = std::chrono::high_resolution_clock::now();
    auto lazy_diff = std::chrono::duration_cast<std::chrono::milliseconds>(lazy_end - lazy_start).count();
    std::cout << "Lazy delete latency: " << lazy_diff << " ms.\n\n";
    
    log_file << "Lazy delete latency: " << lazy_diff << " ms.\n";
    
    // Save lazy delete index
    hnsw_ngfix_lazy->StoreIndex(lazy_delete_index_path);
    std::cout << "Lazy delete index saved to: " << lazy_delete_index_path << "\n";
    
    delete hnsw_ngfix_lazy;
    
    // ========== Test Total Delete (DeleteAllFlagPointsByNGFix) ==========
    // Load the lazy delete index and perform total delete
    std::cout << "=== Testing Total Delete ===\n";
    auto hnsw_ngfix_total = new HNSW_NGFix<float>(metric, lazy_delete_index_path);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Perform total delete (actually remove from graph)
    hnsw_ngfix_total->DeleteAllFlagPointsByNGFix(efC, 32);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_diff = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    std::cout << "Total delete latency: " << total_diff << " ms.\n\n";
    
    log_file << "Total delete latency: " << total_diff << " ms.\n";
    
    std::cout << "Index (after total delete) Information:\n";
    hnsw_ngfix_total->printGraphInfo();
    std::cout << "\n";
    
    // Save total delete index
    hnsw_ngfix_total->StoreIndex(total_delete_index_path);
    std::cout << "Total delete index saved to: " << total_delete_index_path << "\n";
    
    log_file.close();
    
    delete hnsw_ngfix_total;
    
    return 0;
}

