#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
using namespace ngfixlib;

int main(int argc, char* argv[])
{
    int k = 0;
    int efSearch = 0;
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test_query_path")
            paths["test_query_path"] = argv[i + 1];
        if (arg == "--test_gt_path")
            paths["test_gt_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--index_path")
            paths["index_path"] = argv[i + 1];
        if (arg == "--K")
            k = std::stoi(argv[i + 1]);
        if (arg == "--efSearch")
            efSearch = std::stoi(argv[i + 1]);
    }
    
    if (k == 0 || efSearch == 0) {
        std::cerr << "Error: --K and --efSearch are required\n";
        return 1;
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string index_path = paths["index_path"];
    std::string metric_str = paths["metric"];

    size_t test_number = 0, base_number = 0;
    size_t test_gt_dim = 0, vecdim = 0;

    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    Metric metric;
    if(metric_str == "ip_float") {
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        metric = L2_float;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }

    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, index_path);
    
    // Print graph info (includes average out-degree)
    std::cout << "Graph Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    // Test with single efSearch value
    std::vector<SearchResult> results(test_number);
    for(int i = 0; i < test_number; ++i){
        auto gt = test_gt + i*test_gt_dim;
        auto res = TestSingleQuery<float>(test_query+1ll*i*vecdim, gt, k, efSearch, hnsw_ngfix);
        results[i] = {res.recall, res.ndc, res.latency, res.rderr};
    }
    
    double avg_recall = 0, avg_ndc = 0, avg_latency = 0, avg_rderr = 0;
    AllQueriesEvaluation(results, avg_recall, avg_ndc, avg_latency, avg_rderr);
    
    // Output in format easy to parse
    std::cout << "efSearch: " << efSearch << "\n";
    std::cout << "Average Recall: " << avg_recall << "\n";
    std::cout << "Average Latency: " << avg_latency << " ms\n";
    
    return 0;
}

