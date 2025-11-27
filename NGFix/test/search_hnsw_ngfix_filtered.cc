#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace ngfixlib;

template<typename T>
SearchResult TestSingleQueryFiltered(T* query_data, int* gt, size_t k, size_t efs, HNSW_NGFix<T>* searcher, size_t max_valid_id) {
    size_t ndc = 0;

    const unsigned long Converter = 1000 * 1000;
    struct timeval val;
    int ret = gettimeofday(&val, NULL);

    auto aknns = searcher->searchKnn(query_data, k, efs, ndc);
    
    struct timeval newVal;
    ret = gettimeofday(&newVal, NULL);
    int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

    // Filter ground truth to only include valid IDs
    std::vector<int> valid_gt;
    for(int i = 0; i < k; ++i) {
        if(gt[i] >= 0 && gt[i] < (int)max_valid_id) {
            valid_gt.push_back(gt[i]);
        }
    }
    
    std::unordered_set<id_t> gtset;
    for(auto id : valid_gt) {
        gtset.insert(id);
    }

    int acc = 0;
    for(int i = 0; i < k; ++i) {
        if(aknns[i].second < max_valid_id) {
            if(gtset.find(aknns[i].second) != gtset.end()) {
                ++acc;
                gtset.erase(aknns[i].second);
            }
        }
    }

    float recall = (valid_gt.size() > 0) ? (float)acc / valid_gt.size() : 0.0f;
    double rderr = 0;
    
    // Calculate rderr only for valid IDs
    int valid_count = 0;
    for(int i = 0; i < k && i < (int)valid_gt.size(); ++i){
        if(aknns[i].second >= max_valid_id) continue;
        float d0 = searcher->getDist(aknns[i].second, query_data);
        if(valid_gt[i] >= max_valid_id) continue;
        float d1 = searcher->getDist(valid_gt[i], query_data);
        if(fabs(d1) < 0.00001) {continue; }
        rderr += d0/d1;
        valid_count++;
    }
    if(valid_count > 0) {
        rderr = rderr / valid_count;
    }

    return SearchResult{recall, ndc, diff, rderr};
}

template<typename T>
void TestQueriesFiltered(std::ostream& s, T* test_query, int* test_gt, size_t test_number, size_t k,
                            size_t test_gt_d, size_t vecdim, HNSW_NGFix<T>* searcher, size_t max_valid_id)
{
    //output header
    s << "efs, recall, ndc, latency, rderr\n";
    
    std::vector<int> efss;
    if(k == 100) {
        efss = std::vector<int>{100,110,120,130,140,150,160,180,200,250,300,400,500,800,1000,2000};
    } else if(k == 10) {
        efss = std::vector<int>{10,15,20,30,40,50,60,70,80,90,100,120,150,180,200,300,400,500,800,1000};
    }

    for(auto efs : efss) {
        std::cerr<<efs<<"\n";

        std::vector<SearchResult> results(test_number);
        for(int i = 0; i < test_number; ++i){
            auto gt = test_gt + i*test_gt_d;
            auto res = TestSingleQueryFiltered<T>(test_query+1ll*i*vecdim, gt, k, efs, searcher, max_valid_id);
            results[i] = {res.recall, res.ndc, res.latency, res.rderr};
        }
        double avg_recall = 0, avg_ndc = 0, avg_latency = 0, avg_rderr = 0;
        AllQueriesEvaluation(results, avg_recall, avg_ndc, avg_latency, avg_rderr);
        s << efs << ", "<< avg_recall << ", " << avg_ndc << ", " << avg_latency << ", " << avg_rderr << "\n";
    }
}

int main(int argc, char* argv[])
{
    int k = 0;
    size_t max_valid_id = 0;
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
        if (arg == "--result_path")
            paths["result_path"] = argv[i + 1];
        if (arg == "--K")
            k = std::stoi(argv[i + 1]);
        if (arg == "--max_valid_id")
            max_valid_id = std::stoull(argv[i + 1]);
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::cout<<"test_query_path: "<<test_query_path<<"\n";
    std::string test_gt_path = paths["test_gt_path"];
    std::cout<<"test_gt_path: "<<test_gt_path<<"\n";
    std::string index_path = paths["index_path"];
    std::cout<<"index_path: "<<index_path<<"\n";
    std::string result_path = paths["result_path"];
    std::cout<<"result_path: "<<result_path<<"\n";
    std::string metric_str = paths["metric"];

    size_t test_number = 0, base_number = 0;
    size_t test_gt_dim = 0, vecdim = 0;

    auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
    Metric metric;
    if(metric_str == "ip_float") {
        std::cout<<"metric ip\n";
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        std::cout<<"metric l2\n";
        metric = L2_float;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }

    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, index_path);
    hnsw_ngfix->printGraphInfo();
    
    // If max_valid_id not specified, use max_elements from index
    if(max_valid_id == 0) {
        max_valid_id = hnsw_ngfix->max_elements;
    }
    std::cout << "Max valid ID for filtering: " << max_valid_id << "\n";

    std::ofstream output;
    output.open(result_path);
    TestQueriesFiltered<float>(output, test_query, test_gt, test_number, k, test_gt_dim, vecdim, hnsw_ngfix, max_valid_id);
    output.close();

    delete[] test_query;
    delete[] test_gt;
    delete hnsw_ngfix;

    return 0;
}

