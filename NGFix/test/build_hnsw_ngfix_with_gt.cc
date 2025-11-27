#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <mutex>
using namespace ngfixlib;

static std::mutex gt_loader_lock;

int main(int argc, char* argv[])
{
    size_t efC_AKNN = 1500;  // Default value
    bool use_train_gt = false;
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--train_query_path")
            paths["train_query_path"] = argv[i + 1];
        if (arg == "--train_gt_path")
            paths["train_gt_path"] = argv[i + 1];
        if (arg == "--base_graph_path")
            paths["base_graph_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--result_index_path")
            paths["result_index_path"] = argv[i + 1];
        if (arg == "--efC_AKNN")
            efC_AKNN = std::stoi(argv[i + 1]);
    }

    std::string train_query_path = paths["train_query_path"];
    std::cout<<"train_query_path: "<<train_query_path<<"\n";
    
    std::string train_gt_path = paths["train_gt_path"];
    if (!train_gt_path.empty()) {
        use_train_gt = true;
        std::cout<<"train_gt_path: "<<train_gt_path<<"\n";
    }
    
    std::string base_index_path = paths["base_graph_path"];
    std::cout<<"base_graph_path: "<<base_index_path<<"\n";
    std::string result_index_path = paths["result_index_path"];
    std::cout<<"result_index_path: "<<result_index_path<<"\n";
    std::string metric_str = paths["metric"];
    std::cout<<"efC_AKNN: "<<efC_AKNN<<"\n";
    std::cout<<"use_train_gt: "<<(use_train_gt ? "yes" : "no (will compute AKNN)")<<"\n";

    size_t train_number = 0;
    size_t train_gt_dim = 0, vecdim = 0;

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

    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, base_index_path);

    std::cout << "HNSW Bottom Layer Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    auto train_query_in = getVectorsHead(train_query_path, train_number, vecdim);
    std::ifstream* train_gt_in = nullptr;
    if (use_train_gt) {
        size_t dummy_n, dummy_d;
        train_gt_in = new std::ifstream();
        train_gt_in->open(train_gt_path, std::ios::in | std::ios::binary);
        train_gt_in->read((char*)&dummy_n, 4);
        train_gt_in->read((char*)&dummy_d, 4);
        train_gt_dim = dummy_d;
        std::cout << "Loaded train_gt: " << dummy_n << " vectors, dimension: " << train_gt_dim << "\n";
        if (dummy_n != train_number) {
            std::cerr << "Warning: train_query and train_gt have different counts: " 
                      << train_number << " vs " << dummy_n << "\n";
        }
    }

    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(int i = 0; i < train_number; ++i) {
        if(i % 100000 == 0) {
            std::cout <<"train queries "<< i <<"\n";
        }
        auto train_query = getNextVector<float>(train_query_in, vecdim);
        int* gt = nullptr;
        
        if (use_train_gt && train_gt_in) {
            // Use provided train_gt
            gt = new int[train_gt_dim];
            gt_loader_lock.lock();
            train_gt_in->read((char*)gt, train_gt_dim * sizeof(int));
            gt_loader_lock.unlock();
        } else {
            // Compute AKNN ground truth
            gt = new int[500];
            hnsw_ngfix->AKNNGroundTruth(train_query, gt, 500, efC_AKNN);
        }
        
        hnsw_ngfix->NGFix(train_query, gt, 100, 100);
        hnsw_ngfix->NGFix(train_query, gt, 10, 10);
        hnsw_ngfix->RFix(train_query, gt, 10);
        delete []train_query;
        delete []gt; 
    }
    
    if (train_gt_in) {
        train_gt_in->close();
        delete train_gt_in;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "NGFix latency: " << diff << " ms.\n\n";

    std::cout << "HNSW_NGFix Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    hnsw_ngfix->StoreIndex(result_index_path);
    return 0;
}

