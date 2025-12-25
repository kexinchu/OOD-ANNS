#include "tools/data_loader.h"
#include "ourslib/graph/hnsw_ours.h"
#include <iostream>
#include <iomanip>
#include <unordered_set>

using namespace ours;

int main(int argc, char* argv[]) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <index_path> <query_path> <gt_path>" << std::endl;
        return 1;
    }
    
    std::string index_path = argv[1];
    std::string query_path = argv[2];
    std::string gt_path = argv[3];
    
    // Load data
    size_t query_number = 0, query_dim = 0;
    size_t gt_number = 0, gt_dim = 0;
    
    auto queries = LoadData<float>(query_path, query_number, query_dim);
    auto gt = LoadData<int>(gt_path, gt_number, gt_dim);
    
    std::cout << "=== Data Info ===" << std::endl;
    std::cout << "Queries: " << query_number << " x " << query_dim << std::endl;
    std::cout << "GT: " << gt_number << " x " << gt_dim << std::endl;
    std::cout << std::endl;
    
    // Load index
    std::cout << "=== Loading Index ===" << std::endl;
    auto index = new HNSW_Ours<float>(IP_float, index_path, false);
    index->printGraphInfo();
    std::cout << "Index dimension: " << index->dim << std::endl;
    std::cout << "Index n: " << index->n.load() << std::endl;
    std::cout << "Index max_elements: " << index->max_elements << std::endl;
    std::cout << std::endl;
    
    // Test first few queries
    size_t k = 100;
    size_t ef_search = 1000;
    size_t test_queries = std::min(query_number, (size_t)10);
    
    std::cout << "=== Testing First " << test_queries << " Queries ===" << std::endl;
    
    for(size_t i = 0; i < test_queries; ++i) {
        float* query_data = queries + i * query_dim;
        int* gt_data = gt + i * gt_dim;
        
        // Search
        size_t ndc = 0;
        auto results = index->searchKnn(query_data, k, ef_search, ndc);
        
        // Check GT data
        std::cout << "\nQuery " << i << ":" << std::endl;
        std::cout << "  Results count: " << results.size() << std::endl;
        std::cout << "  NDC: " << ndc << std::endl;
        std::cout << "  First 5 GT IDs: ";
        for(size_t j = 0; j < std::min((size_t)5, gt_dim); ++j) {
            std::cout << gt_data[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "  First 5 result IDs: ";
        for(size_t j = 0; j < std::min((size_t)5, results.size()); ++j) {
            std::cout << results[j].second << " ";
        }
        std::cout << std::endl;
        
        // Check if GT IDs are valid
        bool gt_valid = true;
        for(size_t j = 0; j < std::min((size_t)10, gt_dim); ++j) {
            if(gt_data[j] >= index->max_elements) {
                std::cout << "  WARNING: GT ID " << gt_data[j] << " >= max_elements " << index->max_elements << std::endl;
                gt_valid = false;
            }
        }
        
        // Calculate recall
        std::unordered_set<id_t> gt_set;
        for(size_t j = 0; j < k && j < gt_dim; ++j) {
            gt_set.insert(gt_data[j]);
        }
        
        size_t acc = 0;
        for(const auto& p : results) {
            if(gt_set.find(p.second) != gt_set.end()) {
                acc++;
                gt_set.erase(p.second);
            }
        }
        
        float recall = (float)acc / k;
        std::cout << "  Recall: " << std::fixed << std::setprecision(4) << recall << " (" << acc << "/" << k << ")" << std::endl;
        
        // Check if any result matches GT
        if(acc == 0 && results.size() > 0) {
            std::cout << "  DEBUG: No matches found!" << std::endl;
            std::cout << "  Sample GT range: [" << gt_data[0] << ", " << gt_data[std::min((size_t)4, gt_dim-1)] << "]" << std::endl;
            std::cout << "  Sample result range: [" << results[0].second << ", " << results[std::min((size_t)4, results.size()-1)].second << "]" << std::endl;
        }
    }
    
    delete index;
    delete[] queries;
    delete[] gt;
    
    return 0;
}

