#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <cmath>
using namespace ngfixlib;

int main(int argc, char* argv[])
{
    size_t efC = 500;
    size_t data_offset = 0;      // Offset in data file (where to start reading)
    size_t insert_id_start = 0;  // Starting ID in index (where to insert)
    size_t insert_count = 0;
    float noise_scale = 0.0f;
    std::unordered_map<std::string, std::string> paths;
    
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base_data_path")
            paths["base_data_path"] = argv[i + 1];
        if (arg == "--raw_index_path")
            paths["raw_index_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--result_index_path")
            paths["result_index_path"] = argv[i + 1];
        if (arg == "--efC")
            efC = std::stoi(argv[i + 1]);
        if (arg == "--data_offset")
            data_offset = std::stoull(argv[i + 1]);
        if (arg == "--insert_id_start")
            insert_id_start = std::stoull(argv[i + 1]);
        if (arg == "--insert_count")
            insert_count = std::stoull(argv[i + 1]);
        if (arg == "--noise_scale")
            noise_scale = std::stof(argv[i + 1]);
    }
    
    std::string base_path = paths["base_data_path"];
    std::cout << "base_data_path: " << base_path << "\n";
    std::string base_index_path = paths["raw_index_path"];
    std::cout << "raw_index_path: " << base_index_path << "\n";
    std::string result_index_path = paths["result_index_path"];
    std::cout << "result_index_path: " << result_index_path << "\n";
    std::string metric_str = paths["metric"];
    
    std::cout << "efC: " << efC << "\n";
    std::cout << "data_offset: " << data_offset << "\n";
    std::cout << "insert_id_start: " << insert_id_start << "\n";
    std::cout << "insert_count: " << insert_count << "\n";
    std::cout << "noise_scale: " << noise_scale << "\n";

    size_t base_number = 0;
    size_t vecdim = 0;

    auto base_data = LoadData<float>(base_path, base_number, vecdim);
    std::cout << "Loaded " << base_number << " vectors from base_data_path\n";

    // Check if we have enough data
    if (data_offset + insert_count > base_number) {
        std::cerr << "Error: Requested " << insert_count << " vectors starting from offset " 
                  << data_offset << ", but only " << base_number << " vectors available\n";
        delete[] base_data;
        return 1;
    }

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

    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, base_index_path);

    std::cout << "Raw Index Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    // Resize to accommodate the new vectors
    size_t max_id = insert_id_start + insert_count;
    hnsw_ngfix->resize(max_id);
    std::cout << "Resized index to accommodate up to ID " << max_id << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    // Initialize random number generator for noise
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> noise_dist(0.0f, noise_scale);
    
    // Insert vectors into base graph
    std::cout << "Inserting " << insert_count << " vectors:";
    std::cout << " reading from data offset " << data_offset;
    std::cout << ", inserting at index IDs " << insert_id_start << " to " << (insert_id_start + insert_count - 1);
    if(noise_scale > 0.0f) {
        std::cout << " (with noise scale=" << noise_scale << ")";
    }
    std::cout << "\n";
    
    #pragma omp parallel for schedule(dynamic) num_threads(32)
    for(size_t i = 0; i < insert_count; ++i) {
        size_t data_idx = data_offset + i;      // Index in data array
        size_t insert_id = insert_id_start + i; // ID in index
        
        if(i % 10000 == 0) {
            std::cout << "Inserting point " << i << " (data_idx=" << data_idx << ", insert_id=" << insert_id << ")\n";
        }
        
        float* data_to_insert;
        std::vector<float> noisy_data;
        
        if(noise_scale > 0.0f) {
            // Add random bias/noise to the vector
            noisy_data.resize(vecdim);
            for(size_t d = 0; d < vecdim; ++d) {
                noisy_data[d] = base_data[data_idx*vecdim + d] + noise_dist(rng);
            }
            data_to_insert = noisy_data.data();
        } else {
            // Use original data without noise
            data_to_insert = base_data + data_idx*vecdim;
        }
        
        hnsw_ngfix->InsertPoint(insert_id, efC, data_to_insert);
    }

    std::cout << "Finished inserting " << insert_count << " vectors\n";

    // No partial rebuilding or training - just insert and test
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Insertion latency: " << diff << " ms.\n\n";

    std::cout << "Index (after insertion) Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    hnsw_ngfix->StoreIndex(result_index_path);
    
    // Clean up
    delete[] base_data;
    delete hnsw_ngfix;
    
    return 0;
}

