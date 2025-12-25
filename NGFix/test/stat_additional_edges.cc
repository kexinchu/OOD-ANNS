#include "ngfixlib/graph/hnsw_ngfix.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <map>
#include <cmath>
using namespace ngfixlib;

int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <index_path> <metric> [output_csv_path]\n";
        std::cerr << "  metric: l2_float or ip_float\n";
        std::cerr << "  output_csv_path: optional, default to stdout\n";
        return 1;
    }

    std::string index_path = argv[1];
    std::string metric_str = argv[2];
    std::string output_path = (argc >= 4) ? argv[3] : "";

    Metric metric;
    if(metric_str == "ip_float") {
        std::cout << "metric ip\n";
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        std::cout << "metric l2\n";
        metric = L2_float;
    } else {
        std::cerr << "Error: Unsupported metric type. Use l2_float or ip_float.\n";
        return 1;
    }

    // Load index
    std::cout << "Loading index from: " << index_path << "\n";
    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, index_path);
    
    size_t n = hnsw_ngfix->n;
    std::cout << "Total nodes: " << n << "\n";

    // Collect additional edge counts for all nodes
    std::vector<uint8_t> additional_edge_counts;
    additional_edge_counts.reserve(n);

    std::cout << "Collecting additional edge counts...\n";
    for(size_t i = 0; i < n; ++i) {
        auto neighbors = hnsw_ngfix->Graph[i].get_neighbors();
        uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
        additional_edge_counts.push_back(ngfix_sz);
        
        if((i + 1) % 1000000 == 0) {
            std::cout << "Processed " << (i + 1) << " nodes...\n";
        }
    }

    std::cout << "Calculating statistics...\n";

    // Calculate statistics
    if(additional_edge_counts.empty()) {
        std::cerr << "Error: No nodes found.\n";
        return 1;
    }

    // Sort for percentile calculation
    std::vector<uint8_t> sorted_counts = additional_edge_counts;
    std::sort(sorted_counts.begin(), sorted_counts.end());

    // Calculate min, max, avg
    uint8_t min_val = sorted_counts[0];
    uint8_t max_val = sorted_counts.back();
    double sum = 0.0;
    for(auto val : additional_edge_counts) {
        sum += val;
    }
    double avg_val = sum / additional_edge_counts.size();

    // Calculate percentiles
    auto percentile = [&](double p) -> uint8_t {
        if(p < 0.0) p = 0.0;
        if(p > 1.0) p = 1.0;
        size_t idx = (size_t)std::round(p * (sorted_counts.size() - 1));
        if(idx >= sorted_counts.size()) idx = sorted_counts.size() - 1;
        return sorted_counts[idx];
    };

    uint8_t p50 = percentile(0.50);
    uint8_t p75 = percentile(0.75);
    uint8_t p90 = percentile(0.90);
    uint8_t p95 = percentile(0.95);
    uint8_t p99 = percentile(0.99);

    // Count distribution
    std::map<uint8_t, size_t> count_distribution;
    for(auto val : additional_edge_counts) {
        count_distribution[val]++;
    }

    // Output results
    std::ofstream output_file;
    std::ostream* output = &std::cout;
    
    if(!output_path.empty()) {
        output_file.open(output_path);
        if(!output_file.is_open()) {
            std::cerr << "Error: Cannot open output file: " << output_path << "\n";
            return 1;
        }
        output = &output_file;
    }

    *output << std::fixed << std::setprecision(2);
    
    *output << "\n=== Additional Edge Statistics ===\n";
    *output << "Total nodes: " << n << "\n\n";
    
    *output << "--- Basic Statistics ---\n";
    *output << "Min: " << (int)min_val << "\n";
    *output << std::setprecision(4);
    *output << "Max: " << (int)max_val << "\n";
    *output << "Avg: " << avg_val << "\n";
    *output << std::setprecision(2);
    *output << "P50: " << (int)p50 << "\n";
    *output << "P75: " << (int)p75 << "\n";
    *output << "P90: " << (int)p90 << "\n";
    *output << "P95: " << (int)p95 << "\n";
    *output << "P99: " << (int)p99 << "\n\n";

    *output << "--- Distribution by Additional Edge Count ---\n";
    *output << "Additional_Edges,Node_Count,Percentage\n";
    
    for(auto& [edge_count, node_count] : count_distribution) {
        double percentage = 100.0 * node_count / n;
        *output << (int)edge_count << "," << node_count << "," 
                << std::setprecision(4) << percentage << "%\n";
    }

    // Also output a summary CSV
    if(!output_path.empty()) {
        std::string summary_path = output_path;
        size_t dot_pos = summary_path.find_last_of('.');
        if(dot_pos != std::string::npos) {
            summary_path = summary_path.substr(0, dot_pos) + "_summary.csv";
        } else {
            summary_path += "_summary.csv";
        }
        
        std::ofstream summary_file(summary_path);
        if(summary_file.is_open()) {
            summary_file << "Metric,Min,Max,Avg,P50,P75,P90,P95,P99,Total_Nodes\n";
            summary_file << metric_str << ","
                        << (int)min_val << ","
                        << (int)max_val << ","
                        << std::setprecision(4) << avg_val << ","
                        << (int)p50 << ","
                        << (int)p75 << ","
                        << (int)p90 << ","
                        << (int)p95 << ","
                        << (int)p99 << ","
                        << n << "\n";
            summary_file.close();
            std::cout << "\nSummary saved to: " << summary_path << "\n";
        }
    }

    if(output_file.is_open()) {
        output_file.close();
        std::cout << "\nDetailed results saved to: " << output_path << "\n";
    }

    delete hnsw_ngfix;
    return 0;
}

