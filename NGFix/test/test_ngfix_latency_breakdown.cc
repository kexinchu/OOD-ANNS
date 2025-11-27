#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <bitset>
#include <unordered_map>
#include <algorithm>

using namespace ngfixlib;

struct LatencyStats {
    double eh_matrix_time_ms = 0.0;
    double prepare_bitset_time_ms = 0.0;
    double get_defects_edges_time_ms = 0.0;
    double add_edges_time_ms = 0.0;
    double other_logic_time_ms = 0.0;
    double total_time_ms = 0.0;
    size_t eh_matrix_count = 0;
    size_t prepare_bitset_count = 0;
    size_t get_defects_edges_count = 0;
    size_t add_edges_count = 0;
    size_t other_logic_count = 0;
    size_t total_count = 0;
};

int main(int argc, char* argv[])
{
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--base_graph_path")
            paths["base_graph_path"] = argv[i + 1];
        if (arg == "--train_gt_path")
            paths["train_gt_path"] = argv[i + 1];
        if (arg == "--metric")
            paths["metric"] = argv[i + 1];
        if (arg == "--num_queries")
            paths["num_queries"] = argv[i + 1];
        if (arg == "--output_path")
            paths["output_path"] = argv[i + 1];
    }

    std::string base_index_path = paths["base_graph_path"];
    std::cout << "base_graph_path: " << base_index_path << "\n";
    std::string train_gt_path = paths["train_gt_path"];
    std::cout << "train_gt_path: " << train_gt_path << "\n";
    std::string metric_str = paths["metric"];
    std::string num_queries_str = paths["num_queries"];
    std::string output_path = paths["output_path"];
    
    size_t num_queries = 1000; // default 1k queries
    if (!num_queries_str.empty()) {
        num_queries = std::stoul(num_queries_str);
    }
    if (output_path.empty()) {
        output_path = "ngfix_latency_breakdown_results.json";
    }

    Metric metric;
    if(metric_str == "ip_float") {
        std::cout << "metric: ip_float\n";
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        std::cout << "metric: l2_float\n";
        metric = L2_float;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }

    // Load base graph
    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, base_index_path);
    std::cout << "HNSW Base Graph Information:\n";
    hnsw_ngfix->printGraphInfo();
    std::cout << "\n";

    // Load train.gt.bin data
    size_t train_gt_number = 0;
    size_t train_gt_dim = 0;
    auto train_gt_in = getVectorsHead(train_gt_path, train_gt_number, train_gt_dim);
    std::cout << "Loaded train.gt.bin: " << train_gt_number << " queries, dimension: " << train_gt_dim << "\n";
    
    // Limit to num_queries
    size_t actual_num_queries = std::min(num_queries, train_gt_number);
    std::cout << "Using " << actual_num_queries << " queries for testing\n\n";

    // Generate dummy queries (we only need gt, not actual query vectors for this test)
    // But we still need query vectors for NGFix, so we'll use the first vector from base graph
    size_t vecdim = hnsw_ngfix->dim;
    float* dummy_query = new float[vecdim];
    memset(dummy_query, 0, sizeof(float) * vecdim);

    LatencyStats stats;
    std::vector<double> eh_times;
    std::vector<double> prepare_bitset_times;
    std::vector<double> get_defects_edges_times;
    std::vector<double> add_edges_times;
    std::vector<double> other_times;
    std::vector<double> total_times;

    std::cout << "Starting latency breakdown test...\n";
    std::cout << "Testing " << actual_num_queries << " queries\n\n";

    for(size_t i = 0; i < actual_num_queries; ++i) {
        if(i % 100 == 0 && i > 0) {
            std::cout << "Processed " << i << " / " << actual_num_queries << " queries\n";
        }

        // Load ground truth
        auto train_gt = getNextVector<int>(train_gt_in, train_gt_dim);
        
        // Use first Nq elements as ground truth
        size_t Nq = std::min((size_t)100, train_gt_dim);
        size_t Kh = 100;
        size_t S = std::min(MAX_S, 2 * Nq);
        
        // Create a subset of gt for testing
        int* gt_subset = new int[S];
        for(size_t j = 0; j < S && j < train_gt_dim; ++j) {
            gt_subset[j] = train_gt[j];
        }

        // Measure EH matrix calculation latency
        auto start_eh = std::chrono::high_resolution_clock::now();
        auto H = hnsw_ngfix->CalculateHardness(gt_subset, Nq, Kh, S);
        auto end_eh = std::chrono::high_resolution_clock::now();
        auto eh_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_eh - start_eh).count();
        double eh_time_ms = eh_duration / 1000.0;
        stats.eh_matrix_time_ms += eh_time_ms;
        stats.eh_matrix_count++;
        eh_times.push_back(eh_time_ms);

        // Measure other logic latency - break down into detailed steps
        
        // Step 1: Prepare bitset f based on H
        auto start_prepare = std::chrono::high_resolution_clock::now();
        std::bitset<MAX_Nq> f[Nq];
        for(int j = 0; j < Nq; ++j) {
            for(int k = 0; k < Nq; ++k) {
                f[j][k] = (H[j][k] <= Kh) ? 1 : 0;
            }
        }
        auto end_prepare = std::chrono::high_resolution_clock::now();
        auto prepare_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_prepare - start_prepare).count();
        double prepare_time_ms = prepare_duration / 1000.0;
        stats.prepare_bitset_time_ms += prepare_time_ms;
        stats.prepare_bitset_count++;
        prepare_bitset_times.push_back(prepare_time_ms);
        
        // Step 2: getDefectsFixingEdges logic
        auto start_get_defects = std::chrono::high_resolution_clock::now();
        std::unordered_map<id_t, std::vector<std::pair<id_t, uint16_t> > > new_edges;
        
        // Calculate distances and prepare candidate edges
        std::vector<std::pair<float, std::pair<int,int> > > vs;
        for(int j = 0; j < Nq; ++j){
            for(int k = 0; k < Nq; ++k){
                if(f[j][k] == 1) {continue;}
                int u = gt_subset[j];
                int v = gt_subset[k]; 
                float d = hnsw_ngfix->getDist(u, v);
                vs.push_back({d,{j,k}});
            }
        }
        std::sort(vs.begin(), vs.end());

        // Select edges to add
        for(auto [d, e] : vs){
            int s = e.first;
            int t = e.second;
            if(f[s][t] == 1) {continue;}

            int u = gt_subset[s];
            int v = gt_subset[t];

            new_edges[u].push_back({v, H[s][t]});

            f[s][t] = 1;
            for(int j = 0; j < Nq; ++j){
                if(f[j][s]){
                    f[j] |= f[t];
                }
            }
        }
        auto end_get_defects = std::chrono::high_resolution_clock::now();
        auto get_defects_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_get_defects - start_get_defects).count();
        double get_defects_time_ms = get_defects_duration / 1000.0;
        stats.get_defects_edges_time_ms += get_defects_time_ms;
        stats.get_defects_edges_count++;
        get_defects_edges_times.push_back(get_defects_time_ms);
        
        // Step 3: Add edges to graph
        auto start_add_edges = std::chrono::high_resolution_clock::now();
        for(auto [u, vs] : new_edges) {
            std::unique_lock <std::shared_mutex> lock(hnsw_ngfix->getNodeLock(u));
            for(auto [v, eh] : vs) {
                hnsw_ngfix->Graph[u].add_ngfix_neighbors(v, eh, hnsw_ngfix->getMEX());
            }
        }
        auto end_add_edges = std::chrono::high_resolution_clock::now();
        auto add_edges_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_add_edges - start_add_edges).count();
        double add_edges_time_ms = add_edges_duration / 1000.0;
        stats.add_edges_time_ms += add_edges_time_ms;
        stats.add_edges_count++;
        add_edges_times.push_back(add_edges_time_ms);
        
        // Total other logic time
        double other_time_ms = prepare_time_ms + get_defects_time_ms + add_edges_time_ms;
        stats.other_logic_time_ms += other_time_ms;
        stats.other_logic_count++;
        other_times.push_back(other_time_ms);

        // Total time
        double total_time_ms = eh_time_ms + other_time_ms;
        stats.total_time_ms += total_time_ms;
        stats.total_count++;
        total_times.push_back(total_time_ms);

        delete[] train_gt;
        delete[] gt_subset;
    }

    // Calculate statistics
    double avg_eh_time = stats.eh_matrix_time_ms / stats.eh_matrix_count;
    double avg_prepare_bitset_time = stats.prepare_bitset_time_ms / stats.prepare_bitset_count;
    double avg_get_defects_time = stats.get_defects_edges_time_ms / stats.get_defects_edges_count;
    double avg_add_edges_time = stats.add_edges_time_ms / stats.add_edges_count;
    double avg_other_time = stats.other_logic_time_ms / stats.other_logic_count;
    double avg_total_time = stats.total_time_ms / stats.total_count;

    // Calculate percentiles
    std::sort(eh_times.begin(), eh_times.end());
    std::sort(prepare_bitset_times.begin(), prepare_bitset_times.end());
    std::sort(get_defects_edges_times.begin(), get_defects_edges_times.end());
    std::sort(add_edges_times.begin(), add_edges_times.end());
    std::sort(other_times.begin(), other_times.end());
    std::sort(total_times.begin(), total_times.end());

    auto percentile = [](const std::vector<double>& sorted, double p) -> double {
        if(sorted.empty()) return 0.0;
        size_t idx = (size_t)(p * sorted.size());
        if(idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted[idx];
    };

    double p50_eh = percentile(eh_times, 0.50);
    double p95_eh = percentile(eh_times, 0.95);
    double p99_eh = percentile(eh_times, 0.99);
    
    double p50_prepare = percentile(prepare_bitset_times, 0.50);
    double p95_prepare = percentile(prepare_bitset_times, 0.95);
    double p99_prepare = percentile(prepare_bitset_times, 0.99);
    
    double p50_get_defects = percentile(get_defects_edges_times, 0.50);
    double p95_get_defects = percentile(get_defects_edges_times, 0.95);
    double p99_get_defects = percentile(get_defects_edges_times, 0.99);
    
    double p50_add_edges = percentile(add_edges_times, 0.50);
    double p95_add_edges = percentile(add_edges_times, 0.95);
    double p99_add_edges = percentile(add_edges_times, 0.99);
    
    double p50_other = percentile(other_times, 0.50);
    double p95_other = percentile(other_times, 0.95);
    double p99_other = percentile(other_times, 0.99);
    
    double p50_total = percentile(total_times, 0.50);
    double p95_total = percentile(total_times, 0.95);
    double p99_total = percentile(total_times, 0.99);

    // Calculate standard deviation
    auto calc_std = [](const std::vector<double>& times, double mean) -> double {
        double sum_sq_diff = 0.0;
        for(double t : times) {
            sum_sq_diff += (t - mean) * (t - mean);
        }
        return std::sqrt(sum_sq_diff / times.size());
    };

    double std_eh = calc_std(eh_times, avg_eh_time);
    double std_prepare = calc_std(prepare_bitset_times, avg_prepare_bitset_time);
    double std_get_defects = calc_std(get_defects_edges_times, avg_get_defects_time);
    double std_add_edges = calc_std(add_edges_times, avg_add_edges_time);
    double std_other = calc_std(other_times, avg_other_time);
    double std_total = calc_std(total_times, avg_total_time);

    // Print results
    std::cout << "\n========================================\n";
    std::cout << "Latency Breakdown Results\n";
    std::cout << "========================================\n";
    std::cout << "Total queries tested: " << actual_num_queries << "\n\n";
    
    std::cout << "EH Matrix Calculation:\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(3) << avg_eh_time << " ms\n";
    std::cout << "  P50:     " << p50_eh << " ms\n";
    std::cout << "  P95:     " << p95_eh << " ms\n";
    std::cout << "  P99:     " << p99_eh << " ms\n";
    std::cout << "  Std Dev: " << std_eh << " ms\n";
    std::cout << "  Total:   " << stats.eh_matrix_time_ms << " ms\n\n";

    std::cout << "Other Logic Breakdown:\n";
    std::cout << "  Total Other Logic:\n";
    std::cout << "    Average: " << avg_other_time << " ms\n";
    std::cout << "    P50:     " << p50_other << " ms\n";
    std::cout << "    P95:     " << p95_other << " ms\n";
    std::cout << "    P99:     " << p99_other << " ms\n";
    std::cout << "    Std Dev: " << std_other << " ms\n";
    std::cout << "    Total:   " << stats.other_logic_time_ms << " ms\n\n";
    
    std::cout << "  1. Prepare Bitset (f matrix initialization):\n";
    std::cout << "     Average: " << avg_prepare_bitset_time << " ms\n";
    std::cout << "     P50:     " << p50_prepare << " ms\n";
    std::cout << "     P95:     " << p95_prepare << " ms\n";
    std::cout << "     P99:     " << p99_prepare << " ms\n";
    std::cout << "     Std Dev: " << std_prepare << " ms\n";
    std::cout << "     Total:   " << stats.prepare_bitset_time_ms << " ms\n";
    double prepare_percentage = (avg_prepare_bitset_time / avg_other_time) * 100.0;
    std::cout << "     Percentage of Other Logic: " << std::fixed << std::setprecision(2) << prepare_percentage << "%\n\n";
    
    std::cout << "  2. getDefectsFixingEdges (distance calc + sort + edge selection):\n";
    std::cout << "     Average: " << avg_get_defects_time << " ms\n";
    std::cout << "     P50:     " << p50_get_defects << " ms\n";
    std::cout << "     P95:     " << p95_get_defects << " ms\n";
    std::cout << "     P99:     " << p99_get_defects << " ms\n";
    std::cout << "     Std Dev: " << std_get_defects << " ms\n";
    std::cout << "     Total:   " << stats.get_defects_edges_time_ms << " ms\n";
    double get_defects_percentage = (avg_get_defects_time / avg_other_time) * 100.0;
    std::cout << "     Percentage of Other Logic: " << get_defects_percentage << "%\n\n";
    
    std::cout << "  3. Add Edges to Graph (lock + add_ngfix_neighbors):\n";
    std::cout << "     Average: " << avg_add_edges_time << " ms\n";
    std::cout << "     P50:     " << p50_add_edges << " ms\n";
    std::cout << "     P95:     " << p95_add_edges << " ms\n";
    std::cout << "     P99:     " << p99_add_edges << " ms\n";
    std::cout << "     Std Dev: " << std_add_edges << " ms\n";
    std::cout << "     Total:   " << stats.add_edges_time_ms << " ms\n";
    double add_edges_percentage = (avg_add_edges_time / avg_other_time) * 100.0;
    std::cout << "     Percentage of Other Logic: " << add_edges_percentage << "%\n\n";

    std::cout << "Total NGFix:\n";
    std::cout << "  Average: " << avg_total_time << " ms\n";
    std::cout << "  P50:     " << p50_total << " ms\n";
    std::cout << "  P95:     " << p95_total << " ms\n";
    std::cout << "  P99:     " << p99_total << " ms\n";
    std::cout << "  Std Dev: " << std_total << " ms\n";
    std::cout << "  Total:   " << stats.total_time_ms << " ms\n\n";

    double eh_percentage = (avg_eh_time / avg_total_time) * 100.0;
    double other_percentage = (avg_other_time / avg_total_time) * 100.0;
    double prepare_percentage_total = (avg_prepare_bitset_time / avg_total_time) * 100.0;
    double get_defects_percentage_total = (avg_get_defects_time / avg_total_time) * 100.0;
    double add_edges_percentage_total = (avg_add_edges_time / avg_total_time) * 100.0;
    
    std::cout << "Percentage Breakdown (of Total NGFix):\n";
    std::cout << "  EH Matrix: " << std::fixed << std::setprecision(2) << eh_percentage << "%\n";
    std::cout << "  Other Logic (Total): " << other_percentage << "%\n";
    std::cout << "    - Prepare Bitset: " << prepare_percentage_total << "%\n";
    std::cout << "    - getDefectsFixingEdges: " << get_defects_percentage_total << "%\n";
    std::cout << "    - Add Edges: " << add_edges_percentage_total << "%\n";
    std::cout << "========================================\n\n";

    // Write JSON results
    std::ofstream output(output_path);
    output << std::fixed << std::setprecision(6);
    output << "{\n";
    output << "  \"summary\": {\n";
    output << "    \"total_queries\": " << actual_num_queries << ",\n";
    output << "    \"base_graph_size\": " << hnsw_ngfix->n << ",\n";
    output << "    \"dimension\": " << vecdim << ",\n";
    output << "    \"eh_matrix\": {\n";
    output << "      \"avg_latency_ms\": " << avg_eh_time << ",\n";
    output << "      \"p50_latency_ms\": " << p50_eh << ",\n";
    output << "      \"p95_latency_ms\": " << p95_eh << ",\n";
    output << "      \"p99_latency_ms\": " << p99_eh << ",\n";
    output << "      \"std_dev_ms\": " << std_eh << ",\n";
    output << "      \"total_time_ms\": " << stats.eh_matrix_time_ms << ",\n";
    output << "      \"percentage\": " << eh_percentage << "\n";
    output << "    },\n";
    output << "    \"other_logic\": {\n";
    output << "      \"avg_latency_ms\": " << avg_other_time << ",\n";
    output << "      \"p50_latency_ms\": " << p50_other << ",\n";
    output << "      \"p95_latency_ms\": " << p95_other << ",\n";
    output << "      \"p99_latency_ms\": " << p99_other << ",\n";
    output << "      \"std_dev_ms\": " << std_other << ",\n";
    output << "      \"total_time_ms\": " << stats.other_logic_time_ms << ",\n";
    output << "      \"percentage\": " << other_percentage << ",\n";
    output << "      \"breakdown\": {\n";
    output << "        \"prepare_bitset\": {\n";
    output << "          \"avg_latency_ms\": " << avg_prepare_bitset_time << ",\n";
    output << "          \"p50_latency_ms\": " << p50_prepare << ",\n";
    output << "          \"p95_latency_ms\": " << p95_prepare << ",\n";
    output << "          \"p99_latency_ms\": " << p99_prepare << ",\n";
    output << "          \"std_dev_ms\": " << std_prepare << ",\n";
    output << "          \"total_time_ms\": " << stats.prepare_bitset_time_ms << ",\n";
    output << "          \"percentage_of_other\": " << prepare_percentage << ",\n";
    output << "          \"percentage_of_total\": " << prepare_percentage_total << "\n";
    output << "        },\n";
    output << "        \"get_defects_edges\": {\n";
    output << "          \"avg_latency_ms\": " << avg_get_defects_time << ",\n";
    output << "          \"p50_latency_ms\": " << p50_get_defects << ",\n";
    output << "          \"p95_latency_ms\": " << p95_get_defects << ",\n";
    output << "          \"p99_latency_ms\": " << p99_get_defects << ",\n";
    output << "          \"std_dev_ms\": " << std_get_defects << ",\n";
    output << "          \"total_time_ms\": " << stats.get_defects_edges_time_ms << ",\n";
    output << "          \"percentage_of_other\": " << get_defects_percentage << ",\n";
    output << "          \"percentage_of_total\": " << get_defects_percentage_total << "\n";
    output << "        },\n";
    output << "        \"add_edges\": {\n";
    output << "          \"avg_latency_ms\": " << avg_add_edges_time << ",\n";
    output << "          \"p50_latency_ms\": " << p50_add_edges << ",\n";
    output << "          \"p95_latency_ms\": " << p95_add_edges << ",\n";
    output << "          \"p99_latency_ms\": " << p99_add_edges << ",\n";
    output << "          \"std_dev_ms\": " << std_add_edges << ",\n";
    output << "          \"total_time_ms\": " << stats.add_edges_time_ms << ",\n";
    output << "          \"percentage_of_other\": " << add_edges_percentage << ",\n";
    output << "          \"percentage_of_total\": " << add_edges_percentage_total << "\n";
    output << "        }\n";
    output << "      }\n";
    output << "    },\n";
    output << "    \"total_ngfix\": {\n";
    output << "      \"avg_latency_ms\": " << avg_total_time << ",\n";
    output << "      \"p50_latency_ms\": " << p50_total << ",\n";
    output << "      \"p95_latency_ms\": " << p95_total << ",\n";
    output << "      \"p99_latency_ms\": " << p99_total << ",\n";
    output << "      \"std_dev_ms\": " << std_total << ",\n";
    output << "      \"total_time_ms\": " << stats.total_time_ms << "\n";
    output << "    }\n";
    output << "  },\n";
    output << "  \"detailed_results\": [\n";
    for(size_t i = 0; i < actual_num_queries && i < 100; ++i) { // Limit to first 100 for JSON size
        output << "    {\n";
        output << "      \"query_id\": " << i << ",\n";
        output << "      \"eh_matrix_time_ms\": " << eh_times[i] << ",\n";
        output << "      \"prepare_bitset_time_ms\": " << prepare_bitset_times[i] << ",\n";
        output << "      \"get_defects_edges_time_ms\": " << get_defects_edges_times[i] << ",\n";
        output << "      \"add_edges_time_ms\": " << add_edges_times[i] << ",\n";
        output << "      \"other_logic_time_ms\": " << other_times[i] << ",\n";
        output << "      \"total_time_ms\": " << total_times[i] << "\n";
        output << "    }";
        if(i < actual_num_queries - 1 && i < 99) {
            output << ",";
        }
        output << "\n";
    }
    output << "  ]\n";
    output << "}\n";
    output.close();

    std::cout << "Results saved to: " << output_path << "\n";

    delete[] dummy_query;
    delete hnsw_ngfix;
    return 0;
}

