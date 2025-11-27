#include "ngfixlib/graph/hnsw_ngfix.h"
#include "ngfixlib/graph/node.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <queue>
#include <chrono>

using namespace ngfixlib;

// Node grouping: group nodes that are reachable within 2 hops
std::vector<std::vector<int> > GroupNodesByReachability(
    HNSW_NGFix<float>* searcher, 
    int* node_ids, 
    size_t M,
    int max_hops = 2) {
    
    std::vector<std::vector<int> > groups;
    std::unordered_set<int> visited;
    
    // Build adjacency list for the subgraph
    std::unordered_map<int, std::vector<int> > adj_list;
    for(size_t i = 0; i < M; ++i) {
        int u = node_ids[i];
        if(u >= searcher->max_elements) continue;
        
        try {
            auto [neighbors, sz, st] = searcher->getNeighbors(u);
            for(int j = st; j < st + sz; ++j) {
                int v = neighbors[j];
                if(v >= searcher->max_elements) continue;
                
                // Check if v is in our node set
                bool v_in_set = false;
                for(size_t k = 0; k < M; ++k) {
                    if(node_ids[k] == v) {
                        v_in_set = true;
                        break;
                    }
                }
                
                if(v_in_set) {
                    adj_list[u].push_back(v);
                }
            }
        } catch(...) {
            // Skip if error accessing neighbors
            continue;
        }
    }
    
    // BFS to find connected components within max_hops
    for(size_t i = 0; i < M; ++i) {
        int start_node = node_ids[i];
        if(visited.find(start_node) != visited.end()) {
            continue;
        }
        
        std::vector<int> group;
        std::queue<std::pair<int, int> > bfs_queue; // {node, hops}
        std::unordered_set<int> group_visited;
        
        bfs_queue.push({start_node, 0});
        group_visited.insert(start_node);
        visited.insert(start_node);
        group.push_back(start_node);
        
        while(!bfs_queue.empty()) {
            auto [current, hops] = bfs_queue.front();
            bfs_queue.pop();
            
            if(hops >= max_hops) {
                continue;
            }
            
            // Add neighbors that are in our node set
            if(adj_list.find(current) != adj_list.end()) {
                for(int neighbor : adj_list[current]) {
                    if(group_visited.find(neighbor) == group_visited.end() && 
                       visited.find(neighbor) == visited.end()) {
                        group_visited.insert(neighbor);
                        visited.insert(neighbor);
                        group.push_back(neighbor);
                        bfs_queue.push({neighbor, hops + 1});
                    }
                }
            }
        }
        
        if(!group.empty()) {
            groups.push_back(group);
        }
    }
    
    return groups;
}

// Calculate EH matrix for grouped nodes
std::vector<std::vector<uint16_t> > CalculateHardnessGrouped(
    HNSW_NGFix<float>* searcher,
    float* query_data,
    int* gt, 
    size_t M,  // original number of nodes
    size_t& m, // output: number of groups
    std::vector<int>& group_representatives, // output: representative node for each group
    std::unordered_map<int, int>& node_to_group, // output: mapping from node to group id
    size_t Kh, 
    size_t S) {
    
    // Step 1: Group nodes
    auto groups = GroupNodesByReachability(searcher, gt, M, 2);
    m = groups.size();
    
    // Step 2: For each group, select representative (closest to query)
    group_representatives.clear();
    node_to_group.clear();
    
    for(size_t group_id = 0; group_id < groups.size(); ++group_id) {
        const auto& group = groups[group_id];
        
        // Find node closest to query in this group
        float min_dist = std::numeric_limits<float>::max();
        int representative = group[0];
        
        for(int node_id : group) {
            if(node_id >= searcher->max_elements) continue;
            try {
                float dist = searcher->getQueryDist(node_id, query_data);
                if(dist < min_dist) {
                    min_dist = dist;
                    representative = node_id;
                }
            } catch(...) {
                continue;
            }
        }
        
        group_representatives.push_back(representative);
        
        // Map all nodes in group to group_id
        for(int node_id : group) {
            node_to_group[node_id] = group_id;
        }
    }
    
    // Step 3: Build reduced graph Gq with groups
    // If a group needs to connect to another group, use the representative node
    std::unordered_map<int, std::vector<int> > Gq_grouped; // group_id -> list of neighbor group_ids
    
    // Build mapping from original node to group_id
    std::unordered_map<int, int> node_to_group_map;
    for(size_t group_id = 0; group_id < groups.size(); ++group_id) {
        for(int node_id : groups[group_id]) {
            node_to_group_map[node_id] = group_id;
        }
    }
    
    // For each group, find connections to other groups
    // When a node in group A connects to group B, we record the connection
    // The representative node (closest to query) will be used for the group
    for(size_t group_id = 0; group_id < groups.size(); ++group_id) {
        std::unordered_set<int> connected_groups;
        
        // Check all nodes in this group
        for(int u : groups[group_id]) {
            if(u >= searcher->max_elements) continue;
            
            try {
                auto [neighbors, sz, st] = searcher->getNeighbors(u);
                
                for(int j = st; j < st + sz; ++j) {
                    int v = neighbors[j];
                    if(v >= searcher->max_elements) continue;
                    
                    // Check if v is in our node set and find its group
                    if(node_to_group_map.find(v) != node_to_group_map.end()) {
                        int v_group = node_to_group_map[v];
                        
                        // If different group and not already added
                        // This means there's a connection between group_id and v_group
                        if(v_group != (int)group_id && connected_groups.find(v_group) == connected_groups.end()) {
                            // When connecting groups, we use the representative nodes
                            // (already selected as closest to query in each group)
                            Gq_grouped[group_id].push_back(v_group);
                            connected_groups.insert(v_group);
                        }
                    }
                }
            } catch(...) {
                continue;
            }
        }
    }
    
    // Step 4: Calculate hardness matrix for groups (m x m)
    std::vector<std::vector<uint16_t> > H_grouped;
    H_grouped.resize(m);
    for(size_t i = 0; i < m; ++i) {
        H_grouped[i].resize(m, EH_INF);
    }
    
    // Create mapping from group representative to group index
    std::unordered_map<int, int> rep_to_group_idx;
    for(size_t i = 0; i < group_representatives.size(); ++i) {
        rep_to_group_idx[group_representatives[i]] = i;
    }
    
    // Initialize: groups are reachable to themselves
    for(size_t i = 0; i < m; ++i) {
        H_grouped[i][i] = i;
    }
    
    // Direct connections between groups
    for(auto [group_id, neighbor_groups] : Gq_grouped) {
        for(int neighbor_group_id : neighbor_groups) {
            if(group_id < m && neighbor_group_id >= 0 && neighbor_group_id < (int)m) {
                size_t g1 = group_id;
                size_t g2 = (size_t)neighbor_group_id;
                H_grouped[g1][g2] = std::max(g1, g2);
            }
        }
    }
    
    // Transitive closure: if group i can reach group h, and group h can reach group j,
    // then group i can reach group j
    // Simplified Floyd-Warshall-like algorithm
    for(size_t h = 0; h < m; ++h) {
        for(size_t i = 0; i < m; ++i) {
            if(H_grouped[i][h] != EH_INF) {
                for(size_t j = 0; j < m; ++j) {
                    if(H_grouped[h][j] != EH_INF) {
                        uint16_t max_val = std::max(H_grouped[i][h], H_grouped[h][j]);
                        uint16_t new_hardness = std::max({(uint16_t)i, (uint16_t)j, (uint16_t)h, max_val});
                        if(H_grouped[i][j] == EH_INF || new_hardness < H_grouped[i][j]) {
                            H_grouped[i][j] = new_hardness;
                        }
                    }
                }
            }
        }
    }
    
    return H_grouped;
}

int main(int argc, char* argv[]) {
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
            paths["K"] = argv[i + 1];
        if (arg == "--num_queries")
            paths["num_queries"] = argv[i + 1];
    }
    
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string index_path = paths["index_path"];
    std::string result_path = paths["result_path"];
    std::string metric_str = paths["metric"];
    size_t k = paths.count("K") ? std::stoi(paths["K"]) : 100;
    size_t num_queries = paths.count("num_queries") ? std::stoi(paths["num_queries"]) : 100;

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

    // Load base graph
    auto hnsw_base = new HNSW_NGFix<float>(metric, index_path);
    hnsw_base->printGraphInfo();

    // Limit to num_queries
    size_t actual_num_queries = std::min(num_queries, test_number);
    
    std::cout << "Testing EH computation with grouping optimization" << std::endl;
    std::cout << "Processing " << actual_num_queries << " queries..." << std::endl;
    std::cout << "k = " << k << std::endl;
    
    // Statistics
    std::vector<size_t> M_sizes;
    std::vector<size_t> m_sizes;
    std::vector<double> original_latencies_us;
    std::vector<double> grouped_latencies_us;
    
    size_t Nq = std::min(k, (size_t)100);  // Use k as Nq, but cap at 100
    size_t Kh = 100;
    size_t S = std::min((size_t)200, 2 * Nq);
    
    for(size_t i = 0; i < actual_num_queries; ++i) {
        if(i % 10 == 0) {
            std::cout << "Processing query " << i << "/" << actual_num_queries << std::endl;
        }
        
        auto query_data = test_query + i * vecdim;
        auto gt = test_gt + i * test_gt_dim;
        
        // Test original EH computation
        auto start_time = std::chrono::high_resolution_clock::now();
        
        size_t actual_M = std::min(Nq, (size_t)test_gt_dim);
        int* gt_subset = new int[actual_M];
        for(size_t j = 0; j < actual_M; ++j) {
            gt_subset[j] = gt[j];
        }
        
        auto H_original = hnsw_base->CalculateHardness(gt_subset, actual_M, Kh, S);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double original_latency_us = duration.count();
        
        M_sizes.push_back(actual_M);
        original_latencies_us.push_back(original_latency_us);
        
        // Test grouped EH computation
        start_time = std::chrono::high_resolution_clock::now();
        
        size_t m = 0;
        std::vector<int> group_representatives;
        std::unordered_map<int, int> node_to_group;
        
        auto H_grouped = CalculateHardnessGrouped(
            hnsw_base, query_data, gt_subset, actual_M, 
            m, group_representatives, node_to_group, Kh, S);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double grouped_latency_us = duration.count();
        
        m_sizes.push_back(m);
        grouped_latencies_us.push_back(grouped_latency_us);
        
        delete[] gt_subset;
        
        // Print progress for first few queries
        if(i < 5) {
            std::cout << "  Query " << i << ": M=" << actual_M << ", m=" << m 
                      << ", original_latency=" << original_latency_us << "us"
                      << ", grouped_latency=" << grouped_latency_us << "us"
                      << ", reduction=" << (1.0 - (double)m / actual_M) * 100.0 << "%"
                      << std::endl;
        }
    }
    
    // Calculate statistics
    double avg_M = 0, avg_m = 0;
    double avg_original_latency = 0, avg_grouped_latency = 0;
    double min_reduction = 100.0, max_reduction = 0.0;
    double total_reduction = 0.0;
    
    for(size_t i = 0; i < actual_num_queries; ++i) {
        avg_M += M_sizes[i];
        avg_m += m_sizes[i];
        avg_original_latency += original_latencies_us[i];
        avg_grouped_latency += grouped_latencies_us[i];
        
        double reduction = (1.0 - (double)m_sizes[i] / M_sizes[i]) * 100.0;
        total_reduction += reduction;
        min_reduction = std::min(min_reduction, reduction);
        max_reduction = std::max(max_reduction, reduction);
    }
    
    avg_M /= actual_num_queries;
    avg_m /= actual_num_queries;
    avg_original_latency /= actual_num_queries;
    avg_grouped_latency /= actual_num_queries;
    double avg_reduction = total_reduction / actual_num_queries;
    
    // Write results to JSON
    std::ofstream output(result_path);
    output << std::fixed << std::setprecision(6);
    
    output << "{\n";
    output << "  \"num_queries\": " << actual_num_queries << ",\n";
    output << "  \"k\": " << k << ",\n";
    output << "  \"Nq\": " << Nq << ",\n";
    output << "  \"statistics\": {\n";
    output << "    \"avg_M\": " << avg_M << ",\n";
    output << "    \"avg_m\": " << avg_m << ",\n";
    output << "    \"avg_reduction_percent\": " << avg_reduction << ",\n";
    output << "    \"min_reduction_percent\": " << min_reduction << ",\n";
    output << "    \"max_reduction_percent\": " << max_reduction << ",\n";
    output << "    \"avg_original_latency_us\": " << avg_original_latency << ",\n";
    output << "    \"avg_grouped_latency_us\": " << avg_grouped_latency << ",\n";
    output << "    \"speedup\": " << (avg_original_latency / avg_grouped_latency) << ",\n";
    output << "    \"original_matrix_size\": " << (avg_M * avg_M) << ",\n";
    output << "    \"grouped_matrix_size\": " << (avg_m * avg_m) << ",\n";
    output << "    \"matrix_size_reduction\": " << ((1.0 - (avg_m * avg_m) / (avg_M * avg_M)) * 100.0) << "\n";
    output << "  },\n";
    output << "  \"queries\": [\n";
    
    for(size_t i = 0; i < actual_num_queries; ++i) {
        output << "    {\n";
        output << "      \"query_id\": " << i << ",\n";
        output << "      \"M\": " << M_sizes[i] << ",\n";
        output << "      \"m\": " << m_sizes[i] << ",\n";
        output << "      \"reduction_percent\": " << ((1.0 - (double)m_sizes[i] / M_sizes[i]) * 100.0) << ",\n";
        output << "      \"original_latency_us\": " << original_latencies_us[i] << ",\n";
        output << "      \"grouped_latency_us\": " << grouped_latencies_us[i] << ",\n";
        output << "      \"speedup\": " << (original_latencies_us[i] / grouped_latencies_us[i]) << ",\n";
        output << "      \"original_matrix_size\": " << (M_sizes[i] * M_sizes[i]) << ",\n";
        output << "      \"grouped_matrix_size\": " << (m_sizes[i] * m_sizes[i]) << "\n";
        output << "    }";
        if(i < actual_num_queries - 1) {
            output << ",";
        }
        output << "\n";
    }
    
    output << "  ]\n";
    output << "}\n";
    output.close();
    
    // Print summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Average M (original node count): " << avg_M << std::endl;
    std::cout << "Average m (group count): " << avg_m << std::endl;
    std::cout << "Average reduction: " << avg_reduction << "%" << std::endl;
    std::cout << "Min reduction: " << min_reduction << "%" << std::endl;
    std::cout << "Max reduction: " << max_reduction << "%" << std::endl;
    std::cout << "Average original EH latency: " << avg_original_latency << " us" << std::endl;
    std::cout << "Average grouped EH latency: " << avg_grouped_latency << " us" << std::endl;
    std::cout << "Speedup: " << (avg_original_latency / avg_grouped_latency) << "x" << std::endl;
    std::cout << "Original matrix size (avg): " << (avg_M * avg_M) << std::endl;
    std::cout << "Grouped matrix size (avg): " << (avg_m * avg_m) << std::endl;
    std::cout << "Matrix size reduction: " << ((1.0 - (avg_m * avg_m) / (avg_M * avg_M)) * 100.0) << "%" << std::endl;
    std::cout << "\nResults written to: " << result_path << std::endl;
    
    delete[] test_query;
    delete[] test_gt;
    delete hnsw_base;
    
    return 0;
}

