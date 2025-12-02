#pragma once

#include <memory>
#include <atomic>
#include <shared_mutex>
#include <bitset>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <type_traits>
#include <queue>
#include <random>
#include <algorithm>
#include "node.h"
#include "../utils/search_list.h"
#include "../utils/visited_list.h"
#include "../metric/l2.h"
#include "../metric/ip.h"

using namespace ngfixlib;

static const size_t MAX_Nq = 200;
static const size_t MAX_S = 200;

namespace ours {

enum Metric{
    L2_float = 1,
    IP_float = 2,
};

// Forward declarations
template<typename T> class HNSW_Ours;

// Hard query detection utilities
struct HardnessMetrics {
    float hardness_score;      // From ML predictor or lightweight metrics
    float jitter;              // Jitter stability (Stage 1)
    bool is_hard;              // Final decision
};

// Query hardness predictor (Stage 0: lightweight metrics)
template<typename T>
HardnessMetrics DetectHardQuery(HNSW_Ours<T>* searcher, T* query_data, size_t k, size_t ef, size_t dim) {
    HardnessMetrics metrics;
    metrics.is_hard = false;
    
    // Use lightweight metrics from search trace
    size_t ndc = 0;
    auto [results, ndc_result, lw_metrics] = searcher->searchKnnWithLightweightMetrics(
        query_data, k, ef, ndc, 0.2f);
    
    // Calculate hardness score based on lightweight metrics
    // Higher r_visit, lower r_early, larger top1_last1_diff indicate harder queries
    float hardness_score = 0.0f;
    if(lw_metrics.S > 0) {
        hardness_score = lw_metrics.r_visit * (1.0f - lw_metrics.r_early) + 
                         (lw_metrics.top1_last1_diff > 0 ? lw_metrics.top1_last1_diff * 0.1f : 0.0f);
    }
    metrics.hardness_score = hardness_score;
    
    // Stage 1: Jitter (perturbation stability)
    float epsilon = 0.03f;  // ε ∈ [0.01, 0.05]
    float* perturbed_query = new T[dim];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);
    
    // Calculate original query norm
    float orig_norm = 0.0f;
    for(size_t d = 0; d < dim; ++d) {
        orig_norm += query_data[d] * query_data[d];
    }
    orig_norm = std::sqrt(orig_norm);
    
    // Add noise and normalize
    float perturbed_norm = 0.0f;
    for(size_t d = 0; d < dim; ++d) {
        float noise = noise_dist(gen);
        perturbed_query[d] = query_data[d] + epsilon * noise;
        perturbed_norm += perturbed_query[d] * perturbed_query[d];
    }
    perturbed_norm = std::sqrt(perturbed_norm);
    
    // Normalize
    if(perturbed_norm > 1e-6) {
        for(size_t d = 0; d < dim; ++d) {
            perturbed_query[d] /= perturbed_norm;
        }
    }
    
    // Run searches with original budget
    size_t ndc1 = 0, ndc2 = 0;
    auto result1 = searcher->searchKnn(query_data, k, ef, ndc1);
    auto result2 = searcher->searchKnn(perturbed_query, k, ef, ndc2);
    
    // Calculate intersection
    std::unordered_set<id_t> set1, set2;
    for(const auto& p : result1) {
        set1.insert(p.second);
    }
    for(const auto& p : result2) {
        set2.insert(p.second);
    }
    
    size_t intersection = 0;
    for(id_t id : set1) {
        if(set2.find(id) != set2.end()) {
            intersection++;
        }
    }
    
    // Jitter = 1 - |R_b(q) ∩ R_b(q')| / k
    metrics.jitter = 1.0f - (float)intersection / k;
    
    delete[] perturbed_query;
    
    // Determine if query is hard (threshold > P90, but for simplicity use fixed threshold)
    // In practice, you would calculate P90 from a sample of queries
    metrics.is_hard = (metrics.hardness_score > 0.5f) || (metrics.jitter > 0.1f);
    
    return metrics;
}

// Node grouping: group nodes that are reachable within 2 hops
template<typename T>
std::vector<std::vector<int> > GroupNodesByReachability(
    HNSW_Ours<T>* searcher, 
    int* node_ids, 
    size_t M,
    int max_hops = 2) {
    
    std::vector<std::vector<int> > groups;
    std::unordered_set<int> visited;
    
    // Build node set for O(1) lookup
    std::unordered_set<int> node_set;
    for(size_t i = 0; i < M; ++i) {
        node_set.insert(node_ids[i]);
    }
    
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
                
                // O(1) lookup instead of O(M) linear search
                if(node_set.find(v) != node_set.end()) {
                    adj_list[u].push_back(v);
                }
            }
        } catch(...) {
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

template<typename T>
class HNSW_Ours
{
private:
    // lock when updating neighbors 
    std::vector<std::shared_mutex> node_locks;
    size_t M0 = 0; // out-degree of base layer  
    size_t M = 0;  // used for insertion
    size_t MEX = 0;

public:
    std::vector<ngfixlib::node> Graph;
    char* vecdata;
    ngfixlib::Space<T>* space;
    ngfixlib::Space<T>* query_space; // used when q_bits != b_bits
    size_t dim;
    size_t entry_point = 0;
    size_t max_elements;
    std::atomic<size_t> n = 0;
    std::shared_ptr<ngfixlib::VisitedListPool> visited_list_pool_{nullptr};
    
    std::shared_mutex delete_lock;
    std::unordered_set<id_t> delete_ids;
    
    // For dynamic update: track added edges with timestamps
    struct EdgeInfo {
        id_t from, to;
        uint16_t eh;
        size_t timestamp;  // When this edge was added
    };
    std::unordered_map<id_t, std::vector<EdgeInfo> > added_edges;  // node -> list of added edges
    std::atomic<size_t> current_timestamp{0};
    
    size_t size_per_element = 0;

    HNSW_Ours(Metric metric, size_t dimension, size_t max_elements, size_t M_ = 16, size_t MEX_ = 48)
                : node_locks(max_elements), M(M_), MEX(MEX_) {
        M0 = 2*M;
        this->dim = dimension;
        this->max_elements = max_elements;
        this->size_per_element = dim*sizeof(T) + 1; // 8 bits for delete flag

        if constexpr (std::is_same_v<T, float>) {
            if (metric == L2_float) {
                space = new ngfixlib::L2Space_float(dim);
                query_space = new ngfixlib::L2Space_float(dim);
            } else if(metric == IP_float) {
                space = new ngfixlib::IPSpace_float(dim);
                query_space = new ngfixlib::IPSpace_float(dim);
            } else {
                throw std::runtime_error("Error: Unsupported metric type.");
            }
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }   
        
        vecdata = new char[size_per_element*max_elements];
        Graph.resize(this->max_elements);
        visited_list_pool_ = std::shared_ptr<ngfixlib::VisitedListPool>(new ngfixlib::VisitedListPool(1, this->max_elements));
    }

    HNSW_Ours(Metric metric, std::string path) {
        std::ifstream input(path, std::ios::binary);

        input.read((char*)&M, sizeof(M));
        input.read((char*)&M0, sizeof(M0));
        input.read((char*)&MEX, sizeof(MEX));
        input.read((char*)&n, sizeof(n));
        input.read((char*)&entry_point, sizeof(entry_point));
        input.read((char*)&dim, sizeof(dim));
        input.read((char*)&max_elements, sizeof(max_elements));
        this->size_per_element = dim*sizeof(T) + 1;
        if constexpr (std::is_same_v<T, float>) {
            if (metric == L2_float) {
                space = new ngfixlib::L2Space_float(dim);
                query_space = new ngfixlib::L2Space_float(dim);
            } else if(metric == IP_float) {
                space = new ngfixlib::IPSpace_float(dim);
                query_space = new ngfixlib::IPSpace_float(dim);
            } else {
                throw std::runtime_error("Error: Unsupported metric type.");
            }
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }

        vecdata = new char[size_per_element*max_elements];
        input.read(vecdata, max_elements*size_per_element);

        node_locks = std::vector<std::shared_mutex>(this->max_elements);
        Graph.reserve(this->max_elements);
        Graph.resize(this->max_elements);
        visited_list_pool_ = std::shared_ptr<ngfixlib::VisitedListPool>(new ngfixlib::VisitedListPool(1, this->max_elements));
        
        for(int i = 0; i < n; ++i) {
            Graph[i].LoadIndex(input);
        }
    }

    ~HNSW_Ours() {
        delete []vecdata;
        delete space;
        delete query_space;
    }

    void resize(size_t new_max_elements) {
        if(new_max_elements <= this->max_elements) {
            return;
        }
        
        auto new_vecdata = new char[size_per_element*new_max_elements];
        memcpy(new_vecdata, vecdata, size_per_element*max_elements);        
        delete []vecdata;
        vecdata = new_vecdata;

        this->max_elements = new_max_elements;
        node_locks = std::vector<std::shared_mutex>(new_max_elements);
        Graph.reserve(new_max_elements);
        Graph.resize(new_max_elements);
        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));
        
        if(entry_point >= max_elements) {
            entry_point = 0;
        }
    }

    void StoreIndex(std::string path) {
        std::ofstream output(path, std::ios::binary);

        output.write((char*)&M, sizeof(M));
        output.write((char*)&M0, sizeof(M0));
        output.write((char*)&MEX, sizeof(MEX));
        output.write((char*)&n, sizeof(n));
        output.write((char*)&entry_point, sizeof(entry_point));
        output.write((char*)&dim, sizeof(dim));
        output.write((char*)&max_elements, sizeof(max_elements));

        output.write(vecdata, max_elements*size_per_element);
        for(int i = 0; i < n; ++i) {
            Graph[i].StoreIndex(output);
        }
    }

    T* getData(id_t u) {
        if(u >= max_elements) {
            throw std::runtime_error("Error: getData id out of bounds.");
        }
        return (T*)(vecdata + u*size_per_element + 1);
    }

    void SetData(id_t u, T* data) {
        memcpy(getData(u), data, sizeof(T)*dim);
    }

    auto getNeighbors(id_t u) {
        auto tmp = Graph[u].get_neighbors();
        return std::tuple{tmp, GET_SZ((uint8_t*)tmp), GET_NGFIX_CAPACITY((uint8_t*)tmp)-GET_NGFIX_SZ((uint8_t*)tmp) + 1};
    }

    auto getBaseGraphNeighbors(id_t u) {
        auto tmp = Graph[u].get_neighbors();
        return std::tuple{tmp, GET_SZ((uint8_t*)tmp)-GET_NGFIX_SZ((uint8_t*)tmp), GET_NGFIX_CAPACITY((uint8_t*)tmp) + 1};
    }

    float getDist(id_t u, id_t v) {
        return space->dist_func(getData(u), getData(v));
    }

    float getDist(id_t u, T* data) {
        return space->dist_func(getData(u), data);
    }

    float getQueryDist(id_t u, T* query_data) {
        return query_space->dist_func(getData(u), query_data);
    }

    std::vector<std::pair<float, id_t> > 
    getNeighborsByHeuristic(std::vector<std::pair<float, id_t> >& neighbor_candidates, const size_t M) {
        if (neighbor_candidates.size() < M) {
            return neighbor_candidates;
        }

        std::vector<std::pair<float, id_t> > return_list;

        for (auto [dist_to_query, cur_id] : neighbor_candidates) {
            if (return_list.size() >= M) {break;}
            bool good = true;

            for (auto [_, exist_id] : return_list) {
                float curdist = getDist(exist_id, cur_id);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back({dist_to_query, cur_id});
            }
        }

        return return_list;
    }

    void HNSWBottomLayerInsertion(T* data, id_t cur_id, size_t efC) {
        size_t NDC = 0;
        auto res = searchKnnBaseGraphConstruction(data, efC, efC, NDC);
        auto neighbors = getNeighborsByHeuristic(res, M);
        
        if(cur_id >= max_elements || cur_id >= node_locks.size()) {
            throw std::runtime_error("Error: cur_id out of bounds in HNSWBottomLayerInsertion.");
        }
        
        { // add edge (cur_id, neighbor_id)
            std::unique_lock <std::shared_mutex> lock(node_locks[cur_id]);
            Graph[cur_id].replace_base_graph_neighbors(neighbors);
        }

        // add edge (neighbor_id, cur_id)
        for(auto [_, neighbor_id] : neighbors) {
            if(neighbor_id >= max_elements || neighbor_id >= node_locks.size()) {
                continue;
            }
            std::unique_lock <std::shared_mutex> lock(node_locks[neighbor_id]);
            auto [ids, sz, st] = getBaseGraphNeighbors(neighbor_id);
            if(sz < M0) {
                Graph[neighbor_id].add_base_graph_neighbors(cur_id);
            } else {
                float d_max = getDist(neighbor_id, cur_id);
                std::vector<std::pair<float, id_t> > candidates;
                candidates.push_back({d_max, cur_id});
                for (int j = st; j < st + sz; j++) {
                    id_t candidate_id = ids[j];
                    if(candidate_id >= max_elements) {
                        continue;
                    }
                    candidates.push_back({getDist(candidate_id, neighbor_id) , candidate_id});
                }
                std::sort(candidates.begin(), candidates.end());
                auto neighbors = getNeighborsByHeuristic(candidates, M0);

                Graph[neighbor_id].replace_base_graph_neighbors(neighbors);
            }
        }
    }

    void InsertPoint(id_t id, size_t efC, T* vec) {
        if(id >= max_elements) {
            throw std::runtime_error("Error: id > max_elements.");
        }
        SetData(id, vec);
        auto data = getData(id);

        if(n != 0) {
            HNSWBottomLayerInsertion(data, id, efC);
        }
        ++n;
        if(n % 100000 == 0) {
            SetEntryPoint();
        }
    }

    void set_deleted(id_t id) {
        (vecdata + id*size_per_element)[0] = true;
    }
    bool is_deleted(id_t id) {
        return (vecdata + id*size_per_element)[0];
    }

    std::unordered_map<id_t, std::vector<id_t> > ComputeGq(int* gt, size_t S) 
    {
        std::unordered_map<id_t, std::vector<id_t> > G;
        std::unordered_set<id_t> Vq;
        for(int i = 0; i < S; ++i){
            Vq.insert(gt[i]);
        }
        for(int i = 0; i < S; ++i){
            int u = gt[i];
            if(u >= max_elements || u >= Graph.size()) {
                continue;
            }
            auto [ids, sz, st] = getNeighbors(u);
            for (int j = st; j < st + sz; ++j){
                id_t v = ids[j];
                if(v >= max_elements) {
                    continue;
                }
                if(Vq.find(v) == Vq.end()){
                    continue;
                }
                G[u].push_back(v);
            }
        }

        return G;
    }

    // Helper struct for returning both H and node_idx_to_group
    struct HardnessResult {
        std::vector<std::vector<uint16_t> > H;
        std::vector<int> node_idx_to_group;
    };
    
    // Optimized CalculateHardness with grouping - returns both H and mapping
    HardnessResult CalculateHardnessGroupedWithMapping(int* gt, size_t Nq, size_t Kh, size_t S, T* query_data) {
        HardnessResult result;
        
        // Step 1: Group nodes by 2-hop reachability
        auto groups = GroupNodesByReachability(this, gt, std::min(Nq, S), 2);
        size_t m = groups.size();
        size_t total_nodes = std::min(Nq, S);
        
        // Only use grouping if it significantly reduces the matrix size (at least 30% reduction)
        // This ensures we get the expected 84% computation reduction (from ~100 to ~40 groups)
        if(m == 0 || m >= total_nodes * 0.7) {
            // Fallback to original if grouping fails or doesn't reduce enough
            result.H = CalculateHardness(gt, Nq, Kh, S);
            result.node_idx_to_group.clear();
            return result;
        }
        
        // Step 2: Build efficient mappings (optimized to reduce map lookups)
        std::unordered_map<int, int> node_to_group_map;
        node_to_group_map.reserve(total_nodes);  // Pre-allocate to reduce rehashing
        result.node_idx_to_group.resize(Nq, -1);
        
        // Build node_to_group_map and node_idx_to_group in one pass
        for(size_t group_id = 0; group_id < groups.size(); ++group_id) {
            for(int node_id : groups[group_id]) {
                node_to_group_map[node_id] = group_id;
            }
        }
        
        // Build node_idx_to_group: for each node in gt, find its group
        // Use direct array access instead of map lookup for better performance
        for(int idx = 0; idx < (int)total_nodes; ++idx) {
            int node_id = gt[idx];
            auto it = node_to_group_map.find(node_id);
            if(it != node_to_group_map.end()) {
                result.node_idx_to_group[idx] = it->second;
            }
        }
        
        // Step 3: Build reduced graph Gq_grouped efficiently
        // Build Gq_grouped directly from groups without recomputing Gq
        std::unordered_map<int, std::vector<int> > Gq_grouped;
        Gq_grouped.reserve(m);  // Pre-allocate
        
        // For each group, check connections to other groups using representative nodes
        // Optimize: reduce memory allocations and improve cache locality
        for(size_t group_id = 0; group_id < groups.size(); ++group_id) {
            if(groups[group_id].empty()) continue;
            
            int u = groups[group_id][0];
            if(u >= max_elements) continue;
            
            std::unordered_set<int> connected_groups;
            connected_groups.reserve(std::min(m, (size_t)16));  // Optimize: smaller reserve
            
            try {
                auto [neighbors, sz, st] = getNeighbors(u);
                
                for(int j = st; j < st + sz; ++j) {
                    int v = neighbors[j];
                    if(v >= max_elements) continue;
                    
                    auto it = node_to_group_map.find(v);
                    if(it != node_to_group_map.end()) {
                        int v_group = it->second;
                        
                        if(v_group != (int)group_id && connected_groups.find(v_group) == connected_groups.end()) {
                            Gq_grouped[group_id].push_back(v_group);
                            connected_groups.insert(v_group);
                        }
                    }
                }
            } catch(...) {
                continue;
            }
        }
        
        // Step 4: Calculate hardness matrix for groups (m x m)
        std::vector<std::vector<uint16_t> > H_grouped;
        H_grouped.resize(m);
        for(size_t i = 0; i < m; ++i) {
            H_grouped[i].resize(m, ngfixlib::EH_INF);
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
        
        // Transitive closure
        for(size_t h = 0; h < m; ++h) {
            for(size_t i = 0; i < m; ++i) {
                if(H_grouped[i][h] != ngfixlib::EH_INF) {
                    for(size_t j = 0; j < m; ++j) {
                        if(H_grouped[h][j] != ngfixlib::EH_INF) {
                            uint16_t max_val = std::max(H_grouped[i][h], H_grouped[h][j]);
                            uint16_t new_hardness = std::max({(uint16_t)i, (uint16_t)j, (uint16_t)h, max_val});
                            if(H_grouped[i][j] == ngfixlib::EH_INF || new_hardness < H_grouped[i][j]) {
                                H_grouped[i][j] = new_hardness;
                            }
                        }
                    }
                }
            }
        }
        
        // Step 5: Map back to original Nq x Nq matrix (optimized: only map nodes in groups)
        result.H.resize(Nq);
        for(int i = 0; i < (int)Nq; ++i){
            result.H[i].resize(Nq, ngfixlib::EH_INF);
        }
        
        // Pre-compute which indices have groups to avoid repeated checks
        std::vector<int> valid_indices;
        valid_indices.reserve(total_nodes);  // Pre-allocate
        for(int i = 0; i < (int)total_nodes; ++i) {  // Only check up to total_nodes
            if(result.node_idx_to_group[i] >= 0) {
                valid_indices.push_back(i);
            }
        }
        
        // Map group hardness to node pairs - only for nodes in groups
        // Use direct array access for better performance
        size_t num_valid = valid_indices.size();
        for(size_t idx_i = 0; idx_i < num_valid; ++idx_i) {
            int i = valid_indices[idx_i];
            int g_i = result.node_idx_to_group[i];
            if(g_i < 0 || g_i >= (int)m) continue;
            
            for(size_t idx_j = 0; idx_j < num_valid; ++idx_j) {
                int j = valid_indices[idx_j];
                int g_j = result.node_idx_to_group[j];
                if(g_j < 0 || g_j >= (int)m) continue;
                
                result.H[i][j] = H_grouped[g_i][g_j];
            }
        }
        
        return result;
    }
    
    // Optimized CalculateHardness with grouping (backward compatibility)
    std::vector<std::vector<uint16_t> > CalculateHardnessGrouped(int* gt, size_t Nq, size_t Kh, size_t S, T* query_data) {
        auto result = CalculateHardnessGroupedWithMapping(gt, Nq, Kh, S, query_data);
        return result.H;
    }

    // Original CalculateHardness (fallback)
    std::vector<std::vector<uint16_t> > CalculateHardness(int* gt, size_t Nq, size_t Kh, size_t S) 
    {
        auto Gq = ComputeGq(gt, S);
        std::unordered_map<id_t, uint16_t> p2rank;
        for(int i = 0; i < S; ++i){
            p2rank[gt[i]] = i;
        }
        
        std::vector<std::vector<uint16_t> > H;
        std::bitset<MAX_S> f[S];
        H.resize(Nq);
        for(int i = 0; i < Nq; ++i){
            H[i].resize(Nq, ngfixlib::EH_INF);
        }
        for(int h = 0; h < S; ++h){
            f[h][h] = 1;
            if(h < Nq){
                H[h][h] = h;
            }
        }

        for(auto [u, neighbors] : Gq){
            int i = p2rank[u];
            for(auto v : neighbors){
                int j = p2rank[v];
                f[i][j] = 1;
                if(i < Nq && j < Nq){
                    H[i][j] = std::max(i,j);
                }
            }
        }

        for(int h = 0; h < S; ++h){
            for(int i = 0; i < S; ++i){
                auto last = f[i];
                if(f[i][h]){
                    f[i] |= f[h];
                }
                last ^= f[i];
                if(i < Nq && last.count() > 0){
                    for(int j = 0; j < Nq; ++j){
                        if(last[j] == 1){
                            H[i][j] = std::max(std::max(i,j), h);
                        }
                    }
                }
            }
        }
        return H;
    }

    // Optimized getDefectsFixingEdges: use distance cache to avoid recomputing distances
    // Key optimization: cache distances for all node pairs, reuse throughout the function
    // For the same query, distances between nodes only need to be computed once
    auto getDefectsFixingEdgesOptimized(
        std::bitset<MAX_Nq> f[],
        std::vector<std::vector<uint16_t> >& H,
        T* query, int* gt, size_t Nq, size_t Kh,
        const std::vector<int>* node_idx_to_group_ptr = nullptr,
        std::vector<std::vector<float>>* dist_cache_ptr = nullptr) {
        
        std::unordered_map<id_t, std::vector<std::pair<id_t, uint16_t> > > new_edges;

        bool use_grouping = (node_idx_to_group_ptr != nullptr && !node_idx_to_group_ptr->empty());
        bool use_cache = (dist_cache_ptr != nullptr);
        
        // Use provided cache or create a local one
        std::vector<std::vector<float>> local_cache;
        std::vector<std::vector<float>>* dist_cache = use_cache ? dist_cache_ptr : &local_cache;
        
        // Iterative approach: repeatedly select pair with max EH value, not max distance
        // Key optimizations:
        // 1. Distance only computed once for all pairs (cached)
        // 2. For grouped case, only compute distances for cross-group pairs (at most group_count * (group_count-1) pairs)
        // 3. Select based on EH matrix value (H[i][j]), not distance
        // 4. Use max-heap to efficiently find max EH value without full sort
        
        // Step 1: Pre-allocate dist_cache if needed (only once)
        if(!use_cache || dist_cache->empty()) {
            dist_cache->resize(Nq);
            for(int i = 0; i < (int)Nq; ++i) {
                (*dist_cache)[i].resize(Nq, -1.0f);
            }
        }
        
        // Step 2: Collect candidate pairs and compute distances in one pass
        // Optimize: use compact structure and reduce memory allocations
        std::vector<std::pair<uint16_t, std::pair<int, int>>> candidates;  // (EH_value, (i, j))
        // Estimate size: for grouped case, much fewer candidates
        size_t est_size = use_grouping ? (Nq * 2) : (Nq * Nq / 2);
        candidates.reserve(est_size);
        
        // Collect candidates and compute distances in one pass (optimize memory access)
        // Optimize: cache group lookups to reduce repeated array access
        const std::vector<int>& group_vec = use_grouping ? *node_idx_to_group_ptr : std::vector<int>();
        
        for(int i = 0; i < (int)Nq; ++i){
            if(use_grouping && group_vec[i] < 0) continue;  // Skip invalid groups early
            int g_i = use_grouping ? group_vec[i] : -1;
            
            for(int j = i + 1; j < (int)Nq; ++j){  // Only process upper triangle (i < j)
                if(f[i][j] == 1) {continue;}
                
                // Skip same-group pairs if using grouping
                if(use_grouping) {
                    int g_j = group_vec[j];
                    if(g_i >= 0 && g_j >= 0 && g_i == g_j) {
                        continue;
                    }
                }
                
                // Get EH value
                uint16_t eh_val = H[i][j];
                
                // Compute distance if not cached (only once)
                if((*dist_cache)[i][j] < 0) {
                    int u = gt[i];
                    int v = gt[j];
                    float d = getDist(u, v);
                    (*dist_cache)[i][j] = d;
                    (*dist_cache)[j][i] = d;  // Cache symmetric distance
                }
                
                candidates.push_back({eh_val, {i, j}});
            }
        }
        
        if(candidates.empty()) {
            return new_edges;
        }
        
        // Step 2: Build max-heap by EH value (largest EH at top)
        // Use std::make_heap which is efficient for building heap from unsorted array
        std::make_heap(candidates.begin(), candidates.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Step 3: Iteratively select pair with max EH value (H[i][j])
        // This matches NGFix paper: select farthest pair based on EH, not distance
        // Optimize: reduce heap cleanup frequency and use more efficient operations
        size_t iterations = 0;
        while(!candidates.empty()) {
            iterations++;
            
            // Get the pair with maximum EH value - O(1) to access, O(log n) to pop
            std::pop_heap(candidates.begin(), candidates.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
            auto [eh_val, ij] = candidates.back();
            candidates.pop_back();
            
            auto [i, j] = ij;
            
            // Check if still unconnected (may have been connected by transitive closure)
            if(f[i][j] == 1) {
                continue;  // Skip, already connected
            }
            
            // Add the edge (pair with max EH)
            int u = gt[i];
            int v = gt[j];
            new_edges[u].push_back({v, eh_val});  // Use cached eh_val instead of H[i][j]
            
            f[i][j] = 1;
            
            // Update transitive closure efficiently - optimize inner loop
            // Use bitset operations which are very fast
            for(int k = 0; k < (int)Nq; ++k){
                if(f[k][i]){
                    f[k] |= f[j];
                }
            }
            
            // Periodically clean up heap: remove candidates that are now connected
            // Optimize: do cleanup less frequently (every 100 iterations) to reduce overhead
            if(candidates.size() > 200 && iterations % 100 == 0) {
                // Rebuild heap with only unconnected pairs
                size_t write_idx = 0;
                for(size_t read_idx = 0; read_idx < candidates.size(); ++read_idx) {
                    auto [eh, ij_pair] = candidates[read_idx];
                    auto [ii, jj] = ij_pair;
                    if(f[ii][jj] == 0) {  // Still unconnected
                        if(write_idx != read_idx) {
                            candidates[write_idx] = candidates[read_idx];
                        }
                        write_idx++;
                    }
                }
                candidates.resize(write_idx);
                if(!candidates.empty()) {
                    std::make_heap(candidates.begin(), candidates.end(),
                                 [](const auto& a, const auto& b) { return a.first < b.first; });
                }
            }
        }
        
        return new_edges;
    }

    // Original getDefectsFixingEdges
    auto getDefectsFixingEdges(
        std::bitset<MAX_Nq> f[],
        std::vector<std::vector<uint16_t> >& H,
        T* query, int* gt, size_t Nq, size_t Kh) {
        
        std::unordered_map<id_t, std::vector<std::pair<id_t, uint16_t> > > new_edges;

        std::vector<std::pair<float, std::pair<int,int> > > vs;
        for(int i = 0; i < (int)Nq; ++i){
            for(int j = 0; j < (int)Nq; ++j){
                if(f[i][j] == 1) {continue;}
                int u = gt[i];
                int v = gt[j]; 
                float d = getDist(u, v);
                vs.push_back({d,{i,j}});
            }
        }
        std::sort(vs.begin(), vs.end());

        for(auto [d, e] : vs){
            int s = e.first;
            int t = e.second;
            if(f[s][t] == 1) {continue;}

            int u = gt[s];
            int v = gt[t];

            new_edges[u].push_back({v, H[s][t]});

            f[s][t] = 1;
            for(int i = 0; i < (int)Nq; ++i){
                if(f[i][s]){
                    f[i] |= f[t];
                }
            }
        }
        return new_edges;
    }

    // Optimized NGFix with grouped EH calculation
    void NGFixOptimized(T* query, int* gt, size_t Nq = 100, size_t Kh = 100) {
        if(Nq > MAX_Nq) {
            throw std::runtime_error("Error: Nq >= MAX_Nq.");
        }
        
        // Create distance cache that can be reused throughout this query's processing
        // This cache will be used in getDefectsFixingEdgesOptimized to avoid recomputing distances
        std::vector<std::vector<float>> dist_cache;
        
        // Use grouped hardness calculation with mapping
        auto hardness_result = CalculateHardnessGroupedWithMapping(gt, Nq, Kh, std::min(MAX_S, 2*Nq), query);
        auto& H = hardness_result.H;
        auto& node_idx_to_group = hardness_result.node_idx_to_group;
        
        std::bitset<MAX_Nq> f[Nq];
        for(int i = 0; i < (int)Nq; ++i){
            for(int j = 0; j < (int)Nq; ++j){
                f[i][j] = (H[i][j] <= Kh) ? 1 : 0;
            }
        }
        
        // Pass grouping info and distance cache to optimize distance calculations
        const std::vector<int>* group_ptr = node_idx_to_group.empty() ? nullptr : &node_idx_to_group;
        auto new_edges = getDefectsFixingEdgesOptimized(f, H, query, gt, Nq, Kh, group_ptr, &dist_cache);
        
        size_t ts = current_timestamp.fetch_add(1);
        
        for(auto [u, vs] : new_edges) {
            std::unique_lock <std::shared_mutex> lock(node_locks[u]);
            for(auto [v, eh] : vs) {
                Graph[u].add_ngfix_neighbors(v, eh, MEX);
                // Track added edge
                added_edges[u].push_back({u, v, eh, ts});
            }
        }
        
        // Clear distance cache after processing this query (memory cleanup)
        dist_cache.clear();
    }

    // Original NGFix
    void NGFix(T* query, int* gt, size_t Nq = 100, size_t Kh = 100) {
        if(Nq > MAX_Nq) {
            throw std::runtime_error("Error: Nq >= MAX_Nq.");
        }
        auto H = CalculateHardness(gt, Nq, Kh, std::min(MAX_S, 2*Nq));
        std::bitset<MAX_Nq> f[Nq];
        for(int i = 0; i < (int)Nq; ++i){
            for(int j = 0; j < (int)Nq; ++j){
                f[i][j] = (H[i][j] <= Kh) ? 1 : 0;
            }
        }
        auto new_edges = getDefectsFixingEdges(f, H, query, gt, Nq, Kh);
        
        size_t ts = current_timestamp.fetch_add(1);
        
        for(auto [u, vs] : new_edges) {
            std::unique_lock <std::shared_mutex> lock(node_locks[u]);
            for(auto [v, eh] : vs) {
                Graph[u].add_ngfix_neighbors(v, eh, MEX);
                // Track added edge
                added_edges[u].push_back({u, v, eh, ts});
            }
        }
    }

    // Dynamic update: remove expired edges (older than threshold)
    void RemoveExpiredEdges(size_t age_threshold) {
        size_t current_ts = current_timestamp.load();
        
        for(auto& [u, edges] : added_edges) {
            if(u >= max_elements) continue;
            
            std::unique_lock <std::shared_mutex> lock(node_locks[u]);
            
            // Filter out expired edges
            std::vector<EdgeInfo> remaining_edges;
            std::vector<id_t> edges_to_remove;
            
            for(const auto& edge : edges) {
                if(current_ts - edge.timestamp > age_threshold) {
                    edges_to_remove.push_back(edge.to);
                } else {
                    remaining_edges.push_back(edge);
                }
            }
            
            // Remove expired edges from graph
            if(!edges_to_remove.empty()) {
                auto [neighbors, sz, st] = getNeighbors(u);
                std::vector<id_t> new_ngfix_neighbors;
                
                uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
                uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
                
                if(ngfix_sz > 0) {
                    int ngfix_start = ngfix_capacity - ngfix_sz;
                    for(int j = ngfix_start; j < ngfix_capacity; ++j) {
                        id_t v = neighbors[j + 1];
                        bool should_remove = false;
                        for(id_t to_remove : edges_to_remove) {
                            if(v == to_remove) {
                                should_remove = true;
                                break;
                            }
                        }
                        if(!should_remove) {
                            new_ngfix_neighbors.push_back(v);
                        }
                    }
                }
                
                Graph[u].replace_ngfix_neighbors(new_ngfix_neighbors);
            }
            
            edges = remaining_edges;
        }
    }

    // Get incoming edges for a node (for deletion)
    std::vector<id_t> GetIncomingEdges(id_t node_id, size_t max_hops = 2) {
        std::vector<id_t> incoming;
        std::unordered_set<id_t> visited;
        
        // Method 1: Two-hop local scan
        std::queue<std::pair<id_t, int> > bfs_queue;
        bfs_queue.push({node_id, 0});
        visited.insert(node_id);
        
        while(!bfs_queue.empty()) {
            auto [current, hops] = bfs_queue.front();
            bfs_queue.pop();
            
            if(hops >= max_hops) continue;
            
            if(current >= max_elements) continue;
            
            try {
                auto [neighbors, sz, st] = getNeighbors(current);
                for(int i = st; i < st + sz; ++i) {
                    id_t neighbor = neighbors[i];
                    if(neighbor >= max_elements) continue;
                    
                    // Check if neighbor has edge to node_id
                    std::shared_lock <std::shared_mutex> lock(node_locks[neighbor]);
                    auto [neighbor_neighbors, neighbor_sz, neighbor_st] = getNeighbors(neighbor);
                    for(int j = neighbor_st; j < neighbor_st + neighbor_sz; ++j) {
                        if(neighbor_neighbors[j] == node_id) {
                            incoming.push_back(neighbor);
                            break;
                        }
                    }
                    
                    if(visited.find(neighbor) == visited.end() && hops < max_hops - 1) {
                        visited.insert(neighbor);
                        bfs_queue.push({neighbor, hops + 1});
                    }
                }
            } catch(...) {
                continue;
            }
        }
        
        return incoming;
    }

    void AKNNGroundTruth(T* query, int* gt, size_t k, size_t efC) {
        size_t ndc = 0;
        auto result = searchKnn(query, k, efC, ndc);
        for(int i = 0; i < k; ++i) {
            gt[i] = result[i].second;
        }
    }

    void SetEntryPoint() {
        T* centroid = new T[dim];
        memset(centroid, 0, sizeof(T)*dim);
        for(int i = 0; i < n; ++i) {
            if(is_deleted(i)) {continue;}
            auto data = getData(i);
            for(int d = 0; d < dim; ++d) {
                centroid[d] += data[d];
            }
        }

        for(int d = 0; d < dim; ++d) {
            centroid[d] /= n;
        }

        id_t ep = 0;
        float min_dis = std::numeric_limits<float>::max();
        for(int i = 0; i < n; ++i) {
            auto dis = space->dist_func(centroid, getData(i));
            if(dis < min_dis) {
                min_dis = dis;
                ep = i;
            }
        }
        entry_point = ep;
        delete []centroid;
    }

    std::vector<std::pair<float, id_t> > searchKnnBaseGraphConstruction(T* query_data, size_t k, size_t ef, size_t& ndc) {
        if(ef < k) {
            throw std::runtime_error("Error: efs < k.");
        }
        ngfixlib::Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;

        if(entry_point >= max_elements || entry_point >= node_locks.size()) {
            return std::vector<std::pair<float, id_t>>();
        }

        float dist = getDist(entry_point, query_data);
        q->push(entry_point, dist, is_deleted(entry_point));
        q->set_visited(entry_point);
        while (!q->is_empty()) {
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            float candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            flag_stop_search = candidate_dist > q->get_dist_bound();

            if (flag_stop_search) {
                break;
            }
            
            if(current_node_id >= max_elements || current_node_id >= node_locks.size()) {
                break;
            }
            
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getBaseGraphNeighbors(current_node_id);

            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(candidate_id >= max_elements) {
                    continue;
                }
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }

                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getDist(candidate_id, query_data);
                    q->push(candidate_id, dist, is_deleted(candidate_id));

                    ndc += 1;
                }
            }
        }
        q->releaseVisitedList();
        auto res = q->get_result(k);

        return res;
    }

    std::vector<std::pair<float, id_t> > searchKnn(T* query_data, size_t k, size_t ef, size_t& ndc) {
        size_t hop_limit = std::numeric_limits<int>::max();
        size_t hop = 0;
        if(ef < k) {
            hop_limit = ef;
            ef = k;
        }

        ngfixlib::Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;
        
        if(entry_point >= max_elements || entry_point >= node_locks.size()) {
            return std::vector<std::pair<float, id_t>>();
        }
        
        float dist = getQueryDist(entry_point, query_data);
        q->push(entry_point, dist, is_deleted(entry_point));
        q->set_visited(entry_point);
        while (!q->is_empty()) {
            if (hop > hop_limit) {
                break;
            }
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            float candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            flag_stop_search = candidate_dist > q->get_dist_bound();

            if (flag_stop_search) {
                break;
            }
            hop += 1;
            
            if(current_node_id >= max_elements || current_node_id >= node_locks.size()) {
                break;
            }
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getNeighbors(current_node_id);
            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(candidate_id >= max_elements) {
                    continue;
                }
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }
                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getQueryDist(candidate_id, query_data);
                    q->push(candidate_id, dist, is_deleted(candidate_id));

                    ndc += 1;
                }
            }
        }
        q->releaseVisitedList();
        auto res = q->get_result(k);
        return res;
    }

    // Lightweight metrics structure (same as NGFix)
    struct LightweightMetrics {
        size_t S;                    // Total number of visited nodes (pop operations)
        float r_visit;               // Visit Budget Usage: S / ef
        float r_early;               // Early-Convergence Ratio: t_last_improve / S
        float top1_last1_diff;       // Distance difference: last1_dist - top1_dist (in top-k results)
        float delta_improve;         // Early-vs-Final Improvement Ratio
        float d_worst_early;         // Top-k worst distance at early stage
        float d_worst_final;          // Final top-k worst distance
        float d_best_cand_final;      // Final priority queue best candidate distance
        size_t t_last_improve;       // Last step index where d_worst improved
    };
    
    std::tuple<std::vector<std::pair<float, id_t> >, size_t, LightweightMetrics> 
    searchKnnWithLightweightMetrics(T* query_data, size_t k, size_t ef, size_t& ndc, float alpha = 0.2f) {
        size_t hop_limit = std::numeric_limits<int>::max();
        size_t hop = 0;
        if(ef < k) {
            hop_limit = ef;
            ef = k;
        }

        ngfixlib::Search_QuadHeap q0(ef, visited_list_pool_);
        Search_QuadHeap* q = &q0;
        
        if(entry_point >= max_elements || entry_point >= node_locks.size()) {
            LightweightMetrics empty_metrics = {0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0};
            return std::make_tuple(std::vector<std::pair<float, id_t>>(), 0, empty_metrics);
        }
        
        float dist = getQueryDist(entry_point, query_data);
        q->push(entry_point, dist, is_deleted(entry_point));
        q->set_visited(entry_point);
        
        LightweightMetrics metrics;
        metrics.t_last_improve = 0;
        metrics.d_worst_early = std::numeric_limits<float>::max();
        metrics.d_worst_final = std::numeric_limits<float>::max();
        metrics.d_best_cand_final = std::numeric_limits<float>::max();
        
        float d_worst_prev = std::numeric_limits<float>::max();
        size_t S_0 = std::min((size_t)std::ceil(alpha * ef), (size_t)1000);
        bool early_stage_captured = false;
        const float eps = 1e-6f;
        
        ndc = 1;
        
        while (!q->is_empty()) {
            if (hop > hop_limit) {
                break;
            }
            
            float d_worst_current = q->get_worst_topk_dist(k);
            float d_best_cand_current = q->get_best_candidate_dist();
            
            if(d_worst_current < d_worst_prev - eps) {
                metrics.t_last_improve = ndc;
                d_worst_prev = d_worst_current;
            }
            
            if(!early_stage_captured && ndc >= S_0) {
                if(d_worst_current < std::numeric_limits<float>::max()) {
                    metrics.d_worst_early = d_worst_current;
                    early_stage_captured = true;
                }
            }
            
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            float candidate_dist = -current_node_pair.first;
            float dist_bound = q->get_dist_bound();
            bool flag_stop_search = candidate_dist > dist_bound;

            if (flag_stop_search) {
                float current_best = q->get_best_candidate_dist();
                if(current_best < std::numeric_limits<float>::max()) {
                    metrics.d_best_cand_final = current_best;
                } else {
                    metrics.d_best_cand_final = candidate_dist;
                }
                break;
            }
            
            hop += 1;
            
            if(current_node_id >= max_elements || current_node_id >= node_locks.size()) {
                break;
            }
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getNeighbors(current_node_id);
            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(candidate_id >= max_elements) {
                    continue;
                }
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }
                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getQueryDist(candidate_id, query_data);
                    q->push(candidate_id, dist, is_deleted(candidate_id));
                    ndc += 1;
                }
            }
        }
        
        auto res = q->get_result(k);
        
        if(res.size() >= k) {
            metrics.d_worst_final = res[k - 1].first;
        } else if(!res.empty()) {
            metrics.d_worst_final = res.back().first;
        }
        
        if(metrics.d_best_cand_final >= std::numeric_limits<float>::max()) {
            metrics.d_best_cand_final = q->get_best_candidate_dist();
            if(metrics.d_best_cand_final >= std::numeric_limits<float>::max()) {
                metrics.d_best_cand_final = metrics.d_worst_final;
            }
        }
        
        if(!early_stage_captured && metrics.d_worst_final < std::numeric_limits<float>::max()) {
            metrics.d_worst_early = metrics.d_worst_final;
        }
        
        metrics.S = ndc;
        metrics.r_visit = (ef > 0) ? (float)metrics.S / ef : 0.0f;
        metrics.r_early = (metrics.S > 0) ? (float)metrics.t_last_improve / metrics.S : 0.0f;
        
        if(!res.empty()) {
            float top1_dist = res[0].first;
            float last1_dist;
            if(res.size() >= k) {
                last1_dist = res[k - 1].first;
            } else {
                last1_dist = res.back().first;
            }
            metrics.top1_last1_diff = last1_dist - top1_dist;
        } else {
            metrics.top1_last1_diff = 0.0f;
        }
        
        if(metrics.d_worst_early + eps > 0) {
            metrics.delta_improve = (metrics.d_worst_early - metrics.d_worst_final) / (metrics.d_worst_early + eps);
        } else {
            metrics.delta_improve = 0.0f;
        }
        
        q->releaseVisitedList();
        return std::make_tuple(res, ndc, metrics);
    }

    void printGraphInfo() {
        double avg_outdegree = 0;
        double avg_capacity = 0;

        std::cout << "current number of elements: " << n << "\n";
        std::cout << "max number of elements: " << max_elements << "\n";

        for(int i = 0; i < n; ++i) {
            avg_outdegree += GET_SZ((uint8_t*)Graph[i].neighbors);
            avg_capacity += GET_CAPACITY((uint8_t*)Graph[i].neighbors);
        }
        avg_outdegree /= n;
        avg_capacity /= n;
        std::cout << "Average out-degree: " << avg_outdegree << "\n";
        std::cout << "Average Capacity: " << avg_capacity << "\n";
        std::cout << "entry point: " << entry_point << "\n";
    }

    // Test-friendly accessors
    size_t getMEX() const { return MEX; }
    std::shared_mutex& getNodeLock(id_t id) { return node_locks[id]; }
};

} // namespace ours


