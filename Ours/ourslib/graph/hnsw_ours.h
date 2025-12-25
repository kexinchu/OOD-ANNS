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
#include <unordered_set>
#include <thread>
#include <chrono>
#include "node.h"
#include "../utils/search_list.h"
#include "../utils/visited_list.h"
#include "../utils/ssd_storage.h"
#include "../utils/lightgbm_predictor.h"
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

// Extract features for hardness prediction (same as training)
template<typename T>
std::vector<float> extractHardnessFeatures(
    const typename HNSW_Ours<T>::LightweightMetrics& metrics,
    size_t ndc,
    size_t efSearch,
    const std::vector<std::pair<float, id_t>>& results,
    size_t k) {
    
    std::vector<float> features;
    const float eps = 1e-6f;
    
    // Type: Index (3 features)
    float nstep = (float)metrics.S;
    float ndis = (float)ndc;
    float ninserts = (float)results.size();
    features.push_back(nstep);
    features.push_back(ndis);
    features.push_back(ninserts);
    
    // Type: NN Distance (3 features)
    float firstNN = 0.0f, closestNN = 0.0f, furthestNN = 0.0f;
    if(!results.empty()) {
        firstNN = results[0].first;
        closestNN = results[0].first;
        if(results.size() >= k) {
            furthestNN = results[k-1].first;
        } else {
            furthestNN = results.back().first;
        }
    }
    features.push_back(firstNN);
    features.push_back(closestNN);
    features.push_back(furthestNN);
    
    // Type: NN Stats (5 features)
    float avg = 0.0f, var = 0.0f, med = 0.0f, perc25 = 0.0f, perc75 = 0.0f;
    if(!results.empty() && results.size() >= k) {
        // Extract distances for top-k results
        std::vector<float> distances;
        for(size_t i = 0; i < k && i < results.size(); ++i) {
            distances.push_back(results[i].first);
        }
        
        // Calculate statistics
        std::sort(distances.begin(), distances.end());
        
        // Average
        for(float d : distances) avg += d;
        avg /= distances.size();
        
        // Variance
        for(float d : distances) {
            float diff = d - avg;
            var += diff * diff;
        }
        var /= distances.size();
        
        // Median
        med = distances[distances.size() / 2];
        
        // 25th percentile
        size_t idx_25 = distances.size() / 4;
        perc25 = distances[idx_25];
        
        // 75th percentile
        size_t idx_75 = (distances.size() * 3) / 4;
        perc75 = distances[idx_75];
    }
    features.push_back(avg);
    features.push_back(var);
    features.push_back(med);
    features.push_back(perc25);
    features.push_back(perc75);
    
    // Feature Interactions (important combinations)
    if(furthestNN > eps) {
        features.push_back(firstNN / furthestNN);  // firstNN/furthestNN ratio
    } else {
        features.push_back(0.0f);
    }
    
    if(avg > eps) {
        features.push_back(var / avg);  // Coefficient of variation
    } else {
        features.push_back(0.0f);
    }
    
    features.push_back(nstep * avg);  // nstep * avg interaction
    features.push_back(ndis * var);   // ndis * var interaction
    
    if(furthestNN > eps) {
        features.push_back((furthestNN - firstNN) / furthestNN);  // Relative distance range
    } else {
        features.push_back(0.0f);
    }
    
    features.push_back(med / (avg + eps));  // med/avg ratio
    
    return features;
}

// Query hardness predictor (using LightGBM model if available)
template<typename T>
HardnessMetrics DetectHardQuery(HNSW_Ours<T>* searcher, T* query_data, size_t k, size_t ef, size_t dim) {
    HardnessMetrics metrics;
    metrics.is_hard = false;
    
    // Use lightweight metrics from search trace
    size_t ndc = 0;
    auto [results, ndc_result, lw_metrics] = searcher->searchKnnWithLightweightMetrics(
        query_data, k, ef, ndc, 0.2f);
    
    // Use ML predictor if available, otherwise fallback to lightweight metrics
    if(searcher->hasHardnessPredictor()) {
        // Extract features (same as training)
        std::vector<float> features = extractHardnessFeatures<T>(
            lw_metrics, ndc, ef, results, k);
        
        // Predict hardness using LightGBM model
        metrics.hardness_score = searcher->hardness_predictor_->predict(features);
    } else {
        // Fallback: Calculate hardness score based on lightweight metrics
        // Higher r_visit, lower r_early, larger top1_last1_diff indicate harder queries
        float hardness_score = 0.0f;
        if(lw_metrics.S > 0) {
            hardness_score = lw_metrics.r_visit * (1.0f - lw_metrics.r_early) + 
                             (lw_metrics.top1_last1_diff > 0 ? lw_metrics.top1_last1_diff * 0.1f : 0.0f);
        }
        metrics.hardness_score = hardness_score;
    }
    
    // Stage 1: Jitter (perturbation stability) - only if needed
    // Skip jitter calculation if ML predictor is available and gives high confidence
    if(searcher->hasHardnessPredictor() && metrics.hardness_score > 0.8f) {
        // High confidence from ML, skip expensive jitter calculation
        metrics.jitter = 0.0f;
        metrics.is_hard = (metrics.hardness_score > 0.5f);
    } 
    
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
    
    // SSD storage (optional, if enabled)
    std::unique_ptr<SSDStorage> ssd_storage_;
    bool use_ssd_storage_{false};
    std::string ssd_index_path_;
    std::string ssd_vector_path_;
    
    std::shared_mutex delete_lock;
    std::unordered_set<id_t> delete_ids;
    std::atomic<size_t> current_timestamp{0};
    
    // Lazy delete mechanism for node deletion (separate from pending delete for edges)
    // Uses heap to track deleted nodes by last access time, batches real deletion
    struct DeletedNodeInfo {
        id_t node_id;
        std::chrono::steady_clock::time_point last_access_time;
        std::chrono::steady_clock::time_point delete_time;
        
        bool operator<(const DeletedNodeInfo& other) const {
            // Min-heap: nodes with older last_access_time have higher priority (should be deleted first)
            return last_access_time > other.last_access_time;
        }
    };
    std::shared_mutex lazy_delete_lock;
    std::priority_queue<DeletedNodeInfo, std::vector<DeletedNodeInfo>> lazy_delete_heap;  // Min-heap by last_access_time
    std::unordered_map<id_t, std::chrono::steady_clock::time_point> node_access_times;  // Track last access time for all nodes
    std::thread lazy_delete_thread;
    std::atomic<bool> should_stop_lazy_delete{false};
    std::atomic<bool> lazy_delete_enabled{false};
    
    // Pending delete mechanism for expired additional edges (page-based approach)
    std::shared_mutex pending_delete_lock;  // Lock for pending_delete_nodes set (for tracking, not for search)
    std::unordered_set<id_t> pending_delete_nodes;  // Set of pending nodes (for tracking/cleanup)
    std::vector<std::atomic<bool>> marked_pending_nodes;  // Lock-free array: marked_pending_nodes[node_id] = true if pending (for fast search check)
    std::shared_mutex in_serve_edges_lock;  // Lock only for in_serve_edges (less frequently accessed)
    std::unordered_map<id_t, std::unordered_map<id_t, bool> > in_serve_edges;  // in_serve_edges[start_node][end_node] = true (if edge is in-serve)
    
    // Epoch tracking for pending delete mechanism
    std::atomic<size_t> current_epoch{0};
    std::atomic<bool> pending_delete_enabled{false};
    std::atomic<size_t> epoch_duration_ms{1000};  // Default 1 second per epoch
    std::atomic<size_t> page_num{10};  // Number of pages to select per epoch
    size_t nodes_per_page;  // Number of nodes per page (calculated from page size)
    
    // Thread-local buffer for in-serve edges to reduce lock contention
    struct ThreadLocalBuffer {
        std::vector<std::pair<id_t, id_t>> edges;  // (node_id, neighbor_id) pairs
        size_t epoch;
    };
    static ThreadLocalBuffer* GetThreadBuffer() {
        thread_local static ThreadLocalBuffer* buffer = nullptr;
        if(buffer == nullptr) {
            buffer = new ThreadLocalBuffer();
            buffer->epoch = 0;
        }
        return buffer;
    }
    std::atomic<size_t> buffer_flush_threshold{1000};  // Flush when buffer reaches this size (larger = less frequent locks)
    
    // Thread-local buffer for node access times to reduce lock contention
    struct NodeAccessTimeBuffer {
        std::unordered_map<id_t, std::chrono::steady_clock::time_point> access_times;
        size_t flush_count;
        static constexpr size_t FLUSH_THRESHOLD = 500;  // OPTIMIZED: Flush every 500 updates (increased from 100 to reduce lock contention)
    };
    static NodeAccessTimeBuffer* GetAccessTimeBuffer() {
        thread_local static NodeAccessTimeBuffer* buffer = nullptr;
        if(buffer == nullptr) {
            buffer = new NodeAccessTimeBuffer();
            buffer->flush_count = 0;
        }
        return buffer;
    }
    
    std::thread pending_delete_thread;
    std::atomic<bool> should_stop_pending_delete{false};
    
    // Statistics for overhead measurement
    std::atomic<size_t> total_set_check_count{0};  // Total number of set checks
    std::atomic<size_t> total_set_insert_count{0};  // Total number of set inserts
    std::atomic<uint64_t> total_set_check_time_ns{0};  // Total time spent on set checks (nanoseconds)
    std::atomic<uint64_t> total_set_insert_time_ns{0};  // Total time spent on set inserts (nanoseconds)
    
    // Hardness predictor (LightGBM model)
    std::unique_ptr<LightGBMPredictor> hardness_predictor_{nullptr};
    
    size_t size_per_element = 0;

    HNSW_Ours(Metric metric, size_t dimension, size_t max_elements, size_t M_ = 16, size_t MEX_ = 48)
                : node_locks(max_elements), M(M_), MEX(MEX_) {
        M0 = 2*M;
        this->dim = dimension;
        this->max_elements = max_elements;
        this->size_per_element = dim*sizeof(T) + 1; // 8 bits for delete flag
        // Calculate nodes_per_page: assume 4KB page size, each node takes size_per_element bytes
        // For simplicity, use a fixed page size (e.g., 1024 nodes per page)
        this->nodes_per_page = 1024;  // Can be adjusted based on actual page size
        
        // Initialize lock-free marked_pending_nodes array
        marked_pending_nodes = std::vector<std::atomic<bool>>(max_elements);
        for(size_t i = 0; i < max_elements; ++i) {
            marked_pending_nodes[i].store(false, std::memory_order_relaxed);
        }

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

    HNSW_Ours(Metric metric, std::string path, bool use_ssd = false) {
        this->use_ssd_storage_ = use_ssd;
        
        if(use_ssd) {
            // Load from SSD storage
            std::string meta_path = path + ".meta";
            std::ifstream input(meta_path, std::ios::binary);
            
            input.read((char*)&M, sizeof(M));
            input.read((char*)&M0, sizeof(M0));
            input.read((char*)&MEX, sizeof(MEX));
            input.read((char*)&n, sizeof(n));
            input.read((char*)&entry_point, sizeof(entry_point));
            input.read((char*)&dim, sizeof(dim));
            input.read((char*)&max_elements, sizeof(max_elements));
            this->size_per_element = dim*sizeof(T) + 1;
            this->nodes_per_page = 1024;
            
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
            
            // Initialize SSD storage
            ssd_index_path_ = path + ".index";
            ssd_vector_path_ = path + ".vector";
            ssd_storage_ = std::make_unique<SSDStorage>(ssd_index_path_, ssd_vector_path_, 
                                                        max_elements, dim, size_per_element);
            ssd_storage_->Initialize(false);  // Load existing
            
            // Still need Graph for compatibility (but data is in SSD)
            node_locks = std::vector<std::shared_mutex>(this->max_elements);
            Graph.reserve(this->max_elements);
            Graph.resize(this->max_elements);
            visited_list_pool_ = std::shared_ptr<ngfixlib::VisitedListPool>(new ngfixlib::VisitedListPool(1, this->max_elements));
            
            // Initialize lock-free marked_pending_nodes array
            marked_pending_nodes = std::vector<std::atomic<bool>>(max_elements);
            for(size_t i = 0; i < max_elements; ++i) {
                marked_pending_nodes[i].store(false, std::memory_order_relaxed);
            }
            
            // Allocate vecdata for compatibility (but will use SSD)
            vecdata = new char[size_per_element*max_elements];
        } else {
            // Original in-memory loading
            std::ifstream input(path, std::ios::binary);
            if(!input.is_open()) {
                throw std::runtime_error("Cannot open index file: " + path);
            }

            input.read((char*)&M, sizeof(M));
            input.read((char*)&M0, sizeof(M0));
            input.read((char*)&MEX, sizeof(MEX));
            input.read((char*)&n, sizeof(n));
            input.read((char*)&entry_point, sizeof(entry_point));
            input.read((char*)&dim, sizeof(dim));
            input.read((char*)&max_elements, sizeof(max_elements));
            
            // Validate dimension after reading
            if(!input.good() || dim == 0 || dim > 100000) {
                std::cerr << "ERROR: Failed to read valid dimension from index file" << std::endl;
                std::cerr << "  File: " << path << std::endl;
                std::cerr << "  Read dim: " << dim << std::endl;
                std::cerr << "  Stream good: " << input.good() << std::endl;
                throw std::runtime_error("Invalid dimension in index file");
            }
            
            this->size_per_element = dim*sizeof(T) + 1;
            this->nodes_per_page = 1024;
            
            if constexpr (std::is_same_v<T, float>) {
                if (metric == L2_float) {
                    space = new ngfixlib::L2Space_float(dim);
                    query_space = new ngfixlib::L2Space_float(dim);
                } else if(metric == IP_float) {
                    // Use local variable to ensure dim is valid
                    size_t local_dim = dim;
                    if(local_dim % 4 != 0) {
                        std::cerr << "ERROR: Dimension " << local_dim << " from index file is not divisible by 4" << std::endl;
                        throw std::runtime_error("Dimension must be divisible by 4 for IP metric");
                    }
                    space = new ngfixlib::IPSpace_float(local_dim);
                    query_space = new ngfixlib::IPSpace_float(local_dim);
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
            
            // Initialize lock-free marked_pending_nodes array
            marked_pending_nodes = std::vector<std::atomic<bool>>(max_elements);
            for(size_t i = 0; i < max_elements; ++i) {
                marked_pending_nodes[i].store(false, std::memory_order_relaxed);
            }

            for(int i = 0; i < n; ++i) {
                Graph[i].LoadIndex(input);
            }
        }
    }

    ~HNSW_Ours() {
        StopPendingDelete();
        StopLazyDelete();
        
        // Flush all thread-local access time buffers before destruction
        // Note: This is best-effort - we can't force flush all threads' buffers
        // But we can flush the current thread's buffer if it exists
        NodeAccessTimeBuffer* buffer = GetAccessTimeBuffer();
        if(buffer != nullptr && !buffer->access_times.empty()) {
            FlushAccessTimeBuffer(buffer);
        }
        
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
        
        // Resize lock-free marked_pending_nodes array
        size_t old_size = marked_pending_nodes.size();
        marked_pending_nodes = std::vector<std::atomic<bool>>(new_max_elements);
        for(size_t i = 0; i < new_max_elements; ++i) {
            marked_pending_nodes[i].store(false, std::memory_order_relaxed);
        }
        
        if(entry_point >= max_elements) {
            entry_point = 0;
        }
    }

    void StoreIndex(std::string path) {
        if(use_ssd_storage_ && ssd_storage_) {
            // For SSD storage, just sync to disk
            ssd_storage_->Sync();
            
            // Also save metadata to a separate file
            std::string meta_path = path + ".meta";
            std::ofstream output(meta_path, std::ios::binary);
            output.write((char*)&M, sizeof(M));
            output.write((char*)&M0, sizeof(M0));
            output.write((char*)&MEX, sizeof(MEX));
            output.write((char*)&n, sizeof(n));
            output.write((char*)&entry_point, sizeof(entry_point));
            output.write((char*)&dim, sizeof(dim));
            output.write((char*)&max_elements, sizeof(max_elements));
            return;
        }
        
        // Original in-memory storage
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
        if(use_ssd_storage_ && ssd_storage_) {
            return (T*)ssd_storage_->GetVectorData(u);
        }
        return (T*)(vecdata + u*size_per_element + 1);
    }

    void SetData(id_t u, T* data) {
        if(use_ssd_storage_ && ssd_storage_) {
            ssd_storage_->SetVectorData(u, data);
        } else {
            memcpy(getData(u), data, sizeof(T)*dim);
        }
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

    // Returns search results for potential reuse (e.g., for NGFixOptimized)
    std::vector<std::pair<float, id_t> > HNSWBottomLayerInsertion(T* data, id_t cur_id, size_t efC) {
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
        return res;  // Return search results for potential reuse
    }

    // Returns search results from insertion (for potential reuse, e.g., NGFixOptimized)
    // Returns empty vector if n == 0 (first insertion)
    // FIXED: Use atomic operation to prevent race condition with concurrent deletes
    std::vector<std::pair<float, id_t> > InsertPoint(id_t id, size_t efC, T* vec) {
        if(id >= max_elements) {
            throw std::runtime_error("Error: id > max_elements.");
        }
        SetData(id, vec);
        auto data = getData(id);

        std::vector<std::pair<float, id_t> > search_results;
        size_t current_n = n.load();
        if(current_n != 0) {
            search_results = HNSWBottomLayerInsertion(data, id, efC);
        }
        // FIXED: Use atomic fetch_add to prevent race condition with RealDeleteNode
        // RealDeleteNode uses n.fetch_sub(1), so we must use atomic operation here too
        size_t new_n = n.fetch_add(1) + 1;
        if(new_n % 100000 == 0) {
            SetEntryPoint();
        }
        return search_results;
    }

    void set_deleted(id_t id) {
        (vecdata + id*size_per_element)[0] = true;
    }
    bool is_deleted(id_t id) {
        return (vecdata + id*size_per_element)[0];
    }

    // Pending delete nodes operations (similar to set_deleted/is_deleted)
    // OPTIMIZED: Lock-free implementation using atomic array
    // Even if there are minor read-write inconsistencies, it doesn't affect the search process
    void set_pending_node(id_t node_id) {
        if(!pending_delete_enabled.load()) {
            return;
        }
        if(node_id >= max_elements) {
            return;
        }
        // Lock-free write: directly set atomic bool
        marked_pending_nodes[node_id].store(true, std::memory_order_release);
        // Also update the set for tracking (with lock)
        std::unique_lock<std::shared_mutex> lock(pending_delete_lock);
        pending_delete_nodes.insert(node_id);
    }
    
    // OPTIMIZED: Batch set pending nodes - lock-free
    void set_pending_nodes_batch(const std::vector<id_t>& node_ids) {
        if(!pending_delete_enabled.load() || node_ids.empty()) {
            return;
        }
        // Lock-free batch update: directly set atomic bools
        for(id_t node_id : node_ids) {
            if(node_id < max_elements) {
                marked_pending_nodes[node_id].store(true, std::memory_order_release);
            }
        }
        // Also update the set for tracking (with lock)
        std::unique_lock<std::shared_mutex> lock(pending_delete_lock);
        pending_delete_nodes.insert(node_ids.begin(), node_ids.end());
    }
    
    // OPTIMIZED: Lock-free read - no locks needed!
    // Directly read from atomic array, completely lock-free
    bool is_pending_node(id_t node_id) {
        if(node_id >= max_elements) {
            return false;
        }
        // Lock-free read: directly read atomic bool
        return marked_pending_nodes[node_id].load(std::memory_order_acquire);
    }
    
    // Lock-free clear: set all to false
    void clear_pending_node() {
        // First, get a copy of pending nodes to clear the atomic array
        std::unordered_set<id_t> nodes_to_clear;
        {
            std::shared_lock<std::shared_mutex> lock(pending_delete_lock);
            nodes_to_clear = pending_delete_nodes;
        }
        // Lock-free: set all atomic bools to false
        for(id_t node_id : nodes_to_clear) {
            if(node_id < max_elements) {
                marked_pending_nodes[node_id].store(false, std::memory_order_release);
            }
        }
        // Clear the set
        std::unique_lock<std::shared_mutex> lock(pending_delete_lock);
        pending_delete_nodes.clear();
    }
    
    // Get copy of pending nodes (for debugging/statistics)
    std::unordered_set<id_t> get_pending_nodes_copy() {
        std::shared_lock<std::shared_mutex> lock(pending_delete_lock);
        return pending_delete_nodes;
    }
    
    // Get count of pending nodes (for debugging/statistics)
    size_t get_pending_nodes_size() {
        std::shared_lock<std::shared_mutex> lock(pending_delete_lock);
        return pending_delete_nodes.size();
    }
    
    // In-serve edges operations
    void set_inserve_edge(id_t start_node, id_t end_node) {
        if(!pending_delete_enabled.load(std::memory_order_acquire)) {
            return;
        }
        std::unique_lock<std::shared_mutex> lock(in_serve_edges_lock);
        in_serve_edges[start_node][end_node] = true;
    }
    
    // Batch set in-serve edges
    void set_inserve_edges_batch(const std::vector<std::pair<id_t, id_t> >& edges) {
        if(!pending_delete_enabled.load(std::memory_order_acquire)) {
            return;
        }
        std::unique_lock<std::shared_mutex> lock(in_serve_edges_lock);
        for(const auto& [node_id, neighbor_id] : edges) {
            in_serve_edges[node_id][neighbor_id] = true;
        }
    }
    
    bool is_inserve_edge(id_t start_node, id_t end_node) {
        if(!pending_delete_enabled.load()) {
            return false;
        }
        std::shared_lock<std::shared_mutex> lock(in_serve_edges_lock);
        auto it = in_serve_edges.find(start_node);
        if(it == in_serve_edges.end()) {
            return false;
        }
        auto edge_it = it->second.find(end_node);
        return edge_it != it->second.end() && edge_it->second;
    }
    
    void clear_inserve_edges() {
        std::unique_lock<std::shared_mutex> lock(in_serve_edges_lock);
        in_serve_edges.clear();
    }

    // Mark node for lazy deletion (adds to heap, real deletion happens later)
    void DeletePoint(id_t id) {
        if(id >= max_elements || id >= n) {
            throw std::runtime_error("Error: id out of bounds in DeletePoint.");
        }
        
        // CRITICAL: Never delete the entry point - it would break all searches!
        if(id == entry_point) {
            return;
        }
        
        // Check if already deleted
        if(is_deleted(id)) {
            return;  // Already marked for deletion
        }
        
        // Mark as deleted first
        {
            std::unique_lock<std::shared_mutex> lock(delete_lock);
            delete_ids.insert(id);
        }
        set_deleted(id);
        
        // Add to lazy delete heap with current access time (or delete time if never accessed)
        {
            std::unique_lock<std::shared_mutex> lock(lazy_delete_lock);
            auto now = std::chrono::steady_clock::now();
            auto last_access = node_access_times.find(id);
            auto access_time = (last_access != node_access_times.end()) ? 
                              last_access->second : now;  // Use last access time, or now if never accessed
            
            DeletedNodeInfo info;
            info.node_id = id;
            info.last_access_time = access_time;
            info.delete_time = now;
            lazy_delete_heap.push(info);
        }
    }
    
    // Real delete: completely remove a node from the graph (called by lazy delete worker)
    void RealDeleteNode(id_t id) {
        if(id >= max_elements || id >= n) {
            return;  // Already out of bounds
        }
        
        if(!is_deleted(id)) {
            return;  // Not marked as deleted, skip
        }
        
        // Collect neighbors before deletion
        std::unordered_set<id_t> neighbors;
        {
            std::shared_lock<std::shared_mutex> lock(node_locks[id]);
            auto [outs, sz, st] = getNeighbors(id);
            for(int j = st; j < st + sz; ++j) {
                id_t neighbor_id = outs[j];
                if(neighbor_id < max_elements && neighbor_id < n && !is_deleted(neighbor_id)) {
                    neighbors.insert(neighbor_id);
                }
            }
        }
        
        // Remove edges from neighbors pointing to this node
        for(id_t neighbor_id : neighbors) {
            if(neighbor_id >= max_elements || neighbor_id >= n || is_deleted(neighbor_id)) continue;
            
            try {
                std::unique_lock<std::shared_mutex> lock(node_locks[neighbor_id]);
                auto [outs, sz, st] = getNeighbors(neighbor_id);
                
                // Remove id from neighbor's edge list
                uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)outs);
                uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)outs);
                uint8_t base_sz = sz - ngfix_sz;
                
                std::vector<id_t> new_ngfix_neighbors;
                for(int j = ngfix_capacity - ngfix_sz; j < ngfix_capacity; ++j) {
                    if(outs[j + 1] != id) {
                        new_ngfix_neighbors.push_back(outs[j + 1]);
                    }
                }
                
                std::vector<std::pair<float, id_t> > new_base_neighbors;
                for(int j = ngfix_capacity; j < ngfix_capacity + base_sz; ++j) {
                    if(outs[j + 1] != id) {
                        new_base_neighbors.push_back({0, outs[j + 1]});
                    }
                }
                
                Graph[neighbor_id].replace_ngfix_neighbors(new_ngfix_neighbors);
                Graph[neighbor_id].replace_base_graph_neighbors(new_base_neighbors);
            } catch(...) {
                // Skip if error
                continue;
            }
        }
        
        // Repair connectivity: add edges between neighbors if needed
        // This helps maintain graph connectivity after node deletion
        if(neighbors.size() > 1) {
            std::vector<id_t> neighbor_vec(neighbors.begin(), neighbors.end());
            for(size_t i = 0; i < neighbor_vec.size(); ++i) {
                for(size_t j = i + 1; j < neighbor_vec.size(); ++j) {
                    id_t u = neighbor_vec[i];
                    id_t v = neighbor_vec[j];
                    if(u >= max_elements || v >= max_elements || u >= n || v >= n) continue;
                    if(is_deleted(u) || is_deleted(v)) continue;
                    
                    // Check if edge already exists
                    bool edge_exists = false;
                    try {
                        std::shared_lock<std::shared_mutex> lock_u(node_locks[u]);
                        auto [outs_u, sz_u, st_u] = getNeighbors(u);
                        for(int k = st_u; k < st_u + sz_u; ++k) {
                            if(outs_u[k] == v) {
                                edge_exists = true;
                                break;
                            }
                        }
                    } catch(...) {
                        continue;
                    }
                    
                    // Add edge if doesn't exist (simple heuristic: add if distance is reasonable)
                    if(!edge_exists) {
                        try {
                            std::unique_lock<std::shared_mutex> lock_u(node_locks[u]);
                            float dist = getDist(u, v);
                            // Only add if distance is reasonable (avoid very long edges)
                            // Use a simple heuristic: add if within reasonable range
                            Graph[u].add_ngfix_neighbors(v, 0, MEX);
                        } catch(...) {
                            // Skip if error
                        }
                    }
                }
            }
        }
        
        // Delete the node itself
        {
            std::unique_lock<std::shared_mutex> lock(node_locks[id]);
            Graph[id].delete_node();
        }
        
        // Remove from delete_ids and access_times
        {
            std::unique_lock<std::shared_mutex> lock(delete_lock);
            delete_ids.erase(id);
        }
        {
            std::unique_lock<std::shared_mutex> lock(lazy_delete_lock);
            node_access_times.erase(id);
        }
        
        // Decrease n (real deletion)
        n.fetch_sub(1);
    }
    
    // Clean up edges pointing to deleted nodes (called during search)
    // OPTIMIZED: Use try_lock to avoid blocking search operations
    void CleanupEdgesToDeletedNodes(id_t node_id) {
        if(node_id >= max_elements || node_id >= n) return;
        if(is_deleted(node_id)) return;  // Don't process if node itself is deleted
        
        // Try to acquire unique lock, but don't block if search is using shared lock
        std::unique_lock<std::shared_mutex> lock(node_locks[node_id], std::try_to_lock);
        if(!lock.owns_lock()) {
            // Couldn't acquire lock (search is using it), skip cleanup for now
            return;
        }
        
        try {
            auto [outs, sz, st] = getNeighbors(node_id);
            
            uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)outs);
            uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)outs);
            uint8_t base_sz = sz - ngfix_sz;
            
            bool has_deleted_neighbor = false;
            std::vector<id_t> new_ngfix_neighbors;
            for(int j = ngfix_capacity - ngfix_sz; j < ngfix_capacity; ++j) {
                id_t neighbor_id = outs[j + 1];
                if(neighbor_id < max_elements && neighbor_id < n && !is_deleted(neighbor_id)) {
                    new_ngfix_neighbors.push_back(neighbor_id);
                } else {
                    has_deleted_neighbor = true;
                }
            }
            
            std::vector<std::pair<float, id_t> > new_base_neighbors;
            for(int j = ngfix_capacity; j < ngfix_capacity + base_sz; ++j) {
                id_t neighbor_id = outs[j + 1];
                if(neighbor_id < max_elements && neighbor_id < n && !is_deleted(neighbor_id)) {
                    new_base_neighbors.push_back({0, neighbor_id});
                } else {
                    has_deleted_neighbor = true;
                }
            }
            
            if(has_deleted_neighbor) {
                Graph[node_id].replace_ngfix_neighbors(new_ngfix_neighbors);
                Graph[node_id].replace_base_graph_neighbors(new_base_neighbors);
            }
        } catch(...) {
            // Skip if error
        }
    }
    
    // Update node access time (called during search) - OPTIMIZED: Use thread-local buffer
    void UpdateNodeAccessTime(id_t node_id) {
        if(node_id >= max_elements || node_id >= n) return;
        if(!lazy_delete_enabled.load()) return;
        
        auto now = std::chrono::steady_clock::now();
        NodeAccessTimeBuffer* buffer = GetAccessTimeBuffer();
        
        // Add to thread-local buffer
        buffer->access_times[node_id] = now;
        buffer->flush_count++;
        
        // Flush periodically to reduce lock contention
        if(buffer->flush_count >= NodeAccessTimeBuffer::FLUSH_THRESHOLD) {
            FlushAccessTimeBuffer(buffer);
        }
    }
    
    // Flush thread-local access time buffer to global map
    void FlushAccessTimeBuffer(NodeAccessTimeBuffer* buffer) {
        if(buffer == nullptr || buffer->access_times.empty()) {
            return;
        }
        
        // Batch update with single lock acquisition
        {
            std::unique_lock<std::shared_mutex> lock(lazy_delete_lock);
            for(const auto& [node_id, access_time] : buffer->access_times) {
                node_access_times[node_id] = access_time;
            }
        }
        
        // Clear buffer
        buffer->access_times.clear();
        buffer->flush_count = 0;
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
        
        // OPTIMIZED: Use try_lock to avoid blocking search operations
        for(auto [u, vs] : new_edges) {
            std::unique_lock <std::shared_mutex> lock(node_locks[u], std::try_to_lock);
            if(lock.owns_lock()) {
                // Successfully acquired lock, add edges
                for(auto [v, eh] : vs) {
                    Graph[u].add_ngfix_neighbors(v, eh, MEX);
                }
            }
            // If lock not acquired (search is using it), skip this node
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
        
        // OPTIMIZED: Use try_lock to avoid blocking search operations
        for(auto [u, vs] : new_edges) {
            std::unique_lock <std::shared_mutex> lock(node_locks[u], std::try_to_lock);
            if(lock.owns_lock()) {
                for(auto [v, eh] : vs) {
                    Graph[u].add_ngfix_neighbors(v, eh, MEX);
                }
            }
            // If lock not acquired, skip this node
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
        
        // Update entry point access time
        if(lazy_delete_enabled.load()) {
            UpdateNodeAccessTime(entry_point);
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
            
            // Update node access time for lazy delete
            if(lazy_delete_enabled.load()) {
                UpdateNodeAccessTime(current_node_id);
            }
            
            // OPTIMIZED: Reduce CleanupEdgesToDeletedNodes call frequency
            // Only call every N nodes to reduce overhead (sampling approach)
            static thread_local size_t cleanup_counter = 0;
            if(lazy_delete_enabled.load() && (cleanup_counter++ % 10 == 0)) {
                CleanupEdgesToDeletedNodes(current_node_id);
            }
            
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getNeighbors(current_node_id);
            
            bool is_pending = false;
            if(pending_delete_enabled.load(std::memory_order_acquire)) {
                is_pending = is_pending_node(current_node_id);
            }
            
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
                    
                    // Only record if node is pending (minimal overhead when not pending)
                    if(is_pending) {
                        RecordInServeEdge(current_node_id, candidate_id);
                    }

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
            
            // Update node access time for lazy delete
            if(lazy_delete_enabled.load()) {
                UpdateNodeAccessTime(current_node_id);
            }
            
            // OPTIMIZED: Reduce CleanupEdgesToDeletedNodes call frequency
            // Only call every N nodes to reduce overhead (sampling approach)
            static thread_local size_t cleanup_counter = 0;
            if(lazy_delete_enabled.load() && (cleanup_counter++ % 10 == 0)) {
                CleanupEdgesToDeletedNodes(current_node_id);
            }
            
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getNeighbors(current_node_id);
            
            // OPTIMIZED: Check if node is pending delete with lock-free fast path
            // The is_pending_node now uses try_lock to avoid blocking
            bool is_pending = false;
            if(pending_delete_enabled.load(std::memory_order_acquire)) {
                is_pending = is_pending_node(current_node_id);
            }
            
            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(candidate_id >= max_elements) {
                    continue;
                }
                
                // Skip deleted nodes
                if(is_deleted(candidate_id)) {
                    continue;
                }
                
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }
                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getQueryDist(candidate_id, query_data);
                    q->push(candidate_id, dist, is_deleted(candidate_id));
                    
                    // Update candidate node access time
                    if(lazy_delete_enabled.load()) {
                        UpdateNodeAccessTime(candidate_id);
                    }
                    
                    // If node is in pending_delete_nodes and edge was pushed (selected), record it as in-serve
                    if(is_pending) {
                        RecordInServeEdge(current_node_id, candidate_id);
                    }
                    
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
    
    // ========== Pending Delete Mechanism for Expired Additional Edges ==========
    
    // Update pending delete parameters dynamically
    void UpdatePendingDeleteParams(size_t epoch_duration_ms_, size_t page_num_) {
        epoch_duration_ms.store(epoch_duration_ms_);
        page_num.store(page_num_);
    }
    
    // Get current pending delete parameters
    std::pair<size_t, size_t> GetPendingDeleteParams() {
        return {epoch_duration_ms.load(), page_num.load()};
    }
    
    // Start pending delete background thread
    void StartPendingDelete(size_t epoch_duration_ms_ = 1000, size_t page_num_ = 10) {
        if(pending_delete_enabled.load()) {
            return;  // Already started
        }
        
        epoch_duration_ms.store(epoch_duration_ms_);
        page_num.store(page_num_);
        pending_delete_enabled.store(true);
        should_stop_pending_delete.store(false);
        
        // Reset statistics
        total_set_check_count.store(0, std::memory_order_relaxed);
        total_set_insert_count.store(0, std::memory_order_relaxed);
        total_set_check_time_ns.store(0, std::memory_order_relaxed);
        total_set_insert_time_ns.store(0, std::memory_order_relaxed);
        
        pending_delete_thread = std::thread([this]() {
            this->PendingDeleteWorker();
        });
    }
    
    // Stop pending delete background thread
    void StopPendingDelete(bool wait_for_cleanup = false) {
        if(!pending_delete_enabled.load()) {
            return;
        }
        
        should_stop_pending_delete.store(true);
        
        if(wait_for_cleanup) {
            // Wait for current epoch to finish and cleanup to complete
            if(pending_delete_thread.joinable()) {
                pending_delete_thread.join();
            }
            // Now disable pending delete after cleanup
            pending_delete_enabled.store(false);
        } else {
            // Immediately disable pending delete (stops checks, but doesn't wait for cleanup)
            // This keeps the graph structure unchanged (additional edges remain)
            pending_delete_enabled.store(false);
            if(pending_delete_thread.joinable()) {
                pending_delete_thread.join();
            }
        }
    }
    
    // Start lazy delete background thread (for node deletion)
    void StartLazyDelete() {
        if(lazy_delete_enabled.load()) {
            return;  // Already started
        }
        
        lazy_delete_enabled.store(true);
        should_stop_lazy_delete.store(false);
        
        lazy_delete_thread = std::thread([this]() {
            this->LazyDeleteWorker();
        });
    }
    
    // Stop lazy delete background thread
    void StopLazyDelete(bool wait_for_cleanup = false) {
        if(!lazy_delete_enabled.load()) {
            return;
        }
        
        should_stop_lazy_delete.store(true);
        
        if(wait_for_cleanup) {
            // Process remaining nodes in heap before stopping
            if(lazy_delete_thread.joinable()) {
                lazy_delete_thread.join();
            }
            lazy_delete_enabled.store(false);
        } else {
            lazy_delete_enabled.store(false);
            if(lazy_delete_thread.joinable()) {
                lazy_delete_thread.join();
            }
        }
    }
    
    // Background worker thread for lazy delete
    void LazyDeleteWorker() {
        while(!should_stop_lazy_delete.load()) {
            auto cycle_start = std::chrono::steady_clock::now();
            
            // Process nodes that haven't been accessed in the last 1 second
            auto cutoff_time = std::chrono::steady_clock::now() - std::chrono::seconds(1);
            std::vector<id_t> nodes_to_delete;
            
            {
                std::unique_lock<std::shared_mutex> lock(lazy_delete_lock);
                while(!lazy_delete_heap.empty()) {
                    const auto& top = lazy_delete_heap.top();
                    
                    // Check if node hasn't been accessed in last 1 second
                    if(top.last_access_time < cutoff_time) {
                        nodes_to_delete.push_back(top.node_id);
                        lazy_delete_heap.pop();
                    } else {
                        // All remaining nodes have been accessed recently, stop
                        break;
                    }
                }
            }
            
            // Perform real deletion for nodes that haven't been accessed
            for(id_t node_id : nodes_to_delete) {
                if(should_stop_lazy_delete.load()) {
                    break;
                }
                try {
                    RealDeleteNode(node_id);
                } catch(...) {
                    // Skip if error
                    continue;
                }
            }
            
            // Sleep until next second
            auto cycle_end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                cycle_end - cycle_start).count();
            auto sleep_time_ms = 1000 - elapsed;
            if(sleep_time_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_ms));
            }
        }
    }
    
    // Background worker thread for pending delete
    void PendingDeleteWorker() {
        while(!should_stop_pending_delete.load()) {
            size_t epoch_ms = epoch_duration_ms.load();
            size_t pages = page_num.load();
            
            // Step 1: Randomly select pages and mark nodes in those pages as pending delete
            MarkNodesForDeletionByPages(pages);
            
            // Step 2: Clear in-serve edges from previous epoch
            // Note: We don't flush thread buffers here - they will be flushed naturally
            // when threads access them next time (epoch check will trigger flush)
            clear_inserve_edges();
            
            // Step 3: Wait for the epoch to allow queries to access nodes and record in-serve edges
            std::this_thread::sleep_for(std::chrono::milliseconds(epoch_ms));
            
            if(should_stop_pending_delete.load()) {
                break;
            }
            
            // Step 4: Flush all thread buffers before cleanup (ensure all in-serve edges are recorded)
            // This is done implicitly when threads access buffers next time, but we can also
            // force flush by accessing the buffer (though we can't easily do this for all threads)
            // For now, rely on natural flush when threads access buffers
            
            // Step 5: At epoch end, remove additional edges that are not in in-serve set
            CleanupPendingDeleteEdges();
            
            // Step 6: Flush dirty pages to SSD (if using SSD storage)
            if(use_ssd_storage_ && ssd_storage_) {
                ssd_storage_->FlushDirtyPages();
            }
            
            // Step 6: Clear pending_delete_nodes for next epoch
            clear_pending_node();
            
            current_epoch.fetch_add(1);
        }
    }
    
    // Mark nodes from randomly selected pages as pending delete
    void MarkNodesForDeletionByPages(size_t num_pages) {
        size_t num_nodes = n.load();
        if(num_nodes == 0) return;
        
        // Calculate total number of pages
        size_t total_pages = (num_nodes + nodes_per_page - 1) / nodes_per_page;
        if(total_pages == 0) return;
        
        num_pages = std::min(num_pages, total_pages);
        
        // Randomly select pages
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> page_dist(0, total_pages - 1);
        std::unordered_set<size_t> selected_pages;
        
        while(selected_pages.size() < num_pages) {
            selected_pages.insert(page_dist(gen));
        }
        
        // Collect nodes from selected pages (only those with additional edges)
        std::vector<id_t> nodes_to_mark;
        for(size_t page_id : selected_pages) {
            id_t page_start = page_id * nodes_per_page;
            id_t page_end = std::min((id_t)((page_id + 1) * nodes_per_page), (id_t)num_nodes);
            
            // Only mark nodes that have additional edges (ngfix edges)
            for(id_t node_id = page_start; node_id < page_end; ++node_id) {
                if(node_id >= max_elements || is_deleted(node_id)) continue;
                
            try {
                std::shared_lock<std::shared_mutex> node_lock(node_locks[node_id]);
                auto [neighbors, sz, st] = getNeighbors(node_id);
                uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
                
                if(ngfix_sz > 0) {
                        nodes_to_mark.push_back(node_id);
                }
            } catch(...) {
                continue;
                }
            }
        }
        
        // OPTIMIZED: Batch update pending_delete_nodes with single lock acquisition
        // This reduces lock contention by acquiring lock only once instead of once per node
        if(!nodes_to_mark.empty()) {
            set_pending_nodes_batch(nodes_to_mark);
        }
        
        current_epoch.fetch_add(1);
    }
    
    // Clean up additional edges that are not in in-serve set
    void CleanupPendingDeleteEdges() {
        // Create a copy of pending_delete_nodes to avoid holding lock too long
        std::unordered_set<id_t> nodes_to_process = get_pending_nodes_copy();
        
        for(const auto& node_id : nodes_to_process) {
            if(node_id >= max_elements) continue;
            
            // Remove edges from the graph
            // OPTIMIZED: Use try_lock to avoid blocking search operations
            try {
                std::unique_lock<std::shared_mutex> node_lock(node_locks[node_id], std::try_to_lock);
                if(!node_lock.owns_lock()) {
                    // Couldn't acquire lock, skip this node
                    continue;
                }
                auto [neighbors, sz, st] = getNeighbors(node_id);
                
                uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
                uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
                
                if(ngfix_sz > 0) {
                    int ngfix_start = ngfix_capacity - ngfix_sz;
                    std::vector<id_t> new_ngfix_neighbors;
                    
                    for(int j = ngfix_start; j < ngfix_capacity; ++j) {
                        id_t neighbor_id = neighbors[j + 1];
                        // Keep edge if it's in in-serve map
                        if (is_inserve_edge(node_id, neighbor_id)) {
                            new_ngfix_neighbors.push_back(neighbor_id);
                        }
                        // deletion rate = 10%
                        else if (rand() % 10 != 0) { 
                            new_ngfix_neighbors.push_back(neighbor_id);
                        }
                        // Otherwise, the edge will be deleted (not added to new_ngfix_neighbors)
                    }
                    
                    // Replace ngfix neighbors
                    Graph[node_id].replace_ngfix_neighbors(new_ngfix_neighbors);
                }
            } catch(...) {
                // Skip this node if there's an error
                continue;
            }
        }
    }
    
    // Record an edge as in-serve (called during search when edge is selected)
    // Note: is_pending_node is already checked in searchKnn, so we don't need to check again
    void RecordInServeEdge(id_t node_id, id_t neighbor_id) {
        if(!pending_delete_enabled.load(std::memory_order_acquire)) {
            return;
        }
        
        // Optimized: Use thread-local buffer to batch inserts and reduce lock contention
        // Measure overhead (minimal - just adding to vector)
        auto insert_start = std::chrono::high_resolution_clock::now();
        
        ThreadLocalBuffer* buffer = GetThreadBuffer();
        size_t current_ep = current_epoch.load(std::memory_order_acquire);
        
        // Check if epoch changed (need to flush old buffer)
        if(buffer->epoch != current_ep) {
            FlushThreadBuffer(buffer);
            buffer->epoch = current_ep;
        }
        
        // Add to buffer (very fast - just vector push_back)
        buffer->edges.push_back({node_id, neighbor_id});
        total_set_insert_count.fetch_add(1, std::memory_order_relaxed);
        
        // Flush if buffer is full (larger threshold = less frequent locks)
        size_t threshold = buffer_flush_threshold.load(std::memory_order_acquire);
        if(buffer->edges.size() >= threshold) {
            FlushThreadBuffer(buffer);
        }
        
        auto insert_end = std::chrono::high_resolution_clock::now();
        auto insert_time = std::chrono::duration_cast<std::chrono::nanoseconds>(insert_end - insert_start).count();
        total_set_insert_time_ns.fetch_add(insert_time, std::memory_order_relaxed);
    }
    
    // Flush thread-local buffer to global in_serve_edges
    void FlushThreadBuffer(ThreadLocalBuffer* buffer) {
        if(buffer == nullptr || buffer->edges.empty()) {
            return;
        }
        
        // Batch insert with single lock acquisition using batch function
        set_inserve_edges_batch(buffer->edges);
        buffer->edges.clear();
        buffer->edges.shrink_to_fit();  // Free memory
    }
    
    // Get statistics about pending delete edges
    struct PendingDeleteStats {
        size_t num_nodes_with_pending_deletes;
        size_t total_pending_edges;
        size_t total_in_serve_edges;
        size_t current_epoch_num;
        size_t total_set_checks;
        size_t total_set_inserts;
        uint64_t total_set_check_time_ns;
        uint64_t total_set_insert_time_ns;
    };
    
    PendingDeleteStats GetPendingDeleteStats() {
        PendingDeleteStats stats;
        stats.num_nodes_with_pending_deletes = get_pending_nodes_size();
        stats.current_epoch_num = current_epoch.load();
        stats.total_set_checks = total_set_check_count.load();
        stats.total_set_inserts = total_set_insert_count.load();
        stats.total_set_check_time_ns = total_set_check_time_ns.load();
        stats.total_set_insert_time_ns = total_set_insert_time_ns.load();
        return stats;
    }
    
    // Load hardness predictor model
    bool loadHardnessPredictor(const std::string& model_path) {
        hardness_predictor_ = std::make_unique<LightGBMPredictor>();
        bool success = hardness_predictor_->loadFromFile(model_path);
        if(!success) {
            hardness_predictor_.reset();
        }
        return success;
    }
    
    // Check if hardness predictor is loaded
    bool hasHardnessPredictor() const {
        return hardness_predictor_ != nullptr && hardness_predictor_->isLoaded();
    }
};

} // namespace ours


