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
#include <random>
#include <numeric>
#include <queue>

using namespace ngfixlib;

struct QueryHardnessMetrics {
    float recall;
    float escape_hardness;  // Escape Hardness in NGFix
    float frontier_churn;   // Stage 0: Frontier Churn (FC)
    float self_consistency; // Stage 1: Self-consistency difference (ef_base vs ef_extended)
    float self_consistency_2x; // Stage 1: Self-consistency difference (ef_base vs ef_2x)
    size_t ndc;            // Number of distance computations
    
    // New metrics
    float top_k_margin;           // Top-k Margin (MG)
    float top_k_cohesion;         // Top-k Cohesion (LCS, sampled)
    float signature_dissimilarity; // Set-Signature Dissimilarity
    
    // Additional metrics
    float jitter;                 // Micro-Perturbation Stability (Jitter J)
    float reachability_probe;     // τ-Reachability Probe on Visited Subgraph
    
    // Lightweight metrics from single search trace
    size_t S;                     // Total number of visited nodes (equals ndc)
    float r_visit;                // Visit Budget Usage: S / ef
    float r_early;                // Early-Convergence Ratio: t_last_improve / S
    float top1_last1_diff;        // Distance difference: last1_dist - top1_dist (in top-k results)
    float delta_improve;          // Early-vs-Final Improvement Ratio
};

// Calculate Escape Hardness for a query
// Uses top-N nodes from search results (not ground truth) to calculate hardness mean
float CalculateEscapeHardness(HNSW_NGFix<float>* searcher, float* query_data, int* gt, size_t k, size_t Nq = 100) {
    // First, search to get top-N nodes around the query
    size_t ndc = 0;
    auto search_results = searcher->searchKnn(query_data, Nq, 1500, ndc);
    
    // Extract top-N node IDs from search results
    size_t actual_Nq = std::min(Nq, search_results.size());
    if(actual_Nq == 0) {
        return 0.0f;
    }
    
    int* top_n_ids = new int[actual_Nq];
    for(size_t i = 0; i < actual_Nq; ++i) {
        top_n_ids[i] = search_results[i].second;
    }
    
    // Calculate hardness matrix for these top-N nodes
    auto H = searcher->CalculateHardness(top_n_ids, actual_Nq, 100, std::min((size_t)200, 2*actual_Nq));
    
    // Calculate escape hardness: MEAN of H[i][j] for i < Nq, j < Nq (excluding EH_INF)
    float escape_hardness_sum = 0.0f;
    int count = 0;
    for(size_t i = 0; i < actual_Nq && i < H.size(); ++i) {
        for(size_t j = 0; j < actual_Nq && j < H[i].size(); ++j) {
            if(H[i][j] != EH_INF) {
                escape_hardness_sum += H[i][j];
                count++;
            }
        }
    }
    
    delete[] top_n_ids;
    
    // Return the mean (average), not the sum
    if(count > 0) {
        return escape_hardness_sum / count;
    }
    return 0.0f;
}

// Calculate self-consistency: difference between two searches with different efSearch
float CalculateSelfConsistency(HNSW_NGFix<float>* searcher, float* query_data, size_t k, 
                               size_t ef_base, size_t ef_extended) {
    size_t ndc1 = 0, ndc2 = 0;
    
    // First search with ef_base
    auto result1 = searcher->searchKnn(query_data, k, ef_base, ndc1);
    
    // Second search with ef_extended
    auto result2 = searcher->searchKnn(query_data, k, ef_extended, ndc2);
    
    // Calculate Jaccard distance (1 - intersection/union)
    std::unordered_set<id_t> set1, set2;
    for(const auto& p : result1) {
        set1.insert(p.second);
    }
    for(const auto& p : result2) {
        set2.insert(p.second);
    }
    
    // Calculate intersection and union
    size_t intersection = 0;
    for(id_t id : set1) {
        if(set2.find(id) != set2.end()) {
            intersection++;
        }
    }
    
    size_t union_size = set1.size() + set2.size() - intersection;
    
    if(union_size == 0) {
        return 0.0f;
    }
    
    // Self-consistency difference: 1 - Jaccard similarity
    float jaccard_similarity = (float)intersection / union_size;
    return 1.0f - jaccard_similarity;
}

// Calculate recall
float CalculateRecall(const std::vector<std::pair<float, id_t> >& results, int* gt, size_t k) {
    std::unordered_set<id_t> gt_set;
    for(int i = 0; i < k; ++i) {
        gt_set.insert(gt[i]);
    }
    
    int acc = 0;
    for(const auto& p : results) {
        if(gt_set.find(p.second) != gt_set.end()) {
            acc++;
            gt_set.erase(p.second);
        }
    }
    
    return (float)acc / k;
}

// 7. Calculate Micro-Perturbation Stability (Jitter J)
// Query tweak: q' = normalize(q + ε·noise), ε ∈ [0.01, 0.05]
// Run at original budget b; J = 1 − |R_b(q) ∩ R_b(q')| / k
float CalculateJitter(HNSW_NGFix<float>* searcher, float* query_data, size_t k, 
                      size_t ef_base, size_t dim, float epsilon = 0.03f) {
    // Create perturbed query: q' = normalize(q + ε·noise)
    float* perturbed_query = new float[dim];
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
    auto result1 = searcher->searchKnn(query_data, k, ef_base, ndc1);
    auto result2 = searcher->searchKnn(perturbed_query, k, ef_base, ndc2);
    
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
    
    delete[] perturbed_query;
    
    // Jitter: J = 1 − |R_b(q) ∩ R_b(q')| / k
    if(k > 0) {
        return 1.0f - (float)intersection / k;
    }
    return 0.0f;
}

// 8. Calculate τ-Reachability Probe on Visited Subgraph
// On already visited candidates, pick K = 4-8 representatives
// Do a tiny layered-BFS / min-bottleneck probe; estimate low-threshold mutual reachability ratio PR
float CalculateReachabilityProbe(HNSW_NGFix<float>* searcher, 
                                 const std::vector<std::pair<float, id_t> >& visited_results,
                                 size_t k, size_t K_repr = 6) {
    if(visited_results.empty() || k == 0) {
        return 0.0f;
    }
    
    size_t actual_k = std::min(k, visited_results.size());
    if(actual_k < 2) {
        return 0.0f;
    }
    
    // Pick K_repr representatives from visited results (evenly spaced)
    std::vector<id_t> representatives;
    size_t step = std::max((size_t)1, actual_k / K_repr);
    for(size_t i = 0; i < actual_k && representatives.size() < K_repr; i += step) {
        representatives.push_back(visited_results[i].second);
    }
    
    if(representatives.size() < 2) {
        return 0.0f;
    }
    
    // Calculate pairwise reachability: check if representatives can reach each other
    // through the graph within a small number of hops (τ = 2-3)
    size_t reachable_pairs = 0;
    size_t total_pairs = 0;
    size_t tau = 2;  // max hops for reachability
    
    for(size_t i = 0; i < representatives.size(); ++i) {
        for(size_t j = i + 1; j < representatives.size(); ++j) {
            total_pairs++;
            id_t src = representatives[i];
            id_t dst = representatives[j];
            
            // Simple BFS to check if dst is reachable from src within tau hops
            std::queue<std::pair<id_t, size_t> > bfs_queue;
            std::unordered_set<id_t> visited_bfs;
            
            bfs_queue.push({src, 0});
            visited_bfs.insert(src);
            
            bool reachable = false;
            while(!bfs_queue.empty()) {
                auto [current, hops] = bfs_queue.front();
                bfs_queue.pop();
                
                if(current == dst) {
                    reachable = true;
                    break;
                }
                
                if(hops >= tau) {
                    continue;
                }
                
                // Get neighbors (limit to avoid too much computation)
                try {
                    if(current >= searcher->max_elements) {
                        continue;
                    }
                    auto [neighbors, sz, st] = searcher->getNeighbors(current);
                    size_t sz_size = (size_t)sz;
                    size_t max_neighbors = std::min(sz_size, (size_t)10);  // Limit neighbors checked
                    for(size_t idx = st; idx < st + max_neighbors && idx < st + sz_size; ++idx) {
                        id_t neighbor = neighbors[idx];
                        if(neighbor < searcher->max_elements && visited_bfs.find(neighbor) == visited_bfs.end()) {
                            visited_bfs.insert(neighbor);
                            bfs_queue.push({neighbor, hops + 1});
                        }
                    }
                } catch(...) {
                    // Skip if error accessing neighbors
                    break;
                }
            }
            
            if(reachable) {
                reachable_pairs++;
            }
        }
    }
    
    // Return mutual reachability ratio
    if(total_pairs > 0) {
        return (float)reachable_pairs / total_pairs;
    }
    return 0.0f;
}

// 1. Calculate Top-k Margin (MG): d_{k+1} - d_k or log d_{k+1} - log d_k
float CalculateTopKMargin(const std::vector<std::pair<float, id_t> >& results, size_t k, bool use_log = true) {
    if(results.size() < k + 1) {
        return 0.0f;  // Not enough results
    }
    
    float d_k = results[k - 1].first;
    float d_k1 = results[k].first;
    
    if(use_log) {
        if(d_k <= 0 || d_k1 <= 0) {
            return 0.0f;
        }
        return std::log(d_k1) - std::log(d_k);
    } else {
        return d_k1 - d_k;
    }
}

// 2. Calculate Top-k Cohesion (LCS, sampled): average pairwise similarity inside Top-k
float CalculateTopKCohesion(HNSW_NGFix<float>* searcher, 
                           const std::vector<std::pair<float, id_t> >& results, 
                           size_t k, size_t s = 16) {
    if(results.size() < k || k < 2) {
        return 0.0f;
    }
    
    size_t actual_k = std::min(k, results.size());
    if(actual_k < 2) {
        return 0.0f;
    }
    
    // Sample s pairs from top-k
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, actual_k - 1);
    
    float total_similarity = 0.0f;
    size_t pair_count = 0;
    
    // Sample s pairs (avoid self-pairs)
    for(size_t i = 0; i < s && pair_count < s * 2; ++i) {
        size_t idx1 = dis(gen);
        size_t idx2 = dis(gen);
        if(idx1 == idx2) {
            continue;
        }
        
        id_t id1 = results[idx1].second;
        id_t id2 = results[idx2].second;
        
        // Calculate similarity (1 - normalized distance)
        float dist = searcher->getDist(id1, id2);
        // For IP metric, similarity can be negative, so we use a normalized version
        // For simplicity, we use 1 / (1 + dist) as similarity measure
        float similarity = 1.0f / (1.0f + dist);
        total_similarity += similarity;
        pair_count++;
    }
    
    if(pair_count > 0) {
        return total_similarity / pair_count;
    }
    return 0.0f;
}

// 3. Calculate Result Hubness (HubMean / HubVar)
// Maintain hit counters for result vectors
struct HubnessTracker {
    std::unordered_map<id_t, uint16_t> hit_counts;
    static constexpr uint16_t MAX_COUNT = 65535;  // 16-bit saturating
    
    void record_hit(id_t id) {
        if(hit_counts[id] < MAX_COUNT) {
            hit_counts[id]++;
        }
    }
    
    std::pair<float, float> get_stats(const std::vector<std::pair<float, id_t> >& results, size_t k) {
        if(results.empty()) {
            return {0.0f, 0.0f};
        }
        
        size_t actual_k = std::min(k, results.size());
        std::vector<float> hubness_values;
        hubness_values.reserve(actual_k);
        
        for(size_t i = 0; i < actual_k; ++i) {
            id_t id = results[i].second;
            hubness_values.push_back((float)hit_counts[id]);
        }
        
        if(hubness_values.empty()) {
            return {0.0f, 0.0f};
        }
        
        float mean = std::accumulate(hubness_values.begin(), hubness_values.end(), 0.0f) / hubness_values.size();
        
        float variance = 0.0f;
        for(float v : hubness_values) {
            float diff = v - mean;
            variance += diff * diff;
        }
        variance /= hubness_values.size();
        
        return {mean, variance};
    }
};

// 4. Calculate Distance Slope and Gini coefficient
std::pair<float, float> CalculateDistanceStats(const std::vector<std::pair<float, id_t> >& results, size_t k) {
    if(results.size() < k || k < 2) {
        return {0.0f, 0.0f};
    }
    
    size_t actual_k = std::min(k, results.size());
    std::vector<float> distances;
    distances.reserve(actual_k);
    
    for(size_t i = 0; i < actual_k; ++i) {
        distances.push_back(results[i].first);
    }
    
    // Calculate slope: linear regression slope of distances
    float slope = 0.0f;
    if(actual_k >= 2) {
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
        for(size_t i = 0; i < actual_k; ++i) {
            float x = (float)i;
            float y = distances[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        float n = (float)actual_k;
        float denominator = n * sum_x2 - sum_x * sum_x;
        if(std::abs(denominator) > 1e-6) {
            slope = (n * sum_xy - sum_x * sum_y) / denominator;
        }
    }
    
    // Calculate Gini coefficient
    float gini = 0.0f;
    if(actual_k >= 2) {
        std::sort(distances.begin(), distances.end());
        float mean = std::accumulate(distances.begin(), distances.end(), 0.0f) / actual_k;
        if(mean > 1e-6) {
            float sum = 0.0f;
            for(size_t i = 0; i < actual_k; ++i) {
                for(size_t j = 0; j < actual_k; ++j) {
                    sum += std::abs(distances[i] - distances[j]);
                }
            }
            gini = sum / (2.0f * actual_k * actual_k * mean);
        }
    }
    
    return {slope, gini};
}

// 5. Calculate Set-Signature Dissimilarity using MinHash
class SignatureCache {
private:
    std::vector<uint64_t> recent_signatures;
    size_t cache_size;
    static constexpr size_t HASH_COUNT = 4;  // 4 × 64-bit = 256 bits
    
public:
    SignatureCache(size_t size = 10) : cache_size(size) {}
    
    uint64_t compute_signature(const std::vector<std::pair<float, id_t> >& results, size_t k) {
        if(results.empty()) {
            return 0;
        }
        
        size_t actual_k = std::min(k, results.size());
        std::vector<id_t> ids;
        ids.reserve(actual_k);
        for(size_t i = 0; i < actual_k; ++i) {
            ids.push_back(results[i].second);
        }
        
        // Simple hash: XOR of IDs (MinHash-like)
        uint64_t signature = 0;
        for(id_t id : ids) {
            // Hash the ID
            uint64_t hash = std::hash<id_t>{}(id);
            signature ^= hash;
        }
        
        return signature;
    }
    
    float get_dissimilarity(uint64_t new_sig) {
        if(recent_signatures.empty()) {
            recent_signatures.push_back(new_sig);
            if(recent_signatures.size() > cache_size) {
                recent_signatures.erase(recent_signatures.begin());
            }
            return 1.0f;  // First signature, maximum dissimilarity
        }
        
        // Calculate Hamming distance (number of differing bits)
        float min_dissimilarity = 1.0f;
        for(uint64_t old_sig : recent_signatures) {
            uint64_t diff = new_sig ^ old_sig;
            // Count set bits (Hamming distance)
            int hamming = 0;
            while(diff) {
                hamming += diff & 1;
                diff >>= 1;
            }
            // Normalize to [0, 1]
            float dissimilarity = (float)hamming / 64.0f;
            min_dissimilarity = std::min(min_dissimilarity, dissimilarity);
        }
        
        // Update cache
        recent_signatures.push_back(new_sig);
        if(recent_signatures.size() > cache_size) {
            recent_signatures.erase(recent_signatures.begin());
        }
        
        return min_dissimilarity;
    }
};

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

    // Load base graph (not NGFix processed)
    auto hnsw_base = new HNSW_NGFix<float>(metric, index_path);
    hnsw_base->printGraphInfo();

    // Limit to num_queries
    size_t actual_num_queries = std::min(num_queries, test_number);
    
    std::vector<QueryHardnessMetrics> metrics(actual_num_queries);
    
    std::cout << "Processing " << actual_num_queries << " queries..." << std::endl;
    
    // Base efSearch for consistency check
    size_t ef_base = 100;
    size_t ef_extended = 150;  // 50% increase
    size_t ef_2x = 200;        // 2x increase (double)
    
    // Initialize trackers for new metrics
    SignatureCache signature_cache(10);
    
    for(size_t i = 0; i < actual_num_queries; ++i) {
        if(i % 10 == 0) {
            std::cout << "Processing query " << i << "/" << actual_num_queries << std::endl;
        }
        
        auto query_data = test_query + i * vecdim;
        auto gt = test_gt + i * test_gt_dim;
        
        // Search with lightweight metrics
        // Search for k+1 results to calculate top-k margin
        size_t ndc = 0;
        auto [results, ndc_result, lw_metrics] = hnsw_base->searchKnnWithLightweightMetrics(
            query_data, k + 1, ef_base, ndc, 0.2f);
        
        // Also get frontier churn using the old method
        size_t ndc2 = 0;
        auto [results2, ndc_result2, frontier_churn] = hnsw_base->searchKnnWithMetrics(
            query_data, k + 1, ef_base, ndc2);
        
        // Truncate to k for other calculations
        std::vector<std::pair<float, id_t> > results_k(results.begin(), results.begin() + std::min(k, results.size()));
        
        // Calculate recall
        float recall = CalculateRecall(results_k, gt, k);
        
        // Calculate Escape Hardness
        float escape_hardness = CalculateEscapeHardness(hnsw_base, query_data, gt, k);
        
        // Calculate self-consistency (ef_base vs ef_extended)
        float self_consistency = CalculateSelfConsistency(hnsw_base, query_data, k, ef_base, ef_extended);
        
        // Calculate self-consistency with 2x efSearch (ef_base vs ef_2x)
        float self_consistency_2x = CalculateSelfConsistency(hnsw_base, query_data, k, ef_base, ef_2x);
        
        // 7. Calculate Micro-Perturbation Stability (Jitter J)
        float jitter = CalculateJitter(hnsw_base, query_data, k, ef_base, vecdim, 0.03f);
        
        // 8. Calculate τ-Reachability Probe on Visited Subgraph
        float reachability_probe = CalculateReachabilityProbe(hnsw_base, results_k, k, 6);
        
        // 1. Calculate Top-k Margin (use full results with k+1)
        float top_k_margin = CalculateTopKMargin(results, k, true);
        
        // 2. Calculate Top-k Cohesion (use results_k)
        float top_k_cohesion = CalculateTopKCohesion(hnsw_base, results_k, k, 16);
        
        // 3. Calculate Set-Signature Dissimilarity (use results_k)
        uint64_t signature = signature_cache.compute_signature(results_k, k);
        float signature_dissimilarity = signature_cache.get_dissimilarity(signature);
        
        // Lightweight metrics are already calculated in lw_metrics
        
        metrics[i] = {
            recall,
            escape_hardness,
            frontier_churn,
            self_consistency,
            self_consistency_2x,
            ndc_result,
            top_k_margin,
            top_k_cohesion,
            signature_dissimilarity,
            jitter,
            reachability_probe,
            lw_metrics.S,
            lw_metrics.r_visit,
            lw_metrics.r_early,
            lw_metrics.top1_last1_diff,
            lw_metrics.delta_improve
        };
    }
    
    // Calculate statistics
    float avg_recall = 0, avg_escape_hardness = 0, avg_frontier_churn = 0, avg_self_consistency = 0;
    float avg_self_consistency_2x = 0;
    float avg_ndc = 0;
    float avg_top_k_margin = 0, avg_top_k_cohesion = 0;
    float avg_signature_dissimilarity = 0;
    float avg_jitter = 0, avg_reachability_probe = 0;
    float avg_r_visit = 0, avg_r_early = 0, avg_top1_last1_diff = 0, avg_delta_improve = 0;
    
    for(const auto& m : metrics) {
        avg_recall += m.recall;
        avg_escape_hardness += m.escape_hardness;
        avg_frontier_churn += m.frontier_churn;
        avg_self_consistency += m.self_consistency;
        avg_self_consistency_2x += m.self_consistency_2x;
        avg_ndc += m.ndc;
        avg_top_k_margin += m.top_k_margin;
        avg_top_k_cohesion += m.top_k_cohesion;
        avg_signature_dissimilarity += m.signature_dissimilarity;
        avg_jitter += m.jitter;
        avg_reachability_probe += m.reachability_probe;
        avg_r_visit += m.r_visit;
        avg_r_early += m.r_early;
        avg_top1_last1_diff += m.top1_last1_diff;
        avg_delta_improve += m.delta_improve;
    }
    
    avg_recall /= actual_num_queries;
    avg_escape_hardness /= actual_num_queries;
    avg_frontier_churn /= actual_num_queries;
    avg_self_consistency /= actual_num_queries;
    avg_self_consistency_2x /= actual_num_queries;
    avg_ndc /= actual_num_queries;
    avg_top_k_margin /= actual_num_queries;
    avg_top_k_cohesion /= actual_num_queries;
    avg_signature_dissimilarity /= actual_num_queries;
    avg_jitter /= actual_num_queries;
    avg_reachability_probe /= actual_num_queries;
    avg_r_visit /= actual_num_queries;
    avg_r_early /= actual_num_queries;
    avg_top1_last1_diff /= actual_num_queries;
    avg_delta_improve /= actual_num_queries;
    
    // Calculate P90 for frontier churn
    std::vector<float> fc_values;
    for(const auto& m : metrics) {
        fc_values.push_back(m.frontier_churn);
    }
    std::sort(fc_values.begin(), fc_values.end());
    size_t p90_idx = (size_t)(fc_values.size() * 0.9);
    float p90_frontier_churn = fc_values[p90_idx];
    
    // Write results to JSON-like format
    std::ofstream output(result_path);
    output << std::fixed << std::setprecision(6);
    
    output << "{\n";
    output << "  \"num_queries\": " << actual_num_queries << ",\n";
    output << "  \"k\": " << k << ",\n";
    output << "  \"ef_base\": " << ef_base << ",\n";
    output << "  \"ef_extended\": " << ef_extended << ",\n";
    output << "  \"ef_2x\": " << ef_2x << ",\n";
    output << "  \"queries\": [\n";
    
    for(size_t i = 0; i < actual_num_queries; ++i) {
        output << "    {\n";
        output << "      \"query_id\": " << i << ",\n";
        output << "      \"recall\": " << metrics[i].recall << ",\n";
        output << "      \"escape_hardness\": " << metrics[i].escape_hardness << ",\n";
        output << "      \"frontier_churn\": " << metrics[i].frontier_churn << ",\n";
        output << "      \"self_consistency\": " << metrics[i].self_consistency << ",\n";
        output << "      \"self_consistency_2x\": " << metrics[i].self_consistency_2x << ",\n";
        output << "      \"ndc\": " << metrics[i].ndc << ",\n";
        output << "      \"top_k_margin\": " << metrics[i].top_k_margin << ",\n";
        output << "      \"top_k_cohesion\": " << metrics[i].top_k_cohesion << ",\n";
        output << "      \"signature_dissimilarity\": " << metrics[i].signature_dissimilarity << ",\n";
        output << "      \"jitter\": " << metrics[i].jitter << ",\n";
        output << "      \"reachability_probe\": " << metrics[i].reachability_probe << ",\n";
        output << "      \"r_visit\": " << metrics[i].r_visit << ",\n";
        output << "      \"r_early\": " << metrics[i].r_early << ",\n";
        output << "      \"top1_last1_diff\": " << metrics[i].top1_last1_diff << ",\n";
        output << "      \"delta_improve\": " << metrics[i].delta_improve << "\n";
        output << "    }";
        if(i < actual_num_queries - 1) {
            output << ",";
        }
        output << "\n";
    }
    
    output << "  ],\n";
    output << "  \"statistics\": {\n";
    output << "    \"avg_recall\": " << avg_recall << ",\n";
    output << "    \"avg_escape_hardness\": " << avg_escape_hardness << ",\n";
    output << "    \"avg_frontier_churn\": " << avg_frontier_churn << ",\n";
    output << "    \"p90_frontier_churn\": " << p90_frontier_churn << ",\n";
        output << "    \"avg_self_consistency\": " << avg_self_consistency << ",\n";
        output << "    \"avg_self_consistency_2x\": " << avg_self_consistency_2x << ",\n";
        output << "    \"avg_ndc\": " << avg_ndc << ",\n";
    output << "    \"avg_top_k_margin\": " << avg_top_k_margin << ",\n";
    output << "    \"avg_top_k_cohesion\": " << avg_top_k_cohesion << ",\n";
    output << "    \"avg_signature_dissimilarity\": " << avg_signature_dissimilarity << ",\n";
    output << "    \"avg_jitter\": " << avg_jitter << ",\n";
    output << "    \"avg_reachability_probe\": " << avg_reachability_probe << ",\n";
    output << "    \"avg_r_visit\": " << avg_r_visit << ",\n";
    output << "    \"avg_r_early\": " << avg_r_early << ",\n";
    output << "    \"avg_top1_last1_diff\": " << avg_top1_last1_diff << ",\n";
    output << "    \"avg_delta_improve\": " << avg_delta_improve << "\n";
    output << "  }\n";
    output << "}\n";
    output.close();
    
    std::cout << "\nResults written to: " << result_path << std::endl;
    std::cout << "Average Recall: " << avg_recall << std::endl;
    std::cout << "Average Escape Hardness: " << avg_escape_hardness << std::endl;
    std::cout << "Average Frontier Churn: " << avg_frontier_churn << std::endl;
    std::cout << "P90 Frontier Churn: " << p90_frontier_churn << std::endl;
    std::cout << "Average Self-Consistency: " << avg_self_consistency << std::endl;
    std::cout << "Average Self-Consistency (2x): " << avg_self_consistency_2x << std::endl;
    std::cout << "Average Top-k Margin: " << avg_top_k_margin << std::endl;
    std::cout << "Average Top-k Cohesion: " << avg_top_k_cohesion << std::endl;
    std::cout << "Average Signature Dissimilarity: " << avg_signature_dissimilarity << std::endl;
    std::cout << "Average Jitter: " << avg_jitter << std::endl;
    std::cout << "Average Reachability Probe: " << avg_reachability_probe << std::endl;
    std::cout << "Average r_visit: " << avg_r_visit << std::endl;
    std::cout << "Average r_early: " << avg_r_early << std::endl;
    std::cout << "Average top1_last1_diff: " << avg_top1_last1_diff << std::endl;
    std::cout << "Average delta_improve: " << avg_delta_improve << std::endl;
    
    delete[] test_query;
    delete[] test_gt;
    delete hnsw_base;
    
    return 0;
}

