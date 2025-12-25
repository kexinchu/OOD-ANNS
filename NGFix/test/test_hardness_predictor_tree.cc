#include "ngfixlib/graph/hnsw_ngfix.h"
#include "ngfixlib/graph/node.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <numeric>
#include <map>
#include <chrono>
#include <sstream>

using namespace ngfixlib;

// Simple Decision Tree Node
struct TreeNode {
    int feature_idx;
    float threshold;
    float value;  // For leaf nodes
    TreeNode* left;
    TreeNode* right;
    bool is_leaf;
    
    TreeNode() : feature_idx(-1), threshold(0.0f), value(0.0f), 
                 left(nullptr), right(nullptr), is_leaf(false) {}
    
    ~TreeNode() {
        if(left) delete left;
        if(right) delete right;
    }
};

// Simple Random Forest implementation
class RandomForestPredictor {
private:
    std::vector<TreeNode*> trees;
    std::vector<bool> feature_mask;
    size_t num_features;
    size_t num_trees;
    size_t max_depth;
    size_t min_samples_split;
    std::mt19937 rng;
    
    // Feature normalization
    std::vector<float> feature_mean;
    std::vector<float> feature_std;
    
    TreeNode* buildTree(const std::vector<std::vector<float>>& features,
                       const std::vector<float>& targets,
                       const std::vector<size_t>& indices,
                       size_t depth,
                       const std::vector<size_t>& feature_indices) {
        if(indices.empty()) {
            TreeNode* node = new TreeNode();
            node->is_leaf = true;
            node->value = 0.0f;
            return node;
        }
        
        // Calculate mean target for this node
        float mean_target = 0.0f;
        for(size_t idx : indices) {
            mean_target += targets[idx];
        }
        mean_target /= indices.size();
        
        // Check stopping conditions
        if(depth >= max_depth || indices.size() < min_samples_split) {
            TreeNode* node = new TreeNode();
            node->is_leaf = true;
            node->value = mean_target;
            return node;
        }
        
        // Try to find best split
        float best_impurity = 1e10f;
        int best_feature = -1;
        float best_threshold = 0.0f;
        std::vector<size_t> best_left, best_right;
        
        // Try random subset of features
        std::vector<size_t> candidate_features = feature_indices;
        std::shuffle(candidate_features.begin(), candidate_features.end(), rng);
        size_t n_features_to_try = std::max((size_t)1, (size_t)std::sqrt(candidate_features.size()));
        
        for(size_t f = 0; f < n_features_to_try && f < candidate_features.size(); ++f) {
            size_t feat_idx = candidate_features[f];
            
            // Collect feature values
            std::vector<float> feat_values;
            for(size_t idx : indices) {
                feat_values.push_back(features[idx][feat_idx]);
            }
            std::sort(feat_values.begin(), feat_values.end());
            
            // Try different thresholds
            size_t n_thresholds = std::min((size_t)10, feat_values.size() - 1);
            for(size_t t = 0; t < n_thresholds; ++t) {
                size_t threshold_idx = (t + 1) * feat_values.size() / (n_thresholds + 1);
                float threshold = feat_values[threshold_idx];
                
                // Split indices
                std::vector<size_t> left_indices, right_indices;
                for(size_t idx : indices) {
                    if(features[idx][feat_idx] <= threshold) {
                        left_indices.push_back(idx);
                    } else {
                        right_indices.push_back(idx);
                    }
                }
                
                if(left_indices.empty() || right_indices.empty()) continue;
                
                // Calculate weighted variance (impurity)
                float left_mean = 0.0f, right_mean = 0.0f;
                for(size_t idx : left_indices) left_mean += targets[idx];
                for(size_t idx : right_indices) right_mean += targets[idx];
                left_mean /= left_indices.size();
                right_mean /= right_indices.size();
                
                float left_var = 0.0f, right_var = 0.0f;
                for(size_t idx : left_indices) {
                    float diff = targets[idx] - left_mean;
                    left_var += diff * diff;
                }
                for(size_t idx : right_indices) {
                    float diff = targets[idx] - right_mean;
                    right_var += diff * diff;
                }
                
                float weighted_impurity = (left_var * left_indices.size() + 
                                          right_var * right_indices.size()) / indices.size();
                
                if(weighted_impurity < best_impurity) {
                    best_impurity = weighted_impurity;
                    best_feature = feat_idx;
                    best_threshold = threshold;
                    best_left = left_indices;
                    best_right = right_indices;
                }
            }
        }
        
        TreeNode* node = new TreeNode();
        if(best_feature == -1 || best_impurity >= 1e9f) {
            // No good split found
            node->is_leaf = true;
            node->value = mean_target;
            return node;
        }
        
        node->is_leaf = false;
        node->feature_idx = best_feature;
        node->threshold = best_threshold;
        node->left = buildTree(features, targets, best_left, depth + 1, feature_indices);
        node->right = buildTree(features, targets, best_right, depth + 1, feature_indices);
        
        return node;
    }
    
    float predictTree(TreeNode* tree, const std::vector<float>& features) const {
        if(tree->is_leaf) {
            return tree->value;
        }
        
        if(features[tree->feature_idx] <= tree->threshold) {
            return predictTree(tree->left, features);
        } else {
            return predictTree(tree->right, features);
        }
    }
    
public:
    RandomForestPredictor(size_t n_trees = 100, size_t max_d = 10, size_t min_split = 10) 
        : num_trees(n_trees), max_depth(max_d), min_samples_split(min_split), rng(42) {}
    
    ~RandomForestPredictor() {
        for(TreeNode* tree : trees) {
            if(tree) delete tree;
        }
    }
    
    void setFeatureMask(const std::vector<bool>& mask) {
        feature_mask = mask;
        num_features = std::count(mask.begin(), mask.end(), true);
        feature_mean.resize(num_features, 0.0f);
        feature_std.resize(num_features, 1.0f);
    }
    
    void train(const std::vector<std::vector<float>>& all_features, const std::vector<float>& targets) {
        if(all_features.empty() || feature_mask.empty()) {
            throw std::runtime_error("Invalid feature dimensions");
        }
        
        // Extract features based on mask
        std::vector<std::vector<float>> features;
        for(const auto& all_feat : all_features) {
            std::vector<float> selected_features;
            for(size_t i = 0; i < all_feat.size() && i < feature_mask.size(); ++i) {
                if(feature_mask[i]) {
                    selected_features.push_back(all_feat[i]);
                }
            }
            features.push_back(selected_features);
        }
        
        if(features.empty() || features[0].size() != num_features) {
            throw std::runtime_error("Feature extraction failed");
        }
        
        size_t n_samples = features.size();
        
        // Calculate mean and std for normalization
        for(size_t i = 0; i < n_samples; ++i) {
            for(size_t j = 0; j < num_features; ++j) {
                feature_mean[j] += features[i][j];
            }
        }
        for(size_t j = 0; j < num_features; ++j) {
            feature_mean[j] /= n_samples;
        }
        
        for(size_t i = 0; i < n_samples; ++i) {
            for(size_t j = 0; j < num_features; ++j) {
                float diff = features[i][j] - feature_mean[j];
                feature_std[j] += diff * diff;
            }
        }
        for(size_t j = 0; j < num_features; ++j) {
            feature_std[j] = std::sqrt(feature_std[j] / n_samples);
            if(feature_std[j] < 1e-6f) feature_std[j] = 1.0f;
        }
        
        // Normalize features
        std::vector<std::vector<float>> normalized_features = features;
        for(size_t i = 0; i < n_samples; ++i) {
            for(size_t j = 0; j < num_features; ++j) {
                normalized_features[i][j] = (features[i][j] - feature_mean[j]) / feature_std[j];
            }
        }
        
        // Build trees
        trees.clear();
        trees.reserve(num_trees);
        
        std::vector<size_t> feature_indices;
        for(size_t i = 0; i < num_features; ++i) {
            feature_indices.push_back(i);
        }
        
        std::cout << "Building " << num_trees << " trees..." << std::endl;
        for(size_t t = 0; t < num_trees; ++t) {
            if(t % 10 == 0) {
                std::cout << "  Tree " << t << "/" << num_trees << std::endl;
            }
            
            // Bootstrap sampling
            std::vector<size_t> bootstrap_indices;
            std::uniform_int_distribution<size_t> dist(0, n_samples - 1);
            for(size_t i = 0; i < n_samples; ++i) {
                bootstrap_indices.push_back(dist(rng));
            }
            
            TreeNode* tree = buildTree(normalized_features, targets, bootstrap_indices, 0, feature_indices);
            trees.push_back(tree);
        }
    }
    
    float predict(const std::vector<float>& all_features) const {
        if(feature_mask.empty() || trees.empty()) {
            return 0.0f;
        }
        
        // Extract and normalize features
        std::vector<float> features;
        for(size_t i = 0; i < all_features.size() && i < feature_mask.size(); ++i) {
            if(feature_mask[i]) {
                features.push_back(all_features[i]);
            }
        }
        
        if(features.size() != num_features) {
            return 0.0f;
        }
        
        // Normalize
        std::vector<float> normalized_features(num_features);
        for(size_t i = 0; i < num_features; ++i) {
            normalized_features[i] = (features[i] - feature_mean[i]) / feature_std[i];
        }
        
        // Average predictions from all trees
        float sum = 0.0f;
        for(TreeNode* tree : trees) {
            sum += predictTree(tree, normalized_features);
        }
        float result = sum / trees.size();
        
        return std::max(0.0f, std::min(1.0f, result));
    }
    
    float getMSE(const std::vector<std::vector<float>>& all_features, const std::vector<float>& targets) const {
        if(feature_mask.empty() || trees.empty()) {
            return 1.0f;
        }
        
        float total_error = 0.0f;
        for(size_t i = 0; i < all_features.size(); ++i) {
            float pred = predict(all_features[i]);
            float error = targets[i] - pred;
            total_error += error * error;
        }
        return total_error / all_features.size();
    }
};

// Calculate recall
float CalculateRecall(const std::vector<std::pair<float, id_t>>& results, int* gt, size_t k) {
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

// Extract DARTH-inspired features
std::vector<float> extractAllFeatures(
    const HNSW_NGFix<float>::LightweightMetrics& metrics,
    size_t ndc,
    size_t efSearch) {
    
    std::vector<float> features;
    const float eps = 1e-6f;
    
    features.push_back((float)metrics.S);
    features.push_back((float)metrics.t_last_improve);
    features.push_back(metrics.r_visit);
    features.push_back(metrics.r_early);
    features.push_back((float)ndc);
    features.push_back((float)efSearch);
    features.push_back(metrics.d_worst_early);
    features.push_back(metrics.d_worst_final);
    features.push_back(metrics.d_best_cand_final);
    
    float distance_range = metrics.d_worst_final - metrics.d_best_cand_final;
    features.push_back(distance_range);
    
    float distance_improvement = metrics.d_worst_early - metrics.d_worst_final;
    features.push_back(distance_improvement);
    
    float relative_improvement = (metrics.d_worst_early > eps) ? 
                                distance_improvement / metrics.d_worst_early : 0.0f;
    features.push_back(relative_improvement);
    
    float worst_best_ratio = (metrics.d_best_cand_final > eps) ? 
                             metrics.d_worst_final / metrics.d_best_cand_final : 0.0f;
    features.push_back(worst_best_ratio);
    
    float early_final_ratio = (metrics.d_worst_final > eps) ? 
                              metrics.d_worst_early / metrics.d_worst_final : 0.0f;
    features.push_back(early_final_ratio);
    
    float progress_rate = (metrics.S > 0) ? (float)metrics.t_last_improve / (float)metrics.S : 0.0f;
    features.push_back(progress_rate);
    
    float dist_comp_efficiency = (ndc > 0) ? (float)metrics.S / (float)ndc : 0.0f;
    features.push_back(dist_comp_efficiency);
    
    float improvement_freq = (metrics.S > 0 && metrics.t_last_improve > 0) ? 
                             (float)metrics.S / (float)metrics.t_last_improve : 0.0f;
    features.push_back(improvement_freq);
    
    return features;
}

// DARTH feature names (17 features total)
std::vector<std::string> feature_names = {
    // Search Progress (0-5)
    "S", "t_last_improve", "r_visit", "r_early", "ndc", "efSearch",
    // Distance Statistics (6-13)
    "d_worst_early", "d_worst_final", "d_best_cand_final", "distance_range",
    "distance_improvement", "relative_improvement_rate", "worst/best_ratio", "early/final_ratio",
    // Convergence Indicators (14-16)
    "progress_rate", "dist_comp_efficiency", "improvement_freq"
};

// Parse feature combination from JSON file (from linear regression results)
std::vector<bool> parseFeatureCombinationFromJSON(const std::string& json_path) {
    std::vector<bool> mask(17, false);
    
    std::ifstream in(json_path);
    if(!in) {
        std::cout << "Warning: Cannot open JSON file, using default feature combination" << std::endl;
        // Default: r_visit, r_early, d_worst_final, d_best_cand_final, progress_rate
        mask[2] = true;   // r_visit
        mask[3] = true;   // r_early
        mask[7] = true;   // d_worst_final
        mask[8] = true;   // d_best_cand_final
        mask[14] = true;  // progress_rate
        return mask;
    }
    
    std::string line;
    bool in_feature_array = false;
    while(std::getline(in, line)) {
        if(line.find("best_feature_combination") != std::string::npos) {
            in_feature_array = true;
            continue;
        }
        
        if(in_feature_array) {
            if(line.find("]") != std::string::npos) {
                break;
            }
            
            // Extract feature name from JSON
            size_t start = line.find('"');
            if(start != std::string::npos) {
                size_t end = line.find('"', start + 1);
                if(end != std::string::npos) {
                    std::string feature_name = line.substr(start + 1, end - start - 1);
                    
                    // Find feature index
                    for(size_t i = 0; i < feature_names.size(); ++i) {
                        if(feature_names[i] == feature_name) {
                            mask[i] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    in.close();
    
    // Check if any features were found
    size_t count = std::count(mask.begin(), mask.end(), true);
    if(count == 0) {
        std::cout << "Warning: No features found in JSON, using default combination" << std::endl;
        mask[2] = true;   // r_visit
        mask[3] = true;   // r_early
        mask[7] = true;   // d_worst_final
        mask[8] = true;   // d_best_cand_final
        mask[14] = true;  // progress_rate
    } else {
        std::cout << "Loaded " << count << " features from JSON" << std::endl;
    }
    
    return mask;
}

// Save features and targets to binary file
void saveFeaturesToFile(const std::string& filename,
                       const std::vector<std::vector<float>>& features,
                       const std::vector<float>& targets) {
    std::ofstream out(filename, std::ios::binary);
    if(!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    size_t n_samples = features.size();
    size_t n_features = features.empty() ? 0 : features[0].size();
    
    out.write((char*)&n_samples, sizeof(size_t));
    out.write((char*)&n_features, sizeof(size_t));
    
    for(size_t i = 0; i < n_samples; ++i) {
        out.write((char*)features[i].data(), sizeof(float) * n_features);
    }
    
    out.write((char*)targets.data(), sizeof(float) * n_samples);
    
    out.close();
    std::cout << "Saved " << n_samples << " samples with " << n_features << " features to " << filename << std::endl;
}

// Load features and targets from binary file
void loadFeaturesFromFile(const std::string& filename,
                          std::vector<std::vector<float>>& features,
                          std::vector<float>& targets) {
    std::ifstream in(filename, std::ios::binary);
    if(!in) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    size_t n_samples, n_features;
    in.read((char*)&n_samples, sizeof(size_t));
    in.read((char*)&n_features, sizeof(size_t));
    
    features.clear();
    features.resize(n_samples);
    targets.resize(n_samples);
    
    for(size_t i = 0; i < n_samples; ++i) {
        features[i].resize(n_features);
        in.read((char*)features[i].data(), sizeof(float) * n_features);
    }
    
    in.read((char*)targets.data(), sizeof(float) * n_samples);
    
    in.close();
    std::cout << "Loaded " << n_samples << " samples with " << n_features << " features from " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> paths;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--train_query_path")
            paths["train_query_path"] = argv[i + 1];
        if (arg == "--train_gt_path")
            paths["train_gt_path"] = argv[i + 1];
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
        if (arg == "--feature_cache_path")
            paths["feature_cache_path"] = argv[i + 1];
        if (arg == "--mode")
            paths["mode"] = argv[i + 1];  // "collect" or "train_test"
        if (arg == "--num_trees")
            paths["num_trees"] = argv[i + 1];
        if (arg == "--best_features_json")
            paths["best_features_json"] = argv[i + 1];
    }
    
    std::string train_query_path = paths["train_query_path"];
    std::string best_features_json = paths.count("best_features_json") ? paths["best_features_json"] : "";
    std::string train_gt_path = paths["train_gt_path"];
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string index_path = paths["index_path"];
    std::string result_path = paths["result_path"];
    std::string metric_str = paths["metric"];
    std::string feature_cache_path = paths.count("feature_cache_path") ? 
                                     paths["feature_cache_path"] : "/tmp/features_cache.bin";
    std::string mode = paths.count("mode") ? paths["mode"] : "collect";
    size_t k = paths.count("K") ? std::stoi(paths["K"]) : 100;
    size_t num_train_queries = paths.count("num_queries") ? std::stoi(paths["num_queries"]) : 10000000;  // 10M default
    size_t num_test_queries = 1000;
    
    size_t train_number = 0, test_number = 0;
    size_t train_gt_dim = 0, test_gt_dim = 0, vecdim = 0;
    
    Metric metric;
    if(metric_str == "ip_float") {
        metric = IP_float;
    } else if(metric_str == "l2_float") {
        metric = L2_float;
    } else {
        throw std::runtime_error("Error: Unsupported metric type.");
    }
    
    // Load index
    auto hnsw_base = new HNSW_NGFix<float>(metric, index_path);
    hnsw_base->printGraphInfo();
    
    if(mode == "collect") {
        // Mode 1: Collect features from 10M queries and save to file
        std::cout << "=== Collecting features from " << num_train_queries << " queries ===" << std::endl;
        
        auto train_query = LoadData<float>(train_query_path, train_number, vecdim);
        auto train_gt = LoadData<int>(train_gt_path, train_number, train_gt_dim);
        
        size_t actual_train = std::min(num_train_queries, train_number);
        size_t ef_train = 100;
        
        std::vector<std::vector<float>> train_all_features;
        std::vector<float> train_hardness;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for(size_t i = 0; i < actual_train; ++i) {
            if(i % 10000 == 0) {
                std::cout << "Processing query " << i << "/" << actual_train << std::endl;
            }
            
            auto query_data = train_query + i * vecdim;
            auto gt = train_gt + i * train_gt_dim;
            
            size_t ndc = 0;
            auto [results, ndc_result, lw_metrics] = hnsw_base->searchKnnWithLightweightMetrics(
                query_data, k, ef_train, ndc, 0.2f);
            
            float actual_recall = CalculateRecall(results, gt, k);
            float actual_hardness = 1.0f - actual_recall;
            
            auto all_features = extractAllFeatures(lw_metrics, ndc_result, ef_train);
            train_all_features.push_back(all_features);
            train_hardness.push_back(actual_hardness);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Collection time: " << duration.count() << " ms" << std::endl;
        
        // Save to file
        saveFeaturesToFile(feature_cache_path, train_all_features, train_hardness);
        
        delete[] train_query;
        delete[] train_gt;
        
    } else if(mode == "train_test") {
        // Mode 2: Load features from file and test different tree numbers
        std::cout << "=== Loading features from cache ===" << std::endl;
        
        std::vector<std::vector<float>> train_all_features;
        std::vector<float> train_hardness;
        loadFeaturesFromFile(feature_cache_path, train_all_features, train_hardness);
        
        // Use best feature combination from linear regression results
        std::vector<bool> best_mask;
        if(paths.count("best_features_json")) {
            best_mask = parseFeatureCombinationFromJSON(paths["best_features_json"]);
        } else {
            // Default: r_visit, r_early, d_worst_final, d_best_cand_final, progress_rate
            best_mask.resize(17, false);
            best_mask[2] = true;   // r_visit
            best_mask[3] = true;   // r_early
            best_mask[7] = true;   // d_worst_final
            best_mask[8] = true;   // d_best_cand_final
            best_mask[14] = true;  // progress_rate
            std::cout << "Using default feature combination" << std::endl;
        }
        
        // Test different tree numbers
        std::vector<size_t> tree_numbers;
        if(paths.count("num_trees")) {
            // Single tree number specified
            tree_numbers = {std::stoi(paths["num_trees"])};
        } else {
            // Test all tree numbers
            tree_numbers = {10, 25, 50, 75, 100};
        }
        
        // Split data: 80% train, 20% test
        size_t test_size = train_all_features.size() / 5;
        size_t train_size = train_all_features.size() - test_size;
        
        std::vector<std::vector<float>> train_features(train_all_features.begin(), 
                                                        train_all_features.begin() + train_size);
        std::vector<std::vector<float>> test_features(train_all_features.begin() + train_size, 
                                                      train_all_features.end());
        std::vector<float> train_targets(train_hardness.begin(), 
                                         train_hardness.begin() + train_size);
        std::vector<float> test_targets(train_hardness.begin() + train_size, 
                                       train_hardness.end());
        
        std::cout << "Using " << train_size << " samples for training, " << test_size << " for testing" << std::endl;
        
        // Load test queries for latency testing
        auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
        auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
        size_t actual_test = std::min(num_test_queries, test_number);
        
        std::ofstream result_file(result_path);
        result_file << std::fixed << std::setprecision(6);
        result_file << "{\n";
        result_file << "  \"results\": [\n";
        
        bool first_result = true;
        
        for(size_t num_trees : tree_numbers) {
            std::cout << "\n=== Testing with " << num_trees << " trees ===" << std::endl;
            
            RandomForestPredictor predictor(num_trees, 10, 10);
            predictor.setFeatureMask(best_mask);
            
            // Train
            auto train_start = std::chrono::high_resolution_clock::now();
            predictor.train(train_features, train_targets);
            auto train_end = std::chrono::high_resolution_clock::now();
            auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
            double train_latency_ms = train_duration.count();
            
            // Test MSE
            float test_mse = predictor.getMSE(test_features, test_targets);
            
            // Test latency: predict on actual queries
            auto test_start = std::chrono::high_resolution_clock::now();
            size_t ef_test = 100;
            for(size_t i = 0; i < actual_test; ++i) {
                auto query_data = test_query + i * vecdim;
                size_t ndc = 0;
                auto [results, ndc_result, lw_metrics] = hnsw_base->searchKnnWithLightweightMetrics(
                    query_data, k, ef_test, ndc, 0.2f);
                
                auto all_features = extractAllFeatures(lw_metrics, ndc_result, ef_test);
                predictor.predict(all_features);
            }
            auto test_end = std::chrono::high_resolution_clock::now();
            auto test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start);
            double test_latency_ms = test_duration.count();
            double avg_test_latency_ms = test_latency_ms / actual_test;
            
            std::cout << "  Train latency: " << train_latency_ms << " ms" << std::endl;
            std::cout << "  Test MSE: " << test_mse << std::endl;
            std::cout << "  Test latency: " << avg_test_latency_ms << " ms/query" << std::endl;
            
            // Write result
            if(!first_result) result_file << ",\n";
            first_result = false;
            
            result_file << "    {\n";
            result_file << "      \"num_trees\": " << num_trees << ",\n";
            result_file << "      \"train_latency_ms\": " << train_latency_ms << ",\n";
            result_file << "      \"test_mse\": " << test_mse << ",\n";
            result_file << "      \"test_latency_ms_per_query\": " << avg_test_latency_ms << "\n";
            result_file << "    }";
        }
        
        result_file << "\n  ]\n";
        result_file << "}\n";
        result_file.close();
        
        delete[] test_query;
        delete[] test_gt;
    }
    
    delete hnsw_base;
    
    return 0;
}

