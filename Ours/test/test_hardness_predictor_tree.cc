#include "ourslib/graph/hnsw_ours.h"
#include "ourslib/graph/node.h"
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

using namespace ours;

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
    
    // Calibration parameters (linear transformation: calibrated = a * raw + b)
    float calib_a;
    float calib_b;
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
    RandomForestPredictor(size_t n_trees = 200, size_t max_d = 15, size_t min_split = 5) 
        : num_trees(n_trees), max_depth(max_d), min_samples_split(min_split), rng(42),
          calib_a(1.0f), calib_b(0.0f) {}
    
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
        // Default: uniform sampling
        std::vector<float> recalls(targets.size());
        for(size_t i = 0; i < targets.size(); ++i) {
            recalls[i] = 1.0f - targets[i];
        }
        train(all_features, targets, recalls);
    }
    
    void train(const std::vector<std::vector<float>>& all_features, const std::vector<float>& targets, 
               const std::vector<float>& recalls) {
        if(all_features.empty() || feature_mask.empty()) {
            throw std::runtime_error("Invalid feature dimensions");
        }
        
        if(recalls.size() != targets.size()) {
            throw std::runtime_error("Recalls size must match targets size");
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
        
        // Calculate sampling weights: balanced aggressive weighting
        // Strategy: significantly increase hard query sampling while maintaining reasonable easy query coverage
        std::vector<float> weights(n_samples);
        const float base_weight = 0.1f;  // Base weight for all queries
        const float hard_threshold = 0.7f;  // Hardness threshold (recall < 0.3)
        const float very_hard_threshold = 0.9f;  // Very hard threshold (recall < 0.1)
        float weight_sum = 0.0f;
        for(size_t i = 0; i < n_samples; ++i) {
            float hardness = 1.0f - recalls[i];
            if(hardness >= very_hard_threshold) {
                // Very hard queries (recall < 0.1): highest weight (3x multiplier)
                weights[i] = std::pow(hardness, 2.5f) * 3.0f + base_weight;
            } else if(hardness >= hard_threshold) {
                // Hard queries (0.3 < recall < 0.7): high weight (2x multiplier)
                weights[i] = std::pow(hardness, 2.0f) * 2.0f + base_weight;
            } else {
                // Easy queries (recall > 0.7): reduced weight but not too low
                weights[i] = hardness * 0.5f + base_weight;
            }
            weight_sum += weights[i];
        }
        
        // Normalize weights to sum to n_samples
        for(size_t i = 0; i < n_samples; ++i) {
            weights[i] = weights[i] * n_samples / weight_sum;
        }
        
        // Print weight statistics for debugging
        float min_weight = *std::min_element(weights.begin(), weights.end());
        float max_weight = *std::max_element(weights.begin(), weights.end());
        float avg_weight = weight_sum / n_samples;
        std::cout << "  Weight statistics: min=" << min_weight << ", max=" << max_weight 
                  << ", avg=" << avg_weight << std::endl;
        
        // Build cumulative distribution for weighted sampling
        std::vector<float> cumsum(n_samples);
        cumsum[0] = weights[0];
        for(size_t i = 1; i < n_samples; ++i) {
            cumsum[i] = cumsum[i-1] + weights[i];
        }
        
        // Build trees
        trees.clear();
        trees.reserve(num_trees);
        
        std::vector<size_t> feature_indices;
        for(size_t i = 0; i < num_features; ++i) {
            feature_indices.push_back(i);
        }
        
        std::cout << "Building " << num_trees << " trees with weighted sampling..." << std::endl;
        std::uniform_real_distribution<float> weight_dist(0.0f, cumsum[n_samples - 1]);
        
        for(size_t t = 0; t < num_trees; ++t) {
            if(t % 10 == 0) {
                std::cout << "  Tree " << t << "/" << num_trees << std::endl;
            }
            
            // Weighted bootstrap sampling
            std::vector<size_t> bootstrap_indices;
            for(size_t i = 0; i < n_samples; ++i) {
                float r = weight_dist(rng);
                // Binary search for the index
                size_t idx = std::lower_bound(cumsum.begin(), cumsum.end(), r) - cumsum.begin();
                if(idx >= n_samples) idx = n_samples - 1;
                bootstrap_indices.push_back(idx);
            }
            
            TreeNode* tree = buildTree(normalized_features, targets, bootstrap_indices, 0, feature_indices);
            trees.push_back(tree);
        }
        
        // Post-training calibration: fit linear transformation to improve correlation
        std::cout << "Fitting calibration parameters..." << std::endl;
        size_t calib_size = std::min(n_samples, (size_t)10000);
        std::vector<float> calib_pred(calib_size);
        std::vector<float> calib_actual(calib_size);
        
        for(size_t i = 0; i < calib_size; ++i) {
            calib_pred[i] = predict(all_features[i]);
            calib_actual[i] = targets[i];
        }
        
        // Simple linear regression: actual = a * pred + b
        float pred_mean = 0.0f, actual_mean = 0.0f;
        for(size_t i = 0; i < calib_size; ++i) {
            pred_mean += calib_pred[i];
            actual_mean += calib_actual[i];
        }
        pred_mean /= calib_size;
        actual_mean /= calib_size;
        
        float numerator = 0.0f, denominator = 0.0f;
        for(size_t i = 0; i < calib_size; ++i) {
            float pred_diff = calib_pred[i] - pred_mean;
            numerator += pred_diff * (calib_actual[i] - actual_mean);
            denominator += pred_diff * pred_diff;
        }
        
        if(denominator > 1e-6f) {
            calib_a = numerator / denominator;
            calib_b = actual_mean - calib_a * pred_mean;
            std::cout << "  Calibration: calibrated = " << calib_a << " * raw + " << calib_b << std::endl;
        } else {
            calib_a = 1.0f;
            calib_b = 0.0f;
            std::cout << "  Calibration: skipped (insufficient variance)" << std::endl;
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
        
        // Apply calibration
        result = calib_a * result + calib_b;
        
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

// Extract DARTH features + feature interactions (11 base + interactions)
std::vector<float> extractAllFeatures(
    const HNSW_Ours<float>::LightweightMetrics& metrics,
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
    // These help capture non-linear relationships
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

// DARTH feature names (11 base + 6 interactions = 17 features total)
std::vector<std::string> feature_names = {
    // Type: Index (0-2)
    "nstep",      // Search step
    "ndis",       // No. distance calculations
    "ninserts",   // No. updates in the NN result set
    // Type: NN Distance (3-5)
    "firstNN",    // Distance of first NN found
    "closestNN",  // Distance of current closest NN
    "furthestNN", // Distance of current furthest (k-th) NN
    // Type: NN Stats (6-10)
    "avg",        // Average of distances of the NN
    "var",        // Variance of distances of NN
    "med",        // Median of distances of NN
    "perc25",     // 25th percentile of distances of NN
    "perc75",     // 75th percentile of distances of NN
    // Feature Interactions (11-16)
    "firstNN/furthestNN",  // Distance ratio
    "var/avg",             // Coefficient of variation
    "nstep*avg",           // nstep * avg interaction
    "ndis*var",            // ndis * var interaction
    "rel_range",           // Relative distance range
    "med/avg"              // med/avg ratio
};

// Parse feature combination from JSON file (from linear regression results)
std::vector<bool> parseFeatureCombinationFromJSON(const std::string& json_path) {
    std::vector<bool> mask(17, false);  // 11 base + 6 interactions
    
    std::ifstream in(json_path);
    if(!in) {
        std::cout << "Warning: Cannot open JSON file, using default feature combination" << std::endl;
        // Default: use all 17 features (11 base + 6 interactions)
        for(size_t i = 0; i < 17; ++i) {
            mask[i] = true;
        }
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
        std::cout << "Warning: No features found in JSON, using all features" << std::endl;
        for(size_t i = 0; i < 17; ++i) {
            mask[i] = true;
        }
    } else {
        std::cout << "Loaded " << count << " features from JSON" << std::endl;
    }
    
    return mask;
}

// Save features, targets, and recalls to binary file
void saveFeaturesToFile(const std::string& filename,
                       const std::vector<std::vector<float>>& features,
                       const std::vector<float>& targets,
                       const std::vector<float>& recalls) {
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
    out.write((char*)recalls.data(), sizeof(float) * n_samples);
    
    out.close();
    std::cout << "Saved " << n_samples << " samples with " << n_features << " features to " << filename << std::endl;
}

// Load features, targets, and recalls from binary file
void loadFeaturesFromFile(const std::string& filename,
                          std::vector<std::vector<float>>& features,
                          std::vector<float>& targets,
                          std::vector<float>& recalls) {
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
    recalls.resize(n_samples);
    
    for(size_t i = 0; i < n_samples; ++i) {
        features[i].resize(n_features);
        in.read((char*)features[i].data(), sizeof(float) * n_features);
    }
    
    in.read((char*)targets.data(), sizeof(float) * n_samples);
    
    // Try to read recalls (for backward compatibility)
    std::streampos pos = in.tellg();
    in.seekg(0, std::ios::end);
    std::streampos end_pos = in.tellg();
    in.seekg(pos);
    
    if(end_pos - pos >= (std::streampos)(sizeof(float) * n_samples)) {
        in.read((char*)recalls.data(), sizeof(float) * n_samples);
    } else {
        // Backward compatibility: calculate recalls from hardness
        for(size_t i = 0; i < n_samples; ++i) {
            recalls[i] = 1.0f - targets[i];
        }
    }
    
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
        if (arg == "--use_lightgbm")
            paths["use_lightgbm"] = "true";
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
    auto hnsw_base = new HNSW_Ours<float>(metric, index_path);
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
        std::vector<float> train_recalls;
        
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
            
            auto all_features = extractAllFeatures(lw_metrics, ndc_result, ef_train, results, k);
            train_all_features.push_back(all_features);
            train_hardness.push_back(actual_hardness);
            train_recalls.push_back(actual_recall);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Collection time: " << duration.count() << " ms" << std::endl;
        
        // Save to file
        saveFeaturesToFile(feature_cache_path, train_all_features, train_hardness, train_recalls);
        
        delete[] train_query;
        delete[] train_gt;
        
    } else if(mode == "train_test") {
        // Check if using LightGBM
        bool use_lightgbm = paths.count("use_lightgbm") > 0;
        
        if(use_lightgbm) {
            // Use LightGBM via Python script
            std::cout << "=== Using LightGBM for training ===" << std::endl;
            
            // Get number of trees (default 300 for 10M data, with reduced learning rate)
            size_t num_trees = paths.count("num_trees") ? std::stoi(paths["num_trees"]) : 300;
            size_t max_depth = 20;
            
            // Train LightGBM model
            std::string model_path = result_path + ".lgbm.model";
            std::string script_path = "/workspace/OOD-ANNS/Ours/scripts/train_lightgbm_hardness_predictor.py";
            
            std::string train_cmd = "python3 " + script_path + " " + 
                                   feature_cache_path + " " + 
                                   model_path + " " +
                                   std::to_string(num_trees) + " " +
                                   std::to_string(max_depth);
            
            std::cout << "Training LightGBM model..." << std::endl;
            int ret = system(train_cmd.c_str());
            if(ret != 0) {
                std::cerr << "Error: LightGBM training failed!" << std::endl;
                return 1;
            }
            
            // For testing, we need to extract features from test queries
            // We'll do this in a separate step or modify the Python script to handle it
            // For now, let's create a test feature cache and use Python for prediction too
            std::cout << "\n=== Testing with LightGBM ===" << std::endl;
            
            // Load test queries and extract features
            auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
            auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
            size_t actual_test = std::min(num_test_queries, test_number);
            
            std::cout << "Extracting test features from " << actual_test << " queries..." << std::endl;
            
            // Create test feature cache
            std::string test_cache_path = feature_cache_path + ".test";
            std::vector<std::vector<float>> test_features;
            std::vector<float> test_hardness;
            std::vector<float> test_recalls;
            
            size_t ef_test = 100;
            for(size_t i = 0; i < actual_test; ++i) {
                if(i % 100 == 0) {
                    std::cout << "Processing test query " << i << "/" << actual_test << std::endl;
                }
                
                auto query_data = test_query + i * vecdim;
                auto gt = test_gt + i * test_gt_dim;
                
                size_t ndc = 0;
                auto [results, ndc_result, lw_metrics] = hnsw_base->searchKnnWithLightweightMetrics(
                    query_data, k, ef_test, ndc, 0.2f);
                
                float actual_recall = CalculateRecall(results, gt, k);
                float actual_hard = 1.0f - actual_recall;
                
                auto all_features = extractAllFeatures(lw_metrics, ndc_result, ef_test, results, k);
                
                test_features.push_back(all_features);
                test_hardness.push_back(actual_hard);
                test_recalls.push_back(actual_recall);
            }
            
            // Save test features
            saveFeaturesToFile(test_cache_path, test_features, test_hardness, test_recalls);
            
            // Predict using Python script
            std::string predict_script = "/workspace/OOD-ANNS/Ours/scripts/predict_lightgbm_hardness.py";
            std::string predict_cmd = "python3 " + predict_script + " " + 
                                     model_path + " " + 
                                     test_cache_path + " " + 
                                     result_path;
            
            std::cout << "Running predictions..." << std::endl;
            ret = system(predict_cmd.c_str());
            if(ret != 0) {
                std::cerr << "Error: LightGBM prediction failed!" << std::endl;
                return 1;
            }
            
            std::cout << "LightGBM training and testing completed!" << std::endl;
            std::cout << "Results saved to: " << result_path << std::endl;
            
            delete[] test_query;
            delete[] test_gt;
            return 0;
        }
        
        // Original Random Forest code
        // Mode 2: Load features from file, train model, and test on 1000 queries
        std::cout << "=== Loading features from cache ===" << std::endl;
        
        std::vector<std::vector<float>> train_all_features;
        std::vector<float> train_hardness;
        std::vector<float> train_recalls;
        loadFeaturesFromFile(feature_cache_path, train_all_features, train_hardness, train_recalls);
        
        // Use best feature combination from linear regression results
        std::vector<bool> best_mask;
        if(paths.count("best_features_json")) {
            best_mask = parseFeatureCombinationFromJSON(paths["best_features_json"]);
        } else {
            // Default: use all 17 features (11 base + 6 interactions)
            best_mask.resize(17, false);
            for(size_t i = 0; i < 17; ++i) {
                best_mask[i] = true;
            }
            std::cout << "Using all 17 features (11 DARTH + 6 interactions)" << std::endl;
        }
        
        // Get feature names for output
        std::vector<std::string> selected_features;
        for(size_t i = 0; i < best_mask.size(); ++i) {
            if(best_mask[i]) {
                selected_features.push_back(feature_names[i]);
            }
        }
        
        // Use all data for training (no validation split for final model)
        std::vector<std::vector<float>> train_features = train_all_features;
        std::vector<float> train_targets = train_hardness;
        
        std::cout << "Using " << train_features.size() << " samples for training" << std::endl;
        
        // Get number of trees (default 100)
        size_t num_trees = paths.count("num_trees") ? std::stoi(paths["num_trees"]) : 100;
        
        std::cout << "\n=== Training with " << num_trees << " trees ===" << std::endl;
        
        RandomForestPredictor predictor(num_trees, 10, 10);
        predictor.setFeatureMask(best_mask);
        
        // Train with weighted sampling (probability inversely proportional to recall)
        auto train_start = std::chrono::high_resolution_clock::now();
        predictor.train(train_features, train_targets, train_recalls);
        auto train_end = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
        double train_latency_ms = train_duration.count();
        
        std::cout << "Training completed in " << train_latency_ms << " ms" << std::endl;
        
        // Load test queries
        auto test_query = LoadData<float>(test_query_path, test_number, vecdim);
        auto test_gt = LoadData<int>(test_gt_path, test_number, test_gt_dim);
        size_t actual_test = std::min(num_test_queries, test_number);
        
        std::cout << "\n=== Testing on " << actual_test << " queries ===" << std::endl;
        
        // Test on actual queries and collect results
        std::vector<float> test_recalls;
        std::vector<float> test_hardness;
        std::vector<float> predicted_hardness;
        
        size_t ef_test = 100;
        for(size_t i = 0; i < actual_test; ++i) {
            if(i % 100 == 0) {
                std::cout << "Testing query " << i << "/" << actual_test << std::endl;
            }
            
            auto query_data = test_query + i * vecdim;
            auto gt = test_gt + i * test_gt_dim;
            
            size_t ndc = 0;
            auto [results, ndc_result, lw_metrics] = hnsw_base->searchKnnWithLightweightMetrics(
                query_data, k, ef_test, ndc, 0.2f);
            
            // Calculate actual recall and hardness
            float actual_recall = CalculateRecall(results, gt, k);
            float actual_hard = 1.0f - actual_recall;
            
            // Extract features and predict
            auto all_features = extractAllFeatures(lw_metrics, ndc_result, ef_test, results, k);
            float pred_hard = predictor.predict(all_features);
            
            test_recalls.push_back(actual_recall);
            test_hardness.push_back(actual_hard);
            predicted_hardness.push_back(pred_hard);
        }
        
        // Calculate MSE
        float test_mse = 0.0f;
        for(size_t i = 0; i < actual_test; ++i) {
            float error = test_hardness[i] - predicted_hardness[i];
            test_mse += error * error;
        }
        test_mse /= actual_test;
        
        std::cout << "\nTest MSE: " << test_mse << std::endl;
        
        // Write results in the same format as hardness_predictor_results.json
        std::ofstream result_file(result_path);
        result_file << std::fixed << std::setprecision(6);
        result_file << "{\n";
        result_file << "  \"num_queries\": " << actual_test << ",\n";
        result_file << "  \"k\": " << k << ",\n";
        result_file << "  \"num_trees\": " << num_trees << ",\n";
        result_file << "  \"best_feature_combination\": [";
        bool first = true;
        for(const auto& feat : selected_features) {
            if(!first) result_file << ", ";
            result_file << "\"" << feat << "\"";
            first = false;
        }
        result_file << "],\n";
        result_file << "  \"best_mse\": " << test_mse << ",\n";
        result_file << "  \"queries\": [\n";
        
        for(size_t i = 0; i < actual_test; ++i) {
            result_file << "    {\n";
            result_file << "      \"query_id\": " << i << ",\n";
            result_file << "      \"recall\": " << test_recalls[i] << ",\n";
            result_file << "      \"hardness\": " << test_hardness[i] << ",\n";
            result_file << "      \"predicted_hardness\": " << predicted_hardness[i] << "\n";
            result_file << "    }";
            if(i < actual_test - 1) {
                result_file << ",";
            }
            result_file << "\n";
        }
        
        result_file << "  ]\n";
        result_file << "}\n";
        result_file.close();
        
        std::cout << "\nResults written to: " << result_path << std::endl;
        std::cout << "Test MSE: " << test_mse << std::endl;
        
        delete[] test_query;
        delete[] test_gt;
    }
    
    delete hnsw_base;
    
    return 0;
}

