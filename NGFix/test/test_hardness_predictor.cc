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

using namespace ngfixlib;

// Advanced linear regression model with Adam optimizer and cross-validation
class HardnessPredictor {
private:
    std::vector<float> weights;  // Feature weights
    float bias;
    size_t num_features;
    std::vector<float> feature_mean;  // For normalization
    std::vector<float> feature_std;   // For normalization
    std::vector<bool> feature_mask;  // Which features to use
    
    // Adam optimizer parameters
    std::vector<float> m;  // First moment estimates
    std::vector<float> v;  // Second moment estimates
    float m_bias;
    float v_bias;
    size_t t;  // Time step
    
public:
    HardnessPredictor() : bias(0.0f), num_features(0), m_bias(0.0f), v_bias(0.0f), t(0) {}
    
    static constexpr size_t MAX_FEATURE_DIM = 10;
    
    void setFeatureMask(const std::vector<bool>& mask) {
        feature_mask = mask;
        num_features = std::count(mask.begin(), mask.end(), true);
        weights.resize(num_features, 0.0f);
        feature_mean.resize(num_features, 0.0f);
        feature_std.resize(num_features, 1.0f);
        m.resize(num_features, 0.0f);
        v.resize(num_features, 0.0f);
        t = 0;
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
        
        // Calculate mean
        for(size_t i = 0; i < n_samples; ++i) {
            for(size_t j = 0; j < num_features; ++j) {
                feature_mean[j] += features[i][j];
            }
        }
        for(size_t j = 0; j < num_features; ++j) {
            feature_mean[j] /= n_samples;
        }
        
        // Calculate std
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
        
        // Adam optimizer with L2 regularization and early stopping
        const float alpha = 0.001f;  // Learning rate
        const float beta1 = 0.9f;    // Exponential decay rate for first moment
        const float beta2 = 0.999f;  // Exponential decay rate for second moment
        const float epsilon = 1e-8f; // Small constant for numerical stability
        const float lambda = 0.0005f;  // L2 regularization coefficient (reduced for better fit)
        const size_t epochs = 3000;
        float best_error = 1e6f;
        size_t no_improve_count = 0;
        std::vector<float> best_weights = weights;
        float best_bias = bias;
        
        // Reset Adam optimizer state
        std::fill(m.begin(), m.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        m_bias = 0.0f;
        v_bias = 0.0f;
        t = 0;
        
        for(size_t epoch = 0; epoch < epochs; ++epoch) {
            float total_error = 0.0f;
            
            // Batch gradient descent: accumulate gradients
            std::vector<float> weight_gradients(num_features, 0.0f);
            float bias_gradient = 0.0f;
            
            // Shuffle for better training
            std::vector<size_t> indices(n_samples);
            std::iota(indices.begin(), indices.end(), 0);
            if(epoch % 20 == 0) {
                std::random_shuffle(indices.begin(), indices.end());
            }
            
            // Accumulate gradients over batch
            for(size_t idx = 0; idx < n_samples; ++idx) {
                size_t i = indices[idx];
                float prediction = 0.0f;
                for(size_t j = 0; j < num_features; ++j) {
                    prediction += weights[j] * normalized_features[i][j];
                }
                prediction += bias;
                prediction = std::max(0.0f, std::min(1.0f, prediction));
                
                float error = targets[i] - prediction;
                total_error += error * error;
                
                // Accumulate gradients
                for(size_t j = 0; j < num_features; ++j) {
                    weight_gradients[j] += error * normalized_features[i][j];
                }
                bias_gradient += error;
            }
            
            // Adam optimizer update
            t++;
            float alpha_t = alpha * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));
            
            for(size_t j = 0; j < num_features; ++j) {
                float gradient = weight_gradients[j] / n_samples;
                // Add L2 regularization
                gradient -= lambda * weights[j];
                
                // Update biased first moment estimate
                m[j] = beta1 * m[j] + (1.0f - beta1) * gradient;
                // Update biased second raw moment estimate
                v[j] = beta2 * v[j] + (1.0f - beta2) * gradient * gradient;
                
                // Compute bias-corrected first moment estimate
                float m_hat = m[j] / (1.0f - std::pow(beta1, t));
                // Compute bias-corrected second raw moment estimate
                float v_hat = v[j] / (1.0f - std::pow(beta2, t));
                
                // Update parameters
                weights[j] += alpha_t * m_hat / (std::sqrt(v_hat) + epsilon);
            }
            
            // Update bias with Adam
            float bias_grad = bias_gradient / n_samples;
            m_bias = beta1 * m_bias + (1.0f - beta1) * bias_grad;
            v_bias = beta2 * v_bias + (1.0f - beta2) * bias_grad * bias_grad;
            float m_bias_hat = m_bias / (1.0f - std::pow(beta1, t));
            float v_bias_hat = v_bias / (1.0f - std::pow(beta2, t));
            bias += alpha_t * m_bias_hat / (std::sqrt(v_bias_hat) + epsilon);
            
            float avg_error = total_error / n_samples;
            
            // Track best model
            if(avg_error < best_error) {
                best_error = avg_error;
                best_weights = weights;
                best_bias = bias;
                no_improve_count = 0;
            } else {
                no_improve_count++;
            }
            
            // Early stopping
            if(no_improve_count > 300) {
                break;
            }
            
            if(epoch % 300 == 0) {
                std::cout << "    Epoch " << epoch << ", MSE: " << avg_error << ", Best: " << best_error << std::endl;
            }
            if(avg_error < 0.001f) {
                break;
            }
        }
        
        // Use best model
        weights = best_weights;
        bias = best_bias;
    }
    
    float predict(const std::vector<float>& all_features) const {
        if(feature_mask.empty()) {
            return 0.0f;
        }
        
        // Extract features based on mask
        std::vector<float> features;
        for(size_t i = 0; i < all_features.size() && i < feature_mask.size(); ++i) {
            if(feature_mask[i]) {
                features.push_back(all_features[i]);
            }
        }
        
        if(features.size() != num_features || feature_mean.size() != num_features) {
            return 0.0f;
        }
        
        // Normalize and predict
        float result = bias;
        for(size_t i = 0; i < num_features; ++i) {
            float normalized_feature = (features[i] - feature_mean[i]) / feature_std[i];
            result += weights[i] * normalized_feature;
        }
        
        return std::max(0.0f, std::min(1.0f, result));
    }
    
    float getMSE(const std::vector<std::vector<float>>& all_features, const std::vector<float>& targets) const {
        if(feature_mask.empty()) {
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

// Extract DARTH-inspired features based on the paper
// DARTH Recall Predictor uses: search progress, distance statistics, convergence indicators, and efSearch
std::vector<float> extractAllFeatures(
    const HNSW_NGFix<float>::LightweightMetrics& metrics,
    size_t ndc,
    size_t efSearch) {
    
    std::vector<float> features;
    const float eps = 1e-6f;
    
    // ========== DARTH: Search Progress Features ==========
    features.push_back((float)metrics.S);                    // 0: Total visited nodes (S)
    features.push_back((float)metrics.t_last_improve);         // 1: Last improvement step index
    features.push_back(metrics.r_visit);                      // 2: Visit budget usage (S/efSearch)
    features.push_back(metrics.r_early);                      // 3: Early convergence ratio (t_last_improve/S)
    features.push_back((float)ndc);                           // 4: Number of distance computations
    features.push_back((float)efSearch);                      // 5: efSearch parameter
    
    // ========== DARTH: Distance Statistics Features ==========
    features.push_back(metrics.d_worst_early);                // 6: Early stage worst distance
    features.push_back(metrics.d_worst_final);                // 7: Final worst distance
    features.push_back(metrics.d_best_cand_final);            // 8: Final best candidate distance
    
    // Distance range (worst - best candidate)
    float distance_range = metrics.d_worst_final - metrics.d_best_cand_final;
    features.push_back(distance_range);                       // 9: Distance range
    
    // Distance improvement (early - final)
    float distance_improvement = metrics.d_worst_early - metrics.d_worst_final;
    features.push_back(distance_improvement);                 // 10: Distance improvement
    
    // Relative improvement rate
    float relative_improvement = (metrics.d_worst_early > eps) ? 
                                distance_improvement / metrics.d_worst_early : 0.0f;
    features.push_back(relative_improvement);                  // 11: Relative improvement rate
    
    // Worst/Best ratio
    float worst_best_ratio = (metrics.d_best_cand_final > eps) ? 
                             metrics.d_worst_final / metrics.d_best_cand_final : 0.0f;
    features.push_back(worst_best_ratio);                      // 12: Worst/Best ratio
    
    // Early/Final worst ratio
    float early_final_ratio = (metrics.d_worst_final > eps) ? 
                              metrics.d_worst_early / metrics.d_worst_final : 0.0f;
    features.push_back(early_final_ratio);                     // 13: Early/Final worst ratio
    
    // ========== DARTH: Convergence Indicators ==========
    // Progress rate (same as r_early, but explicit)
    float progress_rate = (metrics.S > 0) ? (float)metrics.t_last_improve / (float)metrics.S : 0.0f;
    features.push_back(progress_rate);                         // 14: Progress rate
    
    // Distance computation efficiency (visits per distance computation)
    float dist_comp_efficiency = (ndc > 0) ? (float)metrics.S / (float)ndc : 0.0f;
    features.push_back(dist_comp_efficiency);                 // 15: Distance computation efficiency
    
    // Improvement frequency (how often we improve)
    float improvement_freq = (metrics.S > 0 && metrics.t_last_improve > 0) ? 
                             (float)metrics.S / (float)metrics.t_last_improve : 0.0f;
    features.push_back(improvement_freq);                      // 16: Improvement frequency
    
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
    }
    
    std::string train_query_path = paths["train_query_path"];
    std::string train_gt_path = paths["train_gt_path"];
    std::string test_query_path = paths["test_query_path"];
    std::string test_gt_path = paths["test_gt_path"];
    std::string index_path = paths["index_path"];
    std::string result_path = paths["result_path"];
    std::string metric_str = paths["metric"];
    size_t k = paths.count("K") ? std::stoi(paths["K"]) : 100;
    size_t num_train_queries = 5000;  // Use 5K for faster testing, can increase to 10M for final run
    size_t num_test_queries = paths.count("num_queries") ? std::stoi(paths["num_queries"]) : 1000;
    
    size_t train_number = 0, test_number = 0;
    size_t train_gt_dim = 0, test_gt_dim = 0, vecdim = 0;
    
    // Load training data
    auto train_query = LoadData<float>(train_query_path, train_number, vecdim);
    auto train_gt = LoadData<int>(train_gt_path, train_number, train_gt_dim);
    
    // Load test data
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
    
    // Load index
    auto hnsw_base = new HNSW_NGFix<float>(metric, index_path);
    hnsw_base->printGraphInfo();
    
    // Training phase: collect features and targets
    std::cout << "Collecting training data..." << std::endl;
    std::vector<std::vector<float>> train_all_features;
    std::vector<float> train_hardness;  // Target: hardness = 1.0 - recall
    
    size_t actual_train = std::min(num_train_queries, train_number);
    size_t ef_train = 100;
    
    for(size_t i = 0; i < actual_train; ++i) {
        if(i % 100 == 0) {
            std::cout << "Training query " << i << "/" << actual_train << std::endl;
        }
        
        auto query_data = train_query + i * vecdim;
        auto gt = train_gt + i * train_gt_dim;
        
        // Search with lightweight metrics to get all required features
        size_t ndc = 0;  // Will be updated by searchKnnWithLightweightMetrics
        auto [results, ndc_result, lw_metrics] = hnsw_base->searchKnnWithLightweightMetrics(
            query_data, k, ef_train, ndc, 0.2f);
        
        // Calculate actual hardness (target for training)
        float actual_recall = CalculateRecall(results, gt, k);
        float actual_hardness = 1.0f - actual_recall;
        
        // Extract DARTH features from lightweight metrics
        // lw_metrics provides: S, t_last_improve, r_visit, r_early, d_worst_early, d_worst_final, d_best_cand_final
        // ndc_result provides: number of distance computations
        // ef_train provides: efSearch parameter
        auto all_features = extractAllFeatures(lw_metrics, ndc_result, ef_train);
        train_all_features.push_back(all_features);
        train_hardness.push_back(actual_hardness);
    }
    
    std::cout << "Training data collected. Exploring DARTH feature combinations..." << std::endl;
    
    // DARTH-inspired feature combinations based on the paper
    // Features: Search Progress (0-5), Distance Statistics (6-13), Convergence Indicators (14-16)
    
    // Helper function to create feature mask
    auto make_mask = [](const std::vector<size_t>& indices, size_t total) {
        std::vector<bool> mask(total, false);
        for(size_t idx : indices) {
            if(idx < total) mask[idx] = true;
        }
        return mask;
    };
    
    std::vector<std::vector<bool>> feature_combinations = {
        // ========== DARTH: Core Search Progress Features ==========
        make_mask({0, 1}, 17),      // 0: S + t_last_improve
        make_mask({2, 3}, 17),      // 1: r_visit + r_early
        make_mask({0, 2, 3}, 17),   // 2: S + r_visit + r_early
        make_mask({0, 1, 2, 3}, 17), // 3: All core search progress
        
        // ========== DARTH: Core Distance Statistics ==========
        make_mask({6, 7}, 17),      // 4: d_worst_early + d_worst_final
        make_mask({7, 8}, 17),      // 5: d_worst_final + d_best_cand_final
        make_mask({6, 7, 8}, 17),   // 6: All distance statistics
        make_mask({9, 10}, 17),     // 7: distance_range + distance_improvement
        make_mask({11, 12}, 17),    // 8: relative_improvement + worst/best_ratio
        
        // ========== DARTH: Core Convergence Indicators ==========
        make_mask({14, 15}, 17),    // 9: progress_rate + dist_comp_efficiency
        make_mask({14, 16}, 17),    // 10: progress_rate + improvement_freq
        make_mask({14, 15, 16}, 17), // 11: All convergence indicators
        
        // ========== DARTH: Search Progress + Distance Statistics ==========
        make_mask({0, 2, 3, 6, 7}, 17),  // 12: Search progress + distance stats
        make_mask({2, 3, 9, 10}, 17),    // 13: r_visit + r_early + distance range/improvement
        make_mask({0, 1, 6, 7, 8}, 17),  // 14: S + t_last_improve + all distances
        
        // ========== DARTH: Search Progress + Convergence ==========
        make_mask({2, 3, 14, 15}, 17),   // 15: r_visit + r_early + convergence
        make_mask({0, 1, 14, 16}, 17),   // 16: S + t_last_improve + convergence
        
        // ========== DARTH: Distance Statistics + Convergence ==========
        make_mask({6, 7, 11, 14}, 17),   // 17: Distance stats + relative improvement + progress
        make_mask({9, 10, 15, 16}, 17),  // 18: Distance range/improvement + convergence
        
        // ========== DARTH: Comprehensive Combinations ==========
        make_mask({0, 2, 3, 6, 7, 14}, 17),  // 19: Core search + distance + convergence
        make_mask({2, 3, 6, 7, 9, 10, 14}, 17), // 20: r_visit/r_early + distances + convergence
        make_mask({0, 1, 2, 3, 6, 7, 8, 14, 15, 16}, 17), // 21: Most features
        
        // ========== DARTH: With efSearch ==========
        make_mask({2, 3, 5}, 17),        // 22: r_visit + r_early + efSearch
        make_mask({0, 2, 3, 5}, 17),     // 23: S + r_visit + r_early + efSearch
        make_mask({0, 1, 2, 3, 5, 14}, 17), // 24: Search progress + efSearch + convergence
        
        // ========== DARTH: All Features ==========
        make_mask({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 17), // 25: All 17 features
    };
    
    // Evaluate each combination with cross-validation for better generalization
    std::vector<std::pair<float, size_t>> combination_scores;  // (MSE, index)
    
    // Use 80% for training, 20% for validation
    size_t val_size = train_all_features.size() / 5;
    size_t train_size = train_all_features.size() - val_size;
    
    std::vector<std::vector<float>> train_features(train_all_features.begin(), train_all_features.begin() + train_size);
    std::vector<std::vector<float>> val_features(train_all_features.begin() + train_size, train_all_features.end());
    std::vector<float> train_targets(train_hardness.begin(), train_hardness.begin() + train_size);
    std::vector<float> val_targets(train_hardness.begin() + train_size, train_hardness.end());
    
    std::cout << "Using " << train_size << " samples for training, " << val_size << " for validation" << std::endl;
    
    for(size_t combo_idx = 0; combo_idx < feature_combinations.size(); ++combo_idx) {
        const auto& mask = feature_combinations[combo_idx];
        
        std::cout << "\nTesting combination " << (combo_idx + 1) << "/" << feature_combinations.size() << ": ";
        for(size_t i = 0; i < mask.size(); ++i) {
            if(mask[i]) {
                std::cout << feature_names[i] << " ";
            }
        }
        std::cout << std::endl;
        
        HardnessPredictor predictor;
        predictor.setFeatureMask(mask);
        
        try {
            // Train on training set
            predictor.train(train_features, train_targets);
            // Evaluate on validation set
            float val_mse = predictor.getMSE(val_features, val_targets);
            combination_scores.push_back({val_mse, combo_idx});
            std::cout << "  Validation MSE: " << val_mse << std::endl;
        } catch(...) {
            combination_scores.push_back({1.0f, combo_idx});
            std::cout << "  Failed" << std::endl;
        }
    }
    
    // Sort by MSE (lower is better)
    std::sort(combination_scores.begin(), combination_scores.end());
    
    std::cout << "\n=== Top 5 Feature Combinations ===" << std::endl;
    for(size_t i = 0; i < std::min((size_t)5, combination_scores.size()); ++i) {
        size_t best_idx = combination_scores[i].second;
        const auto& mask = feature_combinations[best_idx];
        std::cout << "Rank " << (i+1) << " (MSE: " << combination_scores[i].first << "): ";
        for(size_t j = 0; j < mask.size(); ++j) {
            if(mask[j]) {
                std::cout << feature_names[j] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // Use best combination for testing
    size_t best_combo_idx = combination_scores[0].second;
    const auto& best_mask = feature_combinations[best_combo_idx];
    
    std::cout << "\nUsing best combination for final training..." << std::endl;
    HardnessPredictor best_predictor;
    best_predictor.setFeatureMask(best_mask);
    // Retrain on full dataset for final model
    best_predictor.train(train_all_features, train_hardness);
    
    // Testing phase
    size_t actual_test = std::min(num_test_queries, test_number);
    size_t ef_test = 100;
    
    std::vector<float> test_recalls;
    std::vector<float> test_hardness;
    std::vector<float> predicted_hardness;
    
    for(size_t i = 0; i < actual_test; ++i) {
        if(i % 100 == 0) {
            std::cout << "Testing query " << i << "/" << actual_test << std::endl;
        }
        
        auto query_data = test_query + i * vecdim;
        auto gt = test_gt + i * test_gt_dim;
        
        // Search with lightweight metrics to get all required features
        size_t ndc = 0;  // Will be updated by searchKnnWithLightweightMetrics
        auto [results, ndc_result, lw_metrics] = hnsw_base->searchKnnWithLightweightMetrics(
            query_data, k, ef_test, ndc, 0.2f);
        
        // Calculate actual recall for comparison
        float actual_recall = CalculateRecall(results, gt, k);
        float actual_hard = 1.0f - actual_recall;
        
        // Extract DARTH features and predict hardness
        // lw_metrics contains: S, t_last_improve, r_visit, r_early, d_worst_early, d_worst_final, d_best_cand_final
        // ndc_result contains: number of distance computations
        // ef_test provides: efSearch parameter
        auto all_features = extractAllFeatures(lw_metrics, ndc_result, ef_test);
        float pred_hard = best_predictor.predict(all_features);
        
        test_recalls.push_back(actual_recall);
        test_hardness.push_back(actual_hard);
        predicted_hardness.push_back(pred_hard);
    }
    
    // Write results
    std::ofstream output(result_path);
    output << std::fixed << std::setprecision(6);
    output << "{\n";
    output << "  \"num_queries\": " << actual_test << ",\n";
    output << "  \"k\": " << k << ",\n";
    output << "  \"best_feature_combination\": [";
    bool first = true;
    for(size_t i = 0; i < best_mask.size(); ++i) {
        if(best_mask[i]) {
            if(!first) output << ", ";
            output << "\"" << feature_names[i] << "\"";
            first = false;
        }
    }
    output << "],\n";
    output << "  \"best_mse\": " << combination_scores[0].first << ",\n";
    output << "  \"queries\": [\n";
    
    for(size_t i = 0; i < actual_test; ++i) {
        output << "    {\n";
        output << "      \"query_id\": " << i << ",\n";
        output << "      \"recall\": " << test_recalls[i] << ",\n";
        output << "      \"hardness\": " << test_hardness[i] << ",\n";
        output << "      \"predicted_hardness\": " << predicted_hardness[i] << "\n";
        output << "    }";
        if(i < actual_test - 1) {
            output << ",";
        }
        output << "\n";
    }
    
    output << "  ]\n";
    output << "}\n";
    output.close();
    
    std::cout << "\nResults written to: " << result_path << std::endl;
    std::cout << "Best feature combination MSE: " << combination_scores[0].first << std::endl;
    
    delete[] train_query;
    delete[] train_gt;
    delete[] test_query;
    delete[] test_gt;
    delete hnsw_base;
    
    return 0;
}

