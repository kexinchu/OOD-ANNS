#pragma once

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ours {

// LightGBM tree node structure
struct LightGBMNode {
    int split_feature;      // Feature index to split on (-1 for leaf)
    float threshold;        // Split threshold
    int left_child;         // Left child index (-1 if leaf)
    int right_child;        // Right child index (-1 if leaf)
    float leaf_value;       // Leaf value (only valid if split_feature == -1)
    bool is_leaf() const { return split_feature == -1; }
};

// LightGBM tree structure
struct LightGBMTree {
    std::vector<LightGBMNode> nodes;
    
    // Predict for a single sample
    float predict(const std::vector<float>& features) const {
        if(nodes.empty()) return 0.0f;
        
        int node_idx = 0;
        while(!nodes[node_idx].is_leaf()) {
            const auto& node = nodes[node_idx];
            if(node.split_feature < 0 || node.split_feature >= (int)features.size()) {
                // Invalid feature index, return 0
                return 0.0f;
            }
            
            if(features[node.split_feature] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
            
            if(node_idx < 0 || node_idx >= (int)nodes.size()) {
                // Invalid node index, return 0
                return 0.0f;
            }
        }
        
        return nodes[node_idx].leaf_value;
    }
};

// LightGBM predictor class
class LightGBMPredictor {
public:
    LightGBMPredictor() : shrinkage_(0.1f), num_trees_(0) {}
    
    // Load model from file
    bool loadFromFile(const std::string& model_path);
    
    // Predict hardness for given features
    float predict(const std::vector<float>& features) const;
    
    // Check if model is loaded
    bool isLoaded() const { return !trees_.empty(); }
    
    // Get number of trees
    size_t numTrees() const { return num_trees_; }
    
private:
    std::vector<LightGBMTree> trees_;
    float shrinkage_;
    size_t num_trees_;
    
    // Parse a single tree from model file
    bool parseTree(std::istream& is, LightGBMTree& tree);
    
    // Parse a single node from model file
    bool parseNode(const std::string& line, LightGBMNode& node);
};

} // namespace ours

