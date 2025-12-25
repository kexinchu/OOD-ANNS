#include "lightgbm_predictor.h"
#include <iostream>
#include <cmath>
#include <cstring>

namespace ours {

bool LightGBMPredictor::loadFromFile(const std::string& model_path) {
    std::ifstream file(model_path);
    if(!file.is_open()) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return false;
    }
    
    trees_.clear();
    num_trees_ = 0;
    shrinkage_ = 0.1f;  // Default shrinkage
    
    std::string line;
    bool in_tree = false;
    LightGBMTree current_tree;
    
    while(std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if(line.empty()) continue;
        
        // Parse shrinkage
        if(line.find("shrinkage=") == 0) {
            std::string val = line.substr(10);
            shrinkage_ = std::stof(val);
            continue;
        }
        
        // Parse tree start
        if(line.find("Tree=") == 0) {
            if(in_tree) {
                // Save previous tree
                trees_.push_back(current_tree);
                num_trees_++;
            }
            in_tree = true;
            current_tree = LightGBMTree();
            continue;
        }
        
        if(!in_tree) continue;
        
        // Parse tree end (empty line or next Tree=)
        if(line.find("Tree=") == 0) {
            trees_.push_back(current_tree);
            num_trees_++;
            current_tree = LightGBMTree();
            continue;
        }
        
        // Parse node
        LightGBMNode node;
        if(parseNode(line, node)) {
            current_tree.nodes.push_back(node);
        }
    }
    
    // Save last tree
    if(in_tree && !current_tree.nodes.empty()) {
        trees_.push_back(current_tree);
        num_trees_++;
    }
    
    file.close();
    
    if(trees_.empty()) {
        std::cerr << "No trees found in model file" << std::endl;
        return false;
    }
    
    std::cout << "Loaded LightGBM model: " << num_trees_ << " trees, shrinkage=" << shrinkage_ << std::endl;
    return true;
}

bool LightGBMPredictor::parseNode(const std::string& line, LightGBMNode& node) {
    // LightGBM model format can vary. Try to parse as:
    // split_feature threshold left_child right_child leaf_value
    // Or sometimes just numbers separated by spaces
    
    std::istringstream iss(line);
    std::vector<float> values;
    float val;
    
    // Try to parse all numbers
    while(iss >> val) {
        values.push_back(val);
    }
    
    if(values.empty()) {
        return false;
    }
    
    // If we have at least 5 values, assume standard format
    if(values.size() >= 5) {
        node.split_feature = (int)values[0];
        node.threshold = values[1];
        node.left_child = (int)values[2];
        node.right_child = (int)values[3];
        node.leaf_value = values[4];
        
        // Check if it's a leaf node
        if(node.split_feature == -1) {
            node.left_child = -1;
            node.right_child = -1;
        }
        
        return true;
    } else if(values.size() >= 1) {
        // Try to interpret as leaf node
        int split_feature = (int)values[0];
        if(split_feature == -1 || values.size() == 1) {
            node.split_feature = -1;
            node.threshold = 0.0f;
            node.left_child = -1;
            node.right_child = -1;
            node.leaf_value = values.size() > 1 ? values.back() : 0.0f;
            return true;
        }
    }
    
    return false;
}

float LightGBMPredictor::predict(const std::vector<float>& features) const {
    if(trees_.empty()) {
        return 0.0f;
    }
    
    float prediction = 0.0f;
    
    for(const auto& tree : trees_) {
        prediction += tree.predict(features);
    }
    
    // Apply shrinkage
    prediction *= shrinkage_;
    
    // Clamp to [0, 1] for hardness prediction
    return std::max(0.0f, std::min(1.0f, prediction));
}

} // namespace ours

