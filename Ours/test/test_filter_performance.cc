#include "ourslib/graph/hnsw_ours.h"
#include <iostream>
#include <chrono>
#include <random>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <iomanip>

using namespace ours;

// Test performance of different lookup methods
void TestFilterPerformance() {
    const size_t NUM_NODES = 10000000;  // 10M nodes
    const size_t PENDING_SIZE = 10000;  // 10K pending nodes
    const size_t NUM_LOOKUPS = 1000000;  // 1M lookups
    
    std::cout << "=== Filter Performance Test ===" << std::endl;
    std::cout << "Total nodes: " << NUM_NODES << std::endl;
    std::cout << "Pending nodes: " << PENDING_SIZE << std::endl;
    std::cout << "Lookups: " << NUM_LOOKUPS << std::endl;
    std::cout << std::endl;
    
    // Generate random pending nodes
    std::mt19937 rng(42);
    std::uniform_int_distribution<id_t> node_dist(0, NUM_NODES - 1);
    
    std::unordered_set<id_t> pending_set;
    std::vector<id_t> pending_vec;
    
    while(pending_set.size() < PENDING_SIZE) {
        id_t node_id = node_dist(rng);
        if(pending_set.insert(node_id).second) {
            pending_vec.push_back(node_id);
        }
    }
    
    std::sort(pending_vec.begin(), pending_vec.end());
    
    // Generate lookup queries (mix of hits and misses)
    std::vector<id_t> lookup_queries;
    for(size_t i = 0; i < NUM_LOOKUPS; ++i) {
        if(i % 10 == 0) {
            // 10% hits
            lookup_queries.push_back(pending_vec[rng() % pending_vec.size()]);
        } else {
            // 90% misses
            id_t node_id = node_dist(rng);
            while(pending_set.find(node_id) != pending_set.end()) {
                node_id = node_dist(rng);
            }
            lookup_queries.push_back(node_id);
        }
    }
    
    std::cout << "Generated " << lookup_queries.size() << " lookup queries" << std::endl;
    std::cout << "Expected hits: ~" << NUM_LOOKUPS / 10 << std::endl;
    std::cout << std::endl;
    
    // Test 1: Direct hash map lookup (unordered_set)
    {
        std::cout << "Test 1: Direct Hash Map (unordered_set)" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t hits = 0;
        for(id_t node_id : lookup_queries) {
            if(pending_set.find(node_id) != pending_set.end()) {
                hits++;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Hits: " << hits << std::endl;
        std::cout << "  Time: " << duration.count() << " us" << std::endl;
        std::cout << "  Avg per lookup: " << (double)duration.count() / NUM_LOOKUPS << " us" << std::endl;
        std::cout << "  Avg per lookup: " << (double)duration.count() * 1000 / NUM_LOOKUPS << " ns" << std::endl;
        std::cout << std::endl;
    }
    
    // Test 2: Sorted vector + binary search
    {
        std::cout << "Test 2: Sorted Vector + Binary Search" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t hits = 0;
        for(id_t node_id : lookup_queries) {
            auto it = std::lower_bound(pending_vec.begin(), pending_vec.end(), node_id);
            if(it != pending_vec.end() && *it == node_id) {
                hits++;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Hits: " << hits << std::endl;
        std::cout << "  Time: " << duration.count() << " us" << std::endl;
        std::cout << "  Avg per lookup: " << (double)duration.count() / NUM_LOOKUPS << " us" << std::endl;
        std::cout << "  Avg per lookup: " << (double)duration.count() * 1000 / NUM_LOOKUPS << " ns" << std::endl;
        std::cout << std::endl;
    }
    
    // Test 3: Bloom Filter + Binary Search (current implementation)
    {
        std::cout << "Test 3: Bloom Filter + Binary Search (Current)" << std::endl;
        
        // Initialize bloom filter
        constexpr size_t BLOOM_FILTER_SIZE = 8192;
        constexpr size_t BLOOM_FILTER_BITS = BLOOM_FILTER_SIZE * 8;
        std::vector<uint8_t> bloom_filter(BLOOM_FILTER_SIZE, 0);
        
        auto BloomHash1 = [](id_t node_id) {
            return (node_id * 2654435761ULL) % BLOOM_FILTER_BITS;
        };
        auto BloomHash2 = [](id_t node_id) {
            return (node_id * 2246822519ULL) % BLOOM_FILTER_BITS;
        };
        auto BloomHash3 = [](id_t node_id) {
            return (node_id * 3266489917ULL) % BLOOM_FILTER_BITS;
        };
        
        // Add all pending nodes to bloom filter
        for(id_t node_id : pending_vec) {
            size_t h1 = BloomHash1(node_id);
            size_t h2 = BloomHash2(node_id);
            size_t h3 = BloomHash3(node_id);
            
            bloom_filter[h1 / 8] |= (1 << (h1 % 8));
            bloom_filter[h2 / 8] |= (1 << (h2 % 8));
            bloom_filter[h3 / 8] |= (1 << (h3 % 8));
        }
        
        auto BloomFilterMightContain = [&](id_t node_id) -> bool {
            size_t h1 = BloomHash1(node_id);
            size_t h2 = BloomHash2(node_id);
            size_t h3 = BloomHash3(node_id);
            
            size_t byte1 = h1 / 8;
            size_t bit1 = h1 % 8;
            size_t byte2 = h2 / 8;
            size_t bit2 = h2 % 8;
            size_t byte3 = h3 / 8;
            size_t bit3 = h3 % 8;
            
            return ((bloom_filter[byte1] >> bit1) & 1) && 
                   ((bloom_filter[byte2] >> bit2) & 1) && 
                   ((bloom_filter[byte3] >> bit3) & 1);
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t hits = 0;
        size_t bloom_checks = 0;
        size_t binary_searches = 0;
        
        for(id_t node_id : lookup_queries) {
            bloom_checks++;
            if(BloomFilterMightContain(node_id)) {
                // Bloom filter says might be pending, do binary search
                binary_searches++;
                auto it = std::lower_bound(pending_vec.begin(), pending_vec.end(), node_id);
                if(it != pending_vec.end() && *it == node_id) {
                    hits++;
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Hits: " << hits << std::endl;
        std::cout << "  Bloom checks: " << bloom_checks << std::endl;
        std::cout << "  Binary searches: " << binary_searches << " (" 
                  << (100.0 * binary_searches / bloom_checks) << "%)" << std::endl;
        std::cout << "  Time: " << duration.count() << " us" << std::endl;
        std::cout << "  Avg per lookup: " << (double)duration.count() / NUM_LOOKUPS << " us" << std::endl;
        std::cout << "  Avg per lookup: " << (double)duration.count() * 1000 / NUM_LOOKUPS << " ns" << std::endl;
        std::cout << std::endl;
    }
    
    // Test 4: Hash map with shared_lock (simulating actual usage)
    {
        std::cout << "Test 4: Hash Map with Shared Lock (Simulated)" << std::endl;
        std::shared_mutex lock;
        std::unordered_set<id_t> pending_set_locked = pending_set;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t hits = 0;
        for(id_t node_id : lookup_queries) {
            std::shared_lock<std::shared_mutex> shared_lock(lock);
            if(pending_set_locked.find(node_id) != pending_set_locked.end()) {
                hits++;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Hits: " << hits << std::endl;
        std::cout << "  Time: " << duration.count() << " us" << std::endl;
        std::cout << "  Avg per lookup: " << (double)duration.count() / NUM_LOOKUPS << " us" << std::endl;
        std::cout << "  Avg per lookup: " << (double)duration.count() * 1000 / NUM_LOOKUPS << " ns" << std::endl;
        std::cout << std::endl;
    }
}

int main() {
    TestFilterPerformance();
    return 0;
}

