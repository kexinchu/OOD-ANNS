#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

namespace ours {

// Forward declaration
typedef unsigned int id_t;

// SSD-based storage manager for HNSW index
// Similar to DiskANN, stores index and vectors on SSD with page-based writes
class SSDStorage {
public:
    static constexpr size_t PAGE_SIZE = 4096;  // 4KB page size
    static constexpr size_t DEFAULT_PAGE_CACHE_SIZE = 1024 * 1024 * 1024;  // 1GB page cache
    
    struct PageInfo {
        size_t page_id;
        bool dirty;  // Whether page has been modified
        uint64_t last_access;  // For LRU eviction
    };
    
    SSDStorage(const std::string& index_path, const std::string& vector_path, 
               size_t max_elements, size_t dim, size_t size_per_element);
    ~SSDStorage();
    
    // Initialize SSD storage (create files if needed)
    void Initialize(bool create_new = false);
    
    // Get vector data (with caching)
    void* GetVectorData(id_t node_id);
    
    // Set vector data (marks page as dirty)
    void SetVectorData(id_t node_id, const void* data);
    
    // Get node neighbors (with caching)
    // Returns pointer to neighbors array, and sets capacity and ngfix_capacity
    void* GetNodeNeighbors(id_t node_id, size_t& capacity, size_t& ngfix_capacity);
    
    // Set node neighbors (marks page as dirty, page-based write)
    void SetNodeNeighbors(id_t node_id, const void* neighbors, size_t capacity, size_t ngfix_capacity);
    
    // Flush dirty pages to SSD (batch write)
    void FlushDirtyPages();
    
    // Flush specific page to SSD
    void FlushPage(size_t page_id);
    
    // Sync all data to SSD
    void Sync();
    
    // Get page ID for a node
    size_t GetPageId(id_t node_id) const {
        return node_id / nodes_per_page_;
    }
    
    // Get offset within page for a node
    size_t GetPageOffset(id_t node_id) const {
        return (node_id % nodes_per_page_) * node_size_;
    }
    
private:
    std::string index_path_;
    std::string vector_path_;
    size_t max_elements_;
    size_t dim_;
    size_t size_per_element_;
    size_t node_size_;  // Size of node data (neighbors + metadata)
    size_t nodes_per_page_;  // Number of nodes per page
    
    // File descriptors
    int index_fd_;
    int vector_fd_;
    
    // Memory-mapped regions
    void* index_mmap_;
    void* vector_mmap_;
    size_t index_mmap_size_;
    size_t vector_mmap_size_;
    
    // Page cache management
    std::vector<PageInfo> page_info_;
    std::mutex cache_mutex_;
    std::atomic<uint64_t> access_counter_{0};
    
    // Calculate sizes
    void CalculateSizes();
    
    // Allocate and map files
    void AllocateFiles();
    
    // Unmap and close files
    void Cleanup();
};

} // namespace ours

