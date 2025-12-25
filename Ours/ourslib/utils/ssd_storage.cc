#include "ssd_storage.h"
#include <stdexcept>
#include <algorithm>
#include <set>
#include <sys/stat.h>

namespace ours {

typedef unsigned int id_t;

SSDStorage::SSDStorage(const std::string& index_path, const std::string& vector_path,
                       size_t max_elements, size_t dim, size_t size_per_element)
    : index_path_(index_path), vector_path_(vector_path),
      max_elements_(max_elements), dim_(dim), size_per_element_(size_per_element),
      index_fd_(-1), vector_fd_(-1),
      index_mmap_(nullptr), vector_mmap_(nullptr),
      index_mmap_size_(0), vector_mmap_size_(0) {
    CalculateSizes();
}

SSDStorage::~SSDStorage() {
    Cleanup();
}

void SSDStorage::CalculateSizes() {
    // Calculate node size: neighbors array + metadata
    // For now, estimate max node size (will be adjusted based on actual usage)
    // Each node can have up to MEX neighbors (typically 48)
    size_t max_neighbors = 64;  // Conservative estimate
    node_size_ = sizeof(id_t) * (max_neighbors + 1) + sizeof(uint16_t) * max_neighbors;  // neighbors + ehs
    
    // Calculate nodes per page
    nodes_per_page_ = PAGE_SIZE / std::max(node_size_, size_per_element_);
    if(nodes_per_page_ == 0) nodes_per_page_ = 1;
    
    // Align to page boundaries
    size_t total_index_pages = (max_elements_ + nodes_per_page_ - 1) / nodes_per_page_;
    index_mmap_size_ = total_index_pages * PAGE_SIZE;
    
    size_t total_vector_pages = (max_elements_ * size_per_element_ + PAGE_SIZE - 1) / PAGE_SIZE;
    vector_mmap_size_ = total_vector_pages * PAGE_SIZE;
}

void SSDStorage::Initialize(bool create_new) {
    if(create_new) {
        // Create new files
        AllocateFiles();
    } else {
        // Open existing files
        index_fd_ = open(index_path_.c_str(), O_RDWR);
        if(index_fd_ < 0) {
            throw std::runtime_error("Failed to open index file: " + index_path_);
        }
        
        vector_fd_ = open(vector_path_.c_str(), O_RDWR);
        if(vector_fd_ < 0) {
            close(index_fd_);
            throw std::runtime_error("Failed to open vector file: " + vector_path_);
        }
        
        // Get file sizes
        struct stat st;
        fstat(index_fd_, &st);
        index_mmap_size_ = st.st_size;
        
        fstat(vector_fd_, &st);
        vector_mmap_size_ = st.st_size;
    }
    
    // Memory map files
    index_mmap_ = mmap(nullptr, index_mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, index_fd_, 0);
    if(index_mmap_ == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap index file");
    }
    
    vector_mmap_ = mmap(nullptr, vector_mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, vector_fd_, 0);
    if(vector_mmap_ == MAP_FAILED) {
        munmap(index_mmap_, index_mmap_size_);
        throw std::runtime_error("Failed to mmap vector file");
    }
    
    // Initialize page info
    size_t total_pages = (index_mmap_size_ + PAGE_SIZE - 1) / PAGE_SIZE;
    page_info_.resize(total_pages);
    for(size_t i = 0; i < total_pages; ++i) {
        page_info_[i].page_id = i;
        page_info_[i].dirty = false;
        page_info_[i].last_access = 0;
    }
}

void SSDStorage::AllocateFiles() {
    // Create and allocate index file
    index_fd_ = open(index_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if(index_fd_ < 0) {
        throw std::runtime_error("Failed to create index file: " + index_path_);
    }
    
    // Allocate space (write zeros to extend file)
    if(ftruncate(index_fd_, index_mmap_size_) < 0) {
        close(index_fd_);
        throw std::runtime_error("Failed to allocate index file space");
    }
    
    // Create and allocate vector file
    vector_fd_ = open(vector_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if(vector_fd_ < 0) {
        close(index_fd_);
        throw std::runtime_error("Failed to create vector file: " + vector_path_);
    }
    
    if(ftruncate(vector_fd_, vector_mmap_size_) < 0) {
        close(index_fd_);
        close(vector_fd_);
        throw std::runtime_error("Failed to allocate vector file space");
    }
}

void SSDStorage::Cleanup() {
    // Flush dirty pages before cleanup
    FlushDirtyPages();
    
    // Sync to ensure all data is written
    if(index_fd_ >= 0) {
        fsync(index_fd_);
    }
    if(vector_fd_ >= 0) {
        fsync(vector_fd_);
    }
    
    // Unmap memory
    if(index_mmap_ != nullptr && index_mmap_ != MAP_FAILED) {
        munmap(index_mmap_, index_mmap_size_);
    }
    if(vector_mmap_ != nullptr && vector_mmap_ != MAP_FAILED) {
        munmap(vector_mmap_, vector_mmap_size_);
    }
    
    // Close files
    if(index_fd_ >= 0) {
        close(index_fd_);
    }
    if(vector_fd_ >= 0) {
        close(vector_fd_);
    }
}

void* SSDStorage::GetVectorData(id_t node_id) {
    if(node_id >= max_elements_) {
        throw std::runtime_error("Node ID out of bounds");
    }
    
    size_t offset = node_id * size_per_element_ + 1;  // +1 for delete flag
    return (char*)vector_mmap_ + offset;
}

void SSDStorage::SetVectorData(id_t node_id, const void* data) {
    void* dst = GetVectorData(node_id);
    memcpy(dst, data, sizeof(float) * dim_);  // Assuming float type
    
    // Mark page as dirty
    size_t page_id = GetPageId(node_id);
    std::lock_guard<std::mutex> lock(cache_mutex_);
    page_info_[page_id].dirty = true;
    page_info_[page_id].last_access = access_counter_.fetch_add(1);
}

void* SSDStorage::GetNodeNeighbors(id_t node_id, size_t& capacity, size_t& ngfix_capacity) {
    if(node_id >= max_elements_) {
        throw std::runtime_error("Node ID out of bounds");
    }
    
    // Calculate page and offset
    size_t page_id = GetPageId(node_id);
    size_t page_offset = GetPageOffset(node_id);
    
    // Get pointer to node data in mapped memory
    void* page_start = (char*)index_mmap_ + page_id * PAGE_SIZE;
    void* node_data = (char*)page_start + page_offset;
    
    // Update access time
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        page_info_[page_id].last_access = access_counter_.fetch_add(1);
    }
    
    // Read metadata from node
    id_t* neighbors = (id_t*)node_data;
    uint8_t* meta = (uint8_t*)neighbors;
    capacity = meta[0];
    ngfix_capacity = meta[2];
    
    return neighbors;
}

void SSDStorage::SetNodeNeighbors(id_t node_id, const void* neighbors, size_t capacity, size_t ngfix_capacity) {
    if(node_id >= max_elements_) {
        throw std::runtime_error("Node ID out of bounds");
    }
    
    // Calculate page and offset
    size_t page_id = GetPageId(node_id);
    size_t page_offset = GetPageOffset(node_id);
    
    // Get pointer to node data in mapped memory
    void* page_start = (char*)index_mmap_ + page_id * PAGE_SIZE;
    void* node_data = (char*)page_start + page_offset;
    
    // Calculate size to copy
    size_t neighbors_size = sizeof(id_t) * (capacity + 1);
    size_t ehs_size = ngfix_capacity > 0 ? sizeof(uint16_t) * ngfix_capacity : 0;
    size_t total_size = neighbors_size + ehs_size;
    
    // Copy data
    memcpy(node_data, neighbors, neighbors_size);
    if(ehs_size > 0) {
        memcpy((char*)node_data + neighbors_size, 
               (char*)neighbors + neighbors_size, ehs_size);
    }
    
    // Mark page as dirty
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        page_info_[page_id].dirty = true;
        page_info_[page_id].last_access = access_counter_.fetch_add(1);
    }
}

void SSDStorage::FlushPage(size_t page_id) {
    if(page_id >= page_info_.size()) {
        return;
    }
    
    // Use msync to flush specific page
    void* page_start = (char*)index_mmap_ + page_id * PAGE_SIZE;
    msync(page_start, PAGE_SIZE, MS_ASYNC);
    
    // Mark as clean
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        page_info_[page_id].dirty = false;
    }
}

void SSDStorage::FlushDirtyPages() {
    std::set<size_t> dirty_pages_copy;
    
    // Collect dirty pages
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for(size_t i = 0; i < page_info_.size(); ++i) {
            if(page_info_[i].dirty) {
                dirty_pages_copy.insert(i);
            }
        }
    }
    
    // Flush each dirty page
    for(size_t page_id : dirty_pages_copy) {
        FlushPage(page_id);
    }
    
    // Sync vector file as well
    if(vector_mmap_ != nullptr && vector_mmap_ != MAP_FAILED) {
        msync(vector_mmap_, vector_mmap_size_, MS_ASYNC);
    }
}

void SSDStorage::Sync() {
    // Flush all dirty pages
    FlushDirtyPages();
    
    // Force sync to disk
    if(index_fd_ >= 0) {
        fsync(index_fd_);
    }
    if(vector_fd_ >= 0) {
        fsync(vector_fd_);
    }
}

} // namespace ours

