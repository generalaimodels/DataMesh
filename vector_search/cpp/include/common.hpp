/**
 * @file common.hpp
 * @brief Common definitions, types, and utilities for VectorSearch core
 * 
 * Provides:
 *   - Platform detection and SIMD capability
 *   - Memory alignment utilities
 *   - Common type definitions
 *   - Error handling
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>

// =============================================================================
// PLATFORM DETECTION
// =============================================================================

#if defined(_MSC_VER)
    #define VS_MSVC 1
    #define VS_INLINE __forceinline
    #define VS_NOINLINE __declspec(noinline)
    #define VS_ALIGN(x) __declspec(align(x))
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    #define VS_GCC 1
    #define VS_INLINE __attribute__((always_inline)) inline
    #define VS_NOINLINE __attribute__((noinline))
    #define VS_ALIGN(x) __attribute__((aligned(x)))
#else
    #define VS_INLINE inline
    #define VS_NOINLINE
    #define VS_ALIGN(x)
#endif

// SIMD capability macros (set by CMake)
#ifndef VECTORSEARCH_AVX512
    #define VECTORSEARCH_AVX512 0
#endif
#ifndef VECTORSEARCH_AVX2
    #define VECTORSEARCH_AVX2 0
#endif
#ifndef VECTORSEARCH_FMA
    #define VECTORSEARCH_FMA 0
#endif

// =============================================================================
// SIMD HEADERS
// =============================================================================

#if VECTORSEARCH_AVX512 || VECTORSEARCH_AVX2
    #include <immintrin.h>
#endif

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

namespace vectorsearch {

/// Vector element type (32-bit float for SIMD efficiency)
using scalar_t = float;

/// Vector ID type (64-bit for large indices)
using vid_t = uint64_t;

/// Distance/similarity score type
using score_t = float;

/// Dimension type
using dim_t = uint32_t;

/// Index into vectors array
using idx_t = uint32_t;

/// Level type for HNSW
using level_t = uint8_t;

// =============================================================================
// MEMORY ALIGNMENT
// =============================================================================

/// Cache line size (assumed 64 bytes)
constexpr size_t CACHE_LINE_SIZE = 64;

/// SIMD alignment (AVX-512 = 64, AVX2 = 32)
#if VECTORSEARCH_AVX512
    constexpr size_t SIMD_ALIGNMENT = 64;
#elif VECTORSEARCH_AVX2
    constexpr size_t SIMD_ALIGNMENT = 32;
#else
    constexpr size_t SIMD_ALIGNMENT = 16;
#endif

/// Aligned memory allocator
template<typename T, size_t Alignment = SIMD_ALIGNMENT>
struct AlignedAllocator {
    using value_type = T;
    
    AlignedAllocator() = default;
    
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    
    T* allocate(size_t n) {
        void* ptr = nullptr;
        #if defined(_MSC_VER)
            ptr = _aligned_malloc(n * sizeof(T), Alignment);
        #else
            if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
                ptr = nullptr;
            }
        #endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, size_t) noexcept {
        #if defined(_MSC_VER)
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }
};

/// Aligned vector type
template<typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

// =============================================================================
// RESULT TYPE (Rust-style)
// =============================================================================

/// Error type for operations
struct Error {
    std::string message;
    int code;
    
    Error(std::string msg, int c = -1) : message(std::move(msg)), code(c) {}
};

/// Result type for fallible operations
template<typename T>
class Result {
public:
    /// Success constructor
    static Result Ok(T value) {
        Result r;
        r.value_ = std::move(value);
        r.is_ok_ = true;
        return r;
    }
    
    /// Error constructor
    static Result Err(Error error) {
        Result r;
        r.error_ = std::move(error);
        r.is_ok_ = false;
        return r;
    }
    
    bool is_ok() const { return is_ok_; }
    bool is_err() const { return !is_ok_; }
    
    T& unwrap() {
        if (!is_ok_) throw std::runtime_error(error_.message);
        return value_;
    }
    
    const T& unwrap() const {
        if (!is_ok_) throw std::runtime_error(error_.message);
        return value_;
    }
    
    T unwrap_or(T default_val) const {
        return is_ok_ ? value_ : default_val;
    }
    
    const Error& error() const { return error_; }
    
private:
    Result() = default;
    T value_;
    Error error_{""};
    bool is_ok_ = false;
};

// =============================================================================
// DISTANCE METRICS
// =============================================================================

enum class MetricType : uint8_t {
    L2 = 0,           // Euclidean distance (lower = more similar)
    COSINE = 1,       // Cosine similarity (higher = more similar)
    INNER_PRODUCT = 2 // Dot product (higher = more similar)
};

/// Check if metric is similarity-based (higher = better)
inline bool is_similarity_metric(MetricType metric) {
    return metric == MetricType::COSINE || metric == MetricType::INNER_PRODUCT;
}

// =============================================================================
// SEARCH RESULT
// =============================================================================

/// Single search result
struct SearchResult {
    vid_t id;
    score_t score;
    idx_t internal_idx;
    
    bool operator<(const SearchResult& other) const {
        return score < other.score;
    }
    
    bool operator>(const SearchResult& other) const {
        return score > other.score;
    }
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Prefetch memory for read
VS_INLINE void prefetch_read(const void* ptr) {
    #if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(ptr, 0, 3);
    #elif defined(_MSC_VER) && (VECTORSEARCH_AVX2 || VECTORSEARCH_AVX512)
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
    #endif
}

/// Prefetch memory for write
VS_INLINE void prefetch_write(void* ptr) {
    #if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(ptr, 1, 3);
    #elif defined(_MSC_VER) && (VECTORSEARCH_AVX2 || VECTORSEARCH_AVX512)
        _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
    #endif
}

/// Round up to alignment
constexpr size_t align_up(size_t n, size_t alignment) {
    return (n + alignment - 1) & ~(alignment - 1);
}

} // namespace vectorsearch
