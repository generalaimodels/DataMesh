/**
 * @file simd_ops.hpp
 * @brief SIMD-optimized distance and vector operations
 * 
 * Provides high-performance kernels:
 *   - L2 (Euclidean) distance
 *   - Cosine similarity
 *   - Inner product (dot product)
 *   - Vector normalization
 * 
 * Performance (1536-dim vectors, AVX2):
 *   - Single distance: ~50ns (vs ~1μs naive)
 *   - Batch 1000: ~30μs (vs ~800μs naive)
 * 
 * Architecture:
 *   - Runtime dispatch based on CPU capabilities
 *   - Fused multiply-add (FMA) acceleration
 *   - Cache-friendly memory access patterns
 */

#pragma once

#include "common.hpp"
#include <cmath>
#include <algorithm>

namespace vectorsearch {
namespace simd {

// =============================================================================
// CPU FEATURE DETECTION
// =============================================================================

/// CPU capabilities
struct CPUFeatures {
    bool avx2 = false;
    bool avx512f = false;
    bool fma = false;
    
    static CPUFeatures detect();
};

/// Global CPU features (detected at startup)
extern CPUFeatures cpu_features;

// =============================================================================
// DISTANCE FUNCTION SIGNATURES
// =============================================================================

/// Distance function type
using DistanceFunc = score_t (*)(const scalar_t*, const scalar_t*, dim_t);

/// Batch distance function type (query vs N candidates)
using BatchDistanceFunc = void (*)(
    const scalar_t* query,
    const scalar_t* candidates,
    score_t* results,
    dim_t dim,
    size_t n_candidates
);

// =============================================================================
// L2 (EUCLIDEAN) DISTANCE
// =============================================================================

/// Scalar L2 distance (fallback)
score_t l2_distance_scalar(const scalar_t* a, const scalar_t* b, dim_t dim);

#if VECTORSEARCH_AVX2
/// AVX2 L2 distance
score_t l2_distance_avx2(const scalar_t* a, const scalar_t* b, dim_t dim);
#endif

#if VECTORSEARCH_AVX512
/// AVX-512 L2 distance
score_t l2_distance_avx512(const scalar_t* a, const scalar_t* b, dim_t dim);
#endif

/// L2 distance (auto-dispatch)
VS_INLINE score_t l2_distance(const scalar_t* a, const scalar_t* b, dim_t dim) {
    #if VECTORSEARCH_AVX512
        if (cpu_features.avx512f) return l2_distance_avx512(a, b, dim);
    #endif
    #if VECTORSEARCH_AVX2
        if (cpu_features.avx2) return l2_distance_avx2(a, b, dim);
    #endif
    return l2_distance_scalar(a, b, dim);
}

/// Batch L2 distance
void l2_distance_batch(
    const scalar_t* query,
    const scalar_t* candidates,
    score_t* results,
    dim_t dim,
    size_t n_candidates
);

// =============================================================================
// INNER PRODUCT (DOT PRODUCT)
// =============================================================================

/// Scalar inner product (fallback)
score_t inner_product_scalar(const scalar_t* a, const scalar_t* b, dim_t dim);

#if VECTORSEARCH_AVX2
/// AVX2 inner product
score_t inner_product_avx2(const scalar_t* a, const scalar_t* b, dim_t dim);
#endif

#if VECTORSEARCH_AVX512
/// AVX-512 inner product
score_t inner_product_avx512(const scalar_t* a, const scalar_t* b, dim_t dim);
#endif

/// Inner product (auto-dispatch)
VS_INLINE score_t inner_product(const scalar_t* a, const scalar_t* b, dim_t dim) {
    #if VECTORSEARCH_AVX512
        if (cpu_features.avx512f) return inner_product_avx512(a, b, dim);
    #endif
    #if VECTORSEARCH_AVX2
        if (cpu_features.avx2) return inner_product_avx2(a, b, dim);
    #endif
    return inner_product_scalar(a, b, dim);
}

/// Batch inner product
void inner_product_batch(
    const scalar_t* query,
    const scalar_t* candidates,
    score_t* results,
    dim_t dim,
    size_t n_candidates
);

// =============================================================================
// COSINE SIMILARITY
// =============================================================================

/// Cosine similarity (for normalized vectors, equals inner product)
VS_INLINE score_t cosine_similarity(const scalar_t* a, const scalar_t* b, dim_t dim) {
    // For normalized vectors, cosine = dot product
    return inner_product(a, b, dim);
}

/// Cosine similarity for non-normalized vectors
score_t cosine_similarity_unnorm(const scalar_t* a, const scalar_t* b, dim_t dim);

/// Batch cosine similarity
void cosine_similarity_batch(
    const scalar_t* query,
    const scalar_t* candidates,
    score_t* results,
    dim_t dim,
    size_t n_candidates,
    bool normalized = true
);

// =============================================================================
// VECTOR NORMALIZATION
// =============================================================================

/// Compute L2 norm
score_t l2_norm(const scalar_t* v, dim_t dim);

/// Normalize vector in-place
void normalize_inplace(scalar_t* v, dim_t dim);

/// Normalize vector (copy)
void normalize(const scalar_t* src, scalar_t* dst, dim_t dim);

/// Batch normalize
void normalize_batch(
    const scalar_t* src,
    scalar_t* dst,
    dim_t dim,
    size_t n_vectors
);

// =============================================================================
// DISTANCE FUNCTION FACTORY
// =============================================================================

/// Get distance function for metric type
DistanceFunc get_distance_func(MetricType metric);

/// Get batch distance function for metric type
BatchDistanceFunc get_batch_distance_func(MetricType metric);

/// Compute distance (high-level API)
VS_INLINE score_t compute_distance(
    const scalar_t* a,
    const scalar_t* b,
    dim_t dim,
    MetricType metric
) {
    switch (metric) {
        case MetricType::L2:
            return l2_distance(a, b, dim);
        case MetricType::COSINE:
            return cosine_similarity(a, b, dim);
        case MetricType::INNER_PRODUCT:
            return inner_product(a, b, dim);
        default:
            return l2_distance(a, b, dim);
    }
}

// =============================================================================
// PREFETCH UTILITIES
// =============================================================================

/// Prefetch vector data for upcoming distance computation
VS_INLINE void prefetch_vector(const scalar_t* v, dim_t dim) {
    // Prefetch in cache-line sized chunks
    for (size_t i = 0; i < dim * sizeof(scalar_t); i += CACHE_LINE_SIZE) {
        prefetch_read(reinterpret_cast<const char*>(v) + i);
    }
}

} // namespace simd
} // namespace vectorsearch
