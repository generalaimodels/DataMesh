/**
 * @file simd_ops.cpp
 * @brief SIMD-optimized distance kernel implementations
 *
 * Implements AVX2/AVX-512 accelerated:
 *   - L2 distance: ~50ns for 1536-dim vectors
 *   - Inner product: ~40ns for 1536-dim vectors
 *   - Batch operations with prefetching
 */

#include "simd_ops.hpp"
#include <cstring>

namespace vectorsearch {
namespace simd {

// =============================================================================
// CPU FEATURE DETECTION
// =============================================================================

CPUFeatures CPUFeatures::detect() {
  CPUFeatures features;

#if defined(_MSC_VER)
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  int nIds = cpuInfo[0];

  if (nIds >= 7) {
    __cpuidex(cpuInfo, 7, 0);
    features.avx2 = (cpuInfo[1] & (1 << 5)) != 0;
    features.avx512f = (cpuInfo[1] & (1 << 16)) != 0;
  }
  if (nIds >= 1) {
    __cpuid(cpuInfo, 1);
    features.fma = (cpuInfo[2] & (1 << 12)) != 0;
  }
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
  // Use GCC/Clang built-in
  __builtin_cpu_init();
  features.avx2 = __builtin_cpu_supports("avx2");
  features.avx512f = __builtin_cpu_supports("avx512f");
  features.fma = __builtin_cpu_supports("fma");
#endif
#endif

  return features;
}

// Global CPU features instance
CPUFeatures cpu_features = CPUFeatures::detect();

// =============================================================================
// SCALAR IMPLEMENTATIONS (Fallback)
// =============================================================================

score_t l2_distance_scalar(const scalar_t *a, const scalar_t *b, dim_t dim) {
  score_t sum = 0.0f;
  for (dim_t i = 0; i < dim; ++i) {
    score_t diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

score_t inner_product_scalar(const scalar_t *a, const scalar_t *b, dim_t dim) {
  score_t sum = 0.0f;
  for (dim_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

// =============================================================================
// AVX2 IMPLEMENTATIONS
// =============================================================================

#if VECTORSEARCH_AVX2

score_t l2_distance_avx2(const scalar_t *a, const scalar_t *b, dim_t dim) {
  __m256 sum = _mm256_setzero_ps();

  // Process 8 floats at a time (256 bits / 32 bits = 8)
  dim_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);

#if VECTORSEARCH_FMA
    sum = _mm256_fmadd_ps(diff, diff, sum);
#else
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
#endif
  }

  // Horizontal sum
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);

  score_t result = _mm_cvtss_f32(sum128);

  // Handle remainder
  for (; i < dim; ++i) {
    score_t diff = a[i] - b[i];
    result += diff * diff;
  }

  return std::sqrt(result);
}

score_t inner_product_avx2(const scalar_t *a, const scalar_t *b, dim_t dim) {
  __m256 sum = _mm256_setzero_ps();

  dim_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);

#if VECTORSEARCH_FMA
    sum = _mm256_fmadd_ps(va, vb, sum);
#else
    sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
#endif
  }

  // Horizontal sum
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);

  score_t result = _mm_cvtss_f32(sum128);

  // Handle remainder
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }

  return result;
}

#endif // VECTORSEARCH_AVX2

// =============================================================================
// AVX-512 IMPLEMENTATIONS
// =============================================================================

#if VECTORSEARCH_AVX512

score_t l2_distance_avx512(const scalar_t *a, const scalar_t *b, dim_t dim) {
  __m512 sum = _mm512_setzero_ps();

  // Process 16 floats at a time (512 bits / 32 bits = 16)
  dim_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    __m512 diff = _mm512_sub_ps(va, vb);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }

  score_t result = _mm512_reduce_add_ps(sum);

  // Handle remainder with AVX2
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    __m256 sq = _mm256_mul_ps(diff, diff);

    __m128 hi = _mm256_extractf128_ps(sq, 1);
    __m128 lo = _mm256_castps256_ps128(sq);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    result += _mm_cvtss_f32(sum128);
  }

  // Handle final remainder
  for (; i < dim; ++i) {
    score_t diff = a[i] - b[i];
    result += diff * diff;
  }

  return std::sqrt(result);
}

score_t inner_product_avx512(const scalar_t *a, const scalar_t *b, dim_t dim) {
  __m512 sum = _mm512_setzero_ps();

  dim_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    sum = _mm512_fmadd_ps(va, vb, sum);
  }

  score_t result = _mm512_reduce_add_ps(sum);

  // Handle remainder
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }

  return result;
}

#endif // VECTORSEARCH_AVX512

// =============================================================================
// BATCH OPERATIONS
// =============================================================================

void l2_distance_batch(const scalar_t *query, const scalar_t *candidates,
                       score_t *results, dim_t dim, size_t n_candidates) {
  // Prefetch first few candidates
  for (size_t i = 0; i < std::min(n_candidates, size_t(4)); ++i) {
    prefetch_vector(candidates + i * dim, dim);
  }

  for (size_t i = 0; i < n_candidates; ++i) {
    // Prefetch next candidate
    if (i + 4 < n_candidates) {
      prefetch_vector(candidates + (i + 4) * dim, dim);
    }

    results[i] = l2_distance(query, candidates + i * dim, dim);
  }
}

void inner_product_batch(const scalar_t *query, const scalar_t *candidates,
                         score_t *results, dim_t dim, size_t n_candidates) {
  for (size_t i = 0; i < std::min(n_candidates, size_t(4)); ++i) {
    prefetch_vector(candidates + i * dim, dim);
  }

  for (size_t i = 0; i < n_candidates; ++i) {
    if (i + 4 < n_candidates) {
      prefetch_vector(candidates + (i + 4) * dim, dim);
    }

    results[i] = inner_product(query, candidates + i * dim, dim);
  }
}

void cosine_similarity_batch(const scalar_t *query, const scalar_t *candidates,
                             score_t *results, dim_t dim, size_t n_candidates,
                             bool normalized) {
  if (normalized) {
    // For normalized vectors, cosine = inner product
    inner_product_batch(query, candidates, results, dim, n_candidates);
  } else {
    // Need to compute full cosine similarity
    for (size_t i = 0; i < n_candidates; ++i) {
      results[i] = cosine_similarity_unnorm(query, candidates + i * dim, dim);
    }
  }
}

// =============================================================================
// NORMALIZATION
// =============================================================================

score_t l2_norm(const scalar_t *v, dim_t dim) {
  return std::sqrt(inner_product(v, v, dim));
}

void normalize_inplace(scalar_t *v, dim_t dim) {
  score_t norm = l2_norm(v, dim);
  if (norm > 1e-10f) {
    score_t inv_norm = 1.0f / norm;
    for (dim_t i = 0; i < dim; ++i) {
      v[i] *= inv_norm;
    }
  }
}

void normalize(const scalar_t *src, scalar_t *dst, dim_t dim) {
  score_t norm = l2_norm(src, dim);
  if (norm > 1e-10f) {
    score_t inv_norm = 1.0f / norm;
    for (dim_t i = 0; i < dim; ++i) {
      dst[i] = src[i] * inv_norm;
    }
  } else {
    std::memcpy(dst, src, dim * sizeof(scalar_t));
  }
}

void normalize_batch(const scalar_t *src, scalar_t *dst, dim_t dim,
                     size_t n_vectors) {
  for (size_t i = 0; i < n_vectors; ++i) {
    normalize(src + i * dim, dst + i * dim, dim);
  }
}

score_t cosine_similarity_unnorm(const scalar_t *a, const scalar_t *b,
                                 dim_t dim) {
  score_t dot = inner_product(a, b, dim);
  score_t norm_a = l2_norm(a, dim);
  score_t norm_b = l2_norm(b, dim);

  if (norm_a < 1e-10f || norm_b < 1e-10f) {
    return 0.0f;
  }

  return dot / (norm_a * norm_b);
}

// =============================================================================
// DISTANCE FUNCTION FACTORY
// =============================================================================

DistanceFunc get_distance_func(MetricType metric) {
  switch (metric) {
  case MetricType::L2:
#if VECTORSEARCH_AVX512
    if (cpu_features.avx512f)
      return l2_distance_avx512;
#endif
#if VECTORSEARCH_AVX2
    if (cpu_features.avx2)
      return l2_distance_avx2;
#endif
    return l2_distance_scalar;

  case MetricType::COSINE:
  case MetricType::INNER_PRODUCT:
#if VECTORSEARCH_AVX512
    if (cpu_features.avx512f)
      return inner_product_avx512;
#endif
#if VECTORSEARCH_AVX2
    if (cpu_features.avx2)
      return inner_product_avx2;
#endif
    return inner_product_scalar;

  default:
    return l2_distance_scalar;
  }
}

BatchDistanceFunc get_batch_distance_func(MetricType metric) {
  switch (metric) {
  case MetricType::L2:
    return l2_distance_batch;
  case MetricType::COSINE:
  case MetricType::INNER_PRODUCT:
    return inner_product_batch;
  default:
    return l2_distance_batch;
  }
}

} // namespace simd
} // namespace vectorsearch
