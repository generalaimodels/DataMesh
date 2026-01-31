/**
 * @file quantization.cpp
 * @brief Quantization implementations
 */

#include "quantization.hpp"
#include "simd_ops.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <random>


namespace vectorsearch {

// =============================================================================
// SCALAR QUANTIZER
// =============================================================================

ScalarQuantizer::ScalarQuantizer(dim_t dimension, Type type)
    : dimension_(dimension), type_(type), scale_(dimension, 1.0f),
      offset_(dimension, 0.0f) {}

void ScalarQuantizer::train(const scalar_t *vectors, size_t n_vectors) {
  // Find min/max per dimension
  std::vector<scalar_t> min_val(dimension_,
                                std::numeric_limits<scalar_t>::max());
  std::vector<scalar_t> max_val(dimension_,
                                std::numeric_limits<scalar_t>::lowest());

  for (size_t i = 0; i < n_vectors; ++i) {
    const scalar_t *v = vectors + i * dimension_;
    for (dim_t d = 0; d < dimension_; ++d) {
      min_val[d] = std::min(min_val[d], v[d]);
      max_val[d] = std::max(max_val[d], v[d]);
    }
  }

  // Compute scale and offset
  for (dim_t d = 0; d < dimension_; ++d) {
    scalar_t range = max_val[d] - min_val[d];
    if (range < 1e-10f)
      range = 1.0f;

    if (type_ == Type::INT8) {
      scale_[d] = 254.0f / range; // -127 to 127
      offset_[d] = -min_val[d] * scale_[d] - 127.0f;
    } else {
      scale_[d] = 255.0f / range; // 0 to 255
      offset_[d] = -min_val[d] * scale_[d];
    }
  }

  trained_ = true;
}

void ScalarQuantizer::encode(const scalar_t *src, uint8_t *dst) const {
  for (dim_t d = 0; d < dimension_; ++d) {
    float val = src[d] * scale_[d] + offset_[d];
    if (type_ == Type::INT8) {
      val = std::clamp(val, -127.0f, 127.0f);
      dst[d] = static_cast<uint8_t>(static_cast<int8_t>(std::round(val)));
    } else {
      val = std::clamp(val, 0.0f, 255.0f);
      dst[d] = static_cast<uint8_t>(std::round(val));
    }
  }
}

void ScalarQuantizer::decode(const uint8_t *src, scalar_t *dst) const {
  for (dim_t d = 0; d < dimension_; ++d) {
    float val;
    if (type_ == Type::INT8) {
      val = static_cast<float>(static_cast<int8_t>(src[d]));
    } else {
      val = static_cast<float>(src[d]);
    }
    dst[d] = (val - offset_[d]) / scale_[d];
  }
}

void ScalarQuantizer::encode_batch(const scalar_t *src, uint8_t *dst,
                                   size_t n) const {
  for (size_t i = 0; i < n; ++i) {
    encode(src + i * dimension_, dst + i * dimension_);
  }
}

score_t ScalarQuantizer::distance(const uint8_t *a, const uint8_t *b) const {
  // L2 distance in quantized space
  int32_t sum = 0;
  for (dim_t d = 0; d < dimension_; ++d) {
    int32_t diff;
    if (type_ == Type::INT8) {
      diff = static_cast<int8_t>(a[d]) - static_cast<int8_t>(b[d]);
    } else {
      diff = static_cast<int32_t>(a[d]) - static_cast<int32_t>(b[d]);
    }
    sum += diff * diff;
  }
  return std::sqrt(static_cast<float>(sum));
}

// =============================================================================
// PRODUCT QUANTIZER
// =============================================================================

ProductQuantizer::ProductQuantizer(dim_t dimension, size_t M, size_t nbits)
    : dimension_(dimension), M_(M), nbits_(nbits), dsub_(dimension / M) {
  // Allocate codebooks
  size_t K = 1 << nbits;
  codebooks_.resize(M * K * dsub_);
}

const scalar_t *ProductQuantizer::get_centroids(size_t m) const {
  size_t K = 1 << nbits_;
  return codebooks_.data() + m * K * dsub_;
}

scalar_t *ProductQuantizer::get_centroids(size_t m) {
  size_t K = 1 << nbits_;
  return codebooks_.data() + m * K * dsub_;
}

void ProductQuantizer::train_subquantizer(size_t m, const scalar_t *subvectors,
                                          size_t n_vectors, size_t n_iter) {
  size_t K = 1 << nbits_;
  scalar_t *centroids = get_centroids(m);

  // Random initialization
  std::mt19937 rng(42 + m);
  std::uniform_int_distribution<size_t> dist(0, n_vectors - 1);

  for (size_t k = 0; k < K; ++k) {
    size_t idx = dist(rng);
    std::memcpy(centroids + k * dsub_, subvectors + idx * dsub_,
                dsub_ * sizeof(scalar_t));
  }

  // K-means iterations
  std::vector<size_t> assignments(n_vectors);
  std::vector<scalar_t> new_centroids(K * dsub_);
  std::vector<size_t> counts(K);

  for (size_t iter = 0; iter < n_iter; ++iter) {
    // Assignment step
    for (size_t i = 0; i < n_vectors; ++i) {
      const scalar_t *sv = subvectors + i * dsub_;

      score_t best_dist = std::numeric_limits<score_t>::max();
      size_t best_k = 0;

      for (size_t k = 0; k < K; ++k) {
        score_t d = simd::l2_distance(sv, centroids + k * dsub_, dsub_);
        if (d < best_dist) {
          best_dist = d;
          best_k = k;
        }
      }

      assignments[i] = best_k;
    }

    // Update step
    std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t i = 0; i < n_vectors; ++i) {
      size_t k = assignments[i];
      const scalar_t *sv = subvectors + i * dsub_;
      for (size_t d = 0; d < dsub_; ++d) {
        new_centroids[k * dsub_ + d] += sv[d];
      }
      counts[k]++;
    }

    for (size_t k = 0; k < K; ++k) {
      if (counts[k] > 0) {
        for (size_t d = 0; d < dsub_; ++d) {
          centroids[k * dsub_ + d] = new_centroids[k * dsub_ + d] / counts[k];
        }
      }
    }
  }
}

void ProductQuantizer::train(const scalar_t *vectors, size_t n_vectors,
                             size_t n_iter) {
  // Extract and train each subquantizer
  std::vector<scalar_t> subvectors(n_vectors * dsub_);

  for (size_t m = 0; m < M_; ++m) {
    // Extract subvectors for this quantizer
    for (size_t i = 0; i < n_vectors; ++i) {
      const scalar_t *src = vectors + i * dimension_ + m * dsub_;
      scalar_t *dst = subvectors.data() + i * dsub_;
      std::memcpy(dst, src, dsub_ * sizeof(scalar_t));
    }

    train_subquantizer(m, subvectors.data(), n_vectors, n_iter);
  }

  trained_ = true;
}

void ProductQuantizer::encode(const scalar_t *src, uint8_t *dst) const {
  size_t K = 1 << nbits_;

  for (size_t m = 0; m < M_; ++m) {
    const scalar_t *subvec = src + m * dsub_;
    const scalar_t *centroids = get_centroids(m);

    // Find nearest centroid
    score_t best_dist = std::numeric_limits<score_t>::max();
    uint8_t best_k = 0;

    for (size_t k = 0; k < K; ++k) {
      score_t d = simd::l2_distance(subvec, centroids + k * dsub_, dsub_);
      if (d < best_dist) {
        best_dist = d;
        best_k = static_cast<uint8_t>(k);
      }
    }

    dst[m] = best_k;
  }
}

void ProductQuantizer::decode(const uint8_t *src, scalar_t *dst) const {
  for (size_t m = 0; m < M_; ++m) {
    const scalar_t *centroid = get_centroids(m) + src[m] * dsub_;
    std::memcpy(dst + m * dsub_, centroid, dsub_ * sizeof(scalar_t));
  }
}

void ProductQuantizer::compute_distance_table(const scalar_t *query,
                                              float *table) const {
  size_t K = 1 << nbits_;

  for (size_t m = 0; m < M_; ++m) {
    const scalar_t *subquery = query + m * dsub_;
    const scalar_t *centroids = get_centroids(m);

    for (size_t k = 0; k < K; ++k) {
      // Store squared distance for efficiency
      score_t d = simd::l2_distance(subquery, centroids + k * dsub_, dsub_);
      table[m * K + k] = d * d;
    }
  }
}

score_t ProductQuantizer::distance_with_table(const uint8_t *code,
                                              const float *table) const {
  size_t K = 1 << nbits_;
  float sum = 0.0f;

  for (size_t m = 0; m < M_; ++m) {
    sum += table[m * K + code[m]];
  }

  return std::sqrt(sum);
}

score_t ProductQuantizer::distance(const scalar_t *query,
                                   const uint8_t *code) const {
  float sum = 0.0f;

  for (size_t m = 0; m < M_; ++m) {
    const scalar_t *subquery = query + m * dsub_;
    const scalar_t *centroid = get_centroids(m) + code[m] * dsub_;
    score_t d = simd::l2_distance(subquery, centroid, dsub_);
    sum += d * d;
  }

  return std::sqrt(sum);
}

// =============================================================================
// OPTIMIZED PRODUCT QUANTIZER
// =============================================================================

OptimizedProductQuantizer::OptimizedProductQuantizer(dim_t dimension, size_t M,
                                                     size_t nbits)
    : pq_(dimension, M, nbits), rotation_(dimension * dimension, 0.0f) {
  // Initialize with identity rotation
  for (dim_t i = 0; i < dimension; ++i) {
    rotation_[i * dimension + i] = 1.0f;
  }
}

void OptimizedProductQuantizer::apply_rotation(const scalar_t *src,
                                               scalar_t *dst) const {
  dim_t D = pq_.code_size() *
            (pq_.K() > 0 ? pq_.table_size() / pq_.K() / pq_.code_size() : 96);
  // Note: simplified - full impl would use proper dimension
  std::memcpy(dst, src, D * sizeof(scalar_t)); // Placeholder
}

void OptimizedProductQuantizer::apply_inverse_rotation(const scalar_t *src,
                                                       scalar_t *dst) const {
  // For orthogonal matrices, inverse = transpose
  apply_rotation(src, dst); // Placeholder
}

void OptimizedProductQuantizer::train(const scalar_t *vectors, size_t n_vectors,
                                      size_t n_iter) {
  // TODO: Implement rotation learning using alternating optimization
  // For now, just train PQ without rotation
  pq_.train(vectors, n_vectors, 25);
  trained_ = true;
}

void OptimizedProductQuantizer::encode(const scalar_t *src,
                                       uint8_t *dst) const {
  // Apply rotation, then PQ encode
  aligned_vector<scalar_t> rotated(pq_.table_size() / pq_.K());
  apply_rotation(src, rotated.data());
  pq_.encode(rotated.data(), dst);
}

void OptimizedProductQuantizer::decode(const uint8_t *src,
                                       scalar_t *dst) const {
  aligned_vector<scalar_t> decoded(pq_.table_size() / pq_.K());
  pq_.decode(src, decoded.data());
  apply_inverse_rotation(decoded.data(), dst);
}

void OptimizedProductQuantizer::compute_distance_table(const scalar_t *query,
                                                       float *table) const {
  aligned_vector<scalar_t> rotated(pq_.table_size() / pq_.K());
  apply_rotation(query, rotated.data());
  pq_.compute_distance_table(rotated.data(), table);
}

score_t
OptimizedProductQuantizer::distance_with_table(const uint8_t *code,
                                               const float *table) const {
  return pq_.distance_with_table(code, table);
}

} // namespace vectorsearch
