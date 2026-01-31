/**
 * @file quantization.hpp
 * @brief Vector quantization for memory efficiency
 *
 * Implements:
 *   - Scalar Quantization (INT8, UINT8)
 *   - Product Quantization (PQ)
 *   - Optimized Product Quantization (OPQ)
 */

#pragma once

#include "common.hpp"
#include <memory>
#include <vector>


namespace vectorsearch {

// =============================================================================
// SCALAR QUANTIZATION
// =============================================================================

/**
 * @brief INT8/UINT8 scalar quantization
 *
 * Compresses 32-bit floats to 8-bit integers:
 *   - 4x memory reduction
 *   - ~1-3% recall loss at same parameters
 */
class ScalarQuantizer {
public:
  enum class Type { INT8, UINT8 };

  ScalarQuantizer(dim_t dimension, Type type = Type::INT8);

  /// Train quantizer on vectors
  void train(const scalar_t *vectors, size_t n_vectors);

  /// Quantize single vector
  void encode(const scalar_t *src, uint8_t *dst) const;

  /// Decode single vector
  void decode(const uint8_t *src, scalar_t *dst) const;

  /// Batch quantize
  void encode_batch(const scalar_t *src, uint8_t *dst, size_t n) const;

  /// Compute distance in quantized space
  score_t distance(const uint8_t *a, const uint8_t *b) const;

  /// Memory per vector
  size_t code_size() const { return dimension_; }

private:
  dim_t dimension_;
  Type type_;
  std::vector<scalar_t> scale_;
  std::vector<scalar_t> offset_;
  bool trained_ = false;
};

// =============================================================================
// PRODUCT QUANTIZATION
// =============================================================================

/**
 * @brief Product Quantization
 *
 * Splits vector into M subvectors, each quantized to log2(K) bits:
 *   - M=8, K=256: 8 bytes per 768-dim vector (96x compression)
 *   - Supports asymmetric distance (ADC)
 */
class ProductQuantizer {
public:
  /**
   * @param dimension Vector dimension
   * @param M Number of subquantizers (must divide dimension)
   * @param nbits Bits per subquantizer (default 8 = 256 centroids)
   */
  ProductQuantizer(dim_t dimension, size_t M = 8, size_t nbits = 8);

  /// Train codebooks on vectors
  void train(const scalar_t *vectors, size_t n_vectors, size_t n_iter = 25);

  /// Encode vector to PQ code
  void encode(const scalar_t *src, uint8_t *dst) const;

  /// Decode PQ code to vector
  void decode(const uint8_t *src, scalar_t *dst) const;

  /// Precompute distance table for query (for ADC)
  void compute_distance_table(const scalar_t *query, float *table) const;

  /// Compute distance using precomputed table
  score_t distance_with_table(const uint8_t *code, const float *table) const;

  /// Direct distance (slower, no precompute)
  score_t distance(const scalar_t *query, const uint8_t *code) const;

  /// Memory per vector
  size_t code_size() const { return M_; }

  /// Number of centroids per subquantizer
  size_t K() const { return 1 << nbits_; }

  /// Distance table size
  size_t table_size() const { return M_ * K(); }

private:
  dim_t dimension_;
  size_t M_;     // Number of subquantizers
  size_t nbits_; // Bits per subquantizer
  size_t dsub_;  // Dimension per subquantizer

  // Codebooks: M x K x dsub
  std::vector<scalar_t> codebooks_;
  bool trained_ = false;

  /// Get codebook for subquantizer m
  const scalar_t *get_centroids(size_t m) const;
  scalar_t *get_centroids(size_t m);

  /// K-means for subquantizer training
  void train_subquantizer(size_t m, const scalar_t *subvectors,
                          size_t n_vectors, size_t n_iter);
};

// =============================================================================
// OPTIMIZED PRODUCT QUANTIZATION
// =============================================================================

/**
 * @brief OPQ: Rotation-optimized Product Quantization
 *
 * Learns rotation matrix R to minimize quantization error:
 *   - Typically 5-15% better recall than standard PQ
 *   - Same memory footprint
 */
class OptimizedProductQuantizer {
public:
  OptimizedProductQuantizer(dim_t dimension, size_t M = 8, size_t nbits = 8);

  /// Train rotation and codebooks
  void train(const scalar_t *vectors, size_t n_vectors, size_t n_iter = 10);

  /// Encode with rotation
  void encode(const scalar_t *src, uint8_t *dst) const;

  /// Decode with inverse rotation
  void decode(const uint8_t *src, scalar_t *dst) const;

  /// Precompute distance table (query is rotated)
  void compute_distance_table(const scalar_t *query, float *table) const;

  score_t distance_with_table(const uint8_t *code, const float *table) const;

  size_t code_size() const { return pq_.code_size(); }
  size_t table_size() const { return pq_.table_size(); }

private:
  ProductQuantizer pq_;
  std::vector<scalar_t> rotation_; // D x D rotation matrix
  bool trained_ = false;

  void apply_rotation(const scalar_t *src, scalar_t *dst) const;
  void apply_inverse_rotation(const scalar_t *src, scalar_t *dst) const;
};

} // namespace vectorsearch
