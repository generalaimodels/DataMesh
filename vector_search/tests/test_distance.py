"""
Unit Tests: Distance Functions

Tests:
    - Cosine similarity (single and batch)
    - L2 distance (single and batch)
    - Inner product (single and batch)
    - Vector normalization
"""

import pytest
import numpy as np

from vector_search.index.distance import (
    cosine_similarity,
    cosine_similarity_batch,
    cosine_distance,
    l2_distance,
    l2_distance_batch,
    l2_distance_squared,
    inner_product,
    inner_product_batch,
    normalize_vector,
)


class TestNormalization:
    """Tests for vector normalization."""
    
    def test_normalize_single(self):
        """Test single vector normalization."""
        v = np.array([3.0, 4.0])
        normalized = normalize_vector(v)
        
        np.testing.assert_allclose(normalized, [0.6, 0.8], rtol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(normalized), 1.0, rtol=1e-5)
    
    def test_normalize_batch(self):
        """Test batch normalization."""
        v = np.array([
            [3.0, 4.0],
            [1.0, 0.0],
        ])
        normalized = normalize_vector(v)
        
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)
    
    def test_normalize_zero(self):
        """Test normalization of zero vector."""
        v = np.array([0.0, 0.0])
        normalized = normalize_vector(v)
        
        # Should not raise, returns zero vector
        np.testing.assert_allclose(normalized, [0.0, 0.0])


class TestCosineSimilarity:
    """Tests for cosine similarity."""
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(v, v)
        
        np.testing.assert_allclose(sim, 1.0, rtol=1e-5)
    
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        sim = cosine_similarity(v1, v2)
        
        np.testing.assert_allclose(sim, 0.0, rtol=1e-5)
    
    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        v1 = np.array([1.0, 1.0])
        v2 = np.array([-1.0, -1.0])
        sim = cosine_similarity(v1, v2)
        
        np.testing.assert_allclose(sim, -1.0, rtol=1e-5)
    
    def test_batch_similarity(self):
        """Test batch cosine similarity."""
        query = np.array([1.0, 0.0])
        candidates = np.array([
            [1.0, 0.0],   # Identical
            [0.0, 1.0],   # Orthogonal
            [0.707, 0.707],  # 45 degrees
        ])
        
        sims = cosine_similarity_batch(query, candidates)
        
        np.testing.assert_allclose(sims[0], 1.0, rtol=1e-3)
        np.testing.assert_allclose(sims[1], 0.0, rtol=1e-3)
        np.testing.assert_allclose(sims[2], 0.707, rtol=1e-2)
    
    def test_cosine_distance(self):
        """Test cosine distance."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        
        dist = cosine_distance(v1, v2)
        np.testing.assert_allclose(dist, 1.0, rtol=1e-5)


class TestL2Distance:
    """Tests for L2 (Euclidean) distance."""
    
    def test_identical_vectors(self):
        """Test distance of identical vectors."""
        v = np.array([1.0, 2.0, 3.0])
        dist = l2_distance(v, v)
        
        np.testing.assert_allclose(dist, 0.0, rtol=1e-5)
    
    def test_unit_distance(self):
        """Test unit distance."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        dist = l2_distance(v1, v2)
        
        np.testing.assert_allclose(dist, 1.0, rtol=1e-5)
    
    def test_pythagorean(self):
        """Test Pythagorean distance (3-4-5 triangle)."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        dist = l2_distance(v1, v2)
        
        np.testing.assert_allclose(dist, 5.0, rtol=1e-5)
    
    def test_squared_distance(self):
        """Test squared L2 distance."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        dist_sq = l2_distance_squared(v1, v2)
        
        np.testing.assert_allclose(dist_sq, 25.0, rtol=1e-5)
    
    def test_batch_distance(self):
        """Test batch L2 distance."""
        query = np.array([0.0, 0.0])
        candidates = np.array([
            [1.0, 0.0],
            [3.0, 4.0],
            [0.0, 0.0],
        ])
        
        dists = l2_distance_batch(query, candidates)
        
        np.testing.assert_allclose(dists[0], 1.0, rtol=1e-5)
        np.testing.assert_allclose(dists[1], 5.0, rtol=1e-5)
        np.testing.assert_allclose(dists[2], 0.0, rtol=1e-5)


class TestInnerProduct:
    """Tests for inner product."""
    
    def test_basic_dot(self):
        """Test basic dot product."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        
        ip = inner_product(v1, v2)
        expected = 1*4 + 2*5 + 3*6  # 32
        
        np.testing.assert_allclose(ip, expected, rtol=1e-5)
    
    def test_orthogonal(self):
        """Test orthogonal vectors."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        
        ip = inner_product(v1, v2)
        np.testing.assert_allclose(ip, 0.0, rtol=1e-5)
    
    def test_batch_inner_product(self):
        """Test batch inner product."""
        query = np.array([1.0, 2.0])
        candidates = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        
        ips = inner_product_batch(query, candidates)
        
        np.testing.assert_allclose(ips[0], 1.0, rtol=1e-5)
        np.testing.assert_allclose(ips[1], 2.0, rtol=1e-5)
        np.testing.assert_allclose(ips[2], 3.0, rtol=1e-5)


class TestPerformance:
    """Performance tests for distance functions."""
    
    def test_high_dimension(self):
        """Test with high-dimensional vectors."""
        dim = 1536
        v1 = np.random.randn(dim).astype(np.float32)
        v2 = np.random.randn(dim).astype(np.float32)
        
        # Should complete without error
        sim = cosine_similarity(v1, v2)
        dist = l2_distance(v1, v2)
        ip = inner_product(v1, v2)
        
        assert -1.0 <= sim <= 1.0
        assert dist >= 0
    
    def test_large_batch(self):
        """Test with large batch."""
        dim = 768
        n_candidates = 1000
        
        query = np.random.randn(dim).astype(np.float32)
        candidates = np.random.randn(n_candidates, dim).astype(np.float32)
        
        # Should complete without error
        sims = cosine_similarity_batch(query, candidates)
        dists = l2_distance_batch(query, candidates)
        ips = inner_product_batch(query, candidates)
        
        assert len(sims) == n_candidates
        assert len(dists) == n_candidates
        assert len(ips) == n_candidates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
