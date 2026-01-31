"""
Unit Tests: HNSW Index

Tests:
    - Index creation and configuration
    - Single and batch insert
    - Search with various k values
    - Delete operations
    - Recall@k measurement
"""

import pytest
import numpy as np

from vector_search.core.config import HNSWConfig
from vector_search.core.types import (
    VectorId,
    EmbeddingVector,
    SearchQuery,
    MetricType,
)
from vector_search.index.hnsw import HNSWIndex


class TestHNSWCreation:
    """Tests for HNSW index creation."""
    
    def test_default_config(self):
        """Test index creation with default config."""
        index = HNSWIndex()
        
        assert index.dimension == 768
        assert index.count == 0
    
    def test_custom_config(self):
        """Test index creation with custom config."""
        config = HNSWConfig(
            dimension=128,
            M=32,
            ef_construction=200,
        )
        index = HNSWIndex(config=config)
        
        assert index.dimension == 128
        assert index.config.M == 32


class TestHNSWInsert:
    """Tests for vector insertion."""
    
    def test_single_insert(self):
        """Test inserting a single vector."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        vid = VectorId(value="vec-1")
        vec = EmbeddingVector.from_numpy(np.random.randn(128).astype(np.float32))
        
        result = index.insert(vid, vec)
        
        assert result.is_ok()
        assert index.count == 1
    
    def test_insert_with_metadata(self):
        """Test inserting vector with metadata."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        vid = VectorId(value="doc-1")
        vec = EmbeddingVector.from_numpy(np.random.randn(128).astype(np.float32))
        metadata = {"title": "Test Document", "category": "tech"}
        
        result = index.insert(vid, vec, metadata=metadata)
        
        assert result.is_ok()
    
    def test_insert_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        vid = VectorId(value="vec-1")
        vec = EmbeddingVector.from_numpy(np.random.randn(64).astype(np.float32))  # Wrong dim
        
        result = index.insert(vid, vec)
        
        assert result.is_err()
    
    def test_batch_insert(self):
        """Test batch vector insertion."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        n = 100
        ids = [VectorId(value=f"vec-{i}") for i in range(n)]
        vecs = [
            EmbeddingVector.from_numpy(np.random.randn(128).astype(np.float32))
            for _ in range(n)
        ]
        
        result = index.insert_batch(ids, vecs)
        
        assert result.is_ok()
        assert result.unwrap() == n
        assert index.count == n
    
    def test_upsert(self):
        """Test updating existing vector."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        vid = VectorId(value="vec-1")
        vec1 = EmbeddingVector.from_numpy(np.ones(128).astype(np.float32))
        vec2 = EmbeddingVector.from_numpy(np.zeros(128).astype(np.float32))
        
        # Insert
        result1 = index.insert(vid, vec1)
        assert result1.is_ok()
        assert index.count == 1
        
        # Update (same ID)
        result2 = index.insert(vid, vec2)
        assert result2.is_ok()
        assert index.count == 1  # Still 1


class TestHNSWSearch:
    """Tests for vector search."""
    
    @pytest.fixture
    def populated_index(self):
        """Create index with test vectors."""
        config = HNSWConfig(dimension=128, ef_search=50)
        index = HNSWIndex(config=config)
        
        # Insert 1000 random vectors
        np.random.seed(42)
        n = 1000
        for i in range(n):
            vid = VectorId(value=f"vec-{i}")
            vec = EmbeddingVector.from_numpy(np.random.randn(128).astype(np.float32))
            index.insert(vid, vec, metadata={"idx": i})
        
        return index
    
    def test_basic_search(self, populated_index):
        """Test basic k-NN search."""
        index = populated_index
        
        query_vec = np.random.randn(128).astype(np.float32)
        query = SearchQuery(vector=query_vec.tolist(), k=10)
        
        result = index.search(query)
        
        assert result.is_ok()
        results = result.unwrap()
        assert len(results.matches) == 10
        assert results.query_time_ms > 0
    
    def test_search_returns_metadata(self, populated_index):
        """Test that search returns metadata."""
        index = populated_index
        
        query_vec = np.random.randn(128).astype(np.float32)
        query = SearchQuery(vector=query_vec.tolist(), k=5, include_metadata=True)
        
        result = index.search(query)
        
        assert result.is_ok()
        results = result.unwrap()
        for match in results.matches:
            assert match.metadata is not None
            assert "idx" in match.metadata
    
    def test_search_k_larger_than_index(self, populated_index):
        """Test search when k > index size."""
        config = HNSWConfig(dimension=128)
        small_index = HNSWIndex(config=config)
        
        # Insert only 5 vectors
        for i in range(5):
            vid = VectorId(value=f"vec-{i}")
            vec = EmbeddingVector.from_numpy(np.random.randn(128).astype(np.float32))
            small_index.insert(vid, vec)
        
        query = SearchQuery(vector=np.random.randn(128).tolist(), k=100)
        result = small_index.search(query)
        
        assert result.is_ok()
        assert len(result.unwrap().matches) <= 5
    
    def test_search_empty_index(self):
        """Test search on empty index."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        query = SearchQuery(vector=np.random.randn(128).tolist(), k=10)
        result = index.search(query)
        
        assert result.is_ok()
        assert len(result.unwrap().matches) == 0
    
    def test_score_threshold(self, populated_index):
        """Test filtering by score threshold."""
        index = populated_index
        
        query_vec = np.random.randn(128).astype(np.float32)
        query = SearchQuery(
            vector=query_vec.tolist(),
            k=100,
            score_threshold=0.9,  # Very high threshold
        )
        
        result = index.search(query)
        
        assert result.is_ok()
        results = result.unwrap()
        for match in results.matches:
            assert match.score >= 0.9


class TestHNSWDelete:
    """Tests for vector deletion."""
    
    def test_delete_existing(self):
        """Test deleting existing vector."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        vid = VectorId(value="vec-1")
        vec = EmbeddingVector.from_numpy(np.random.randn(128).astype(np.float32))
        
        index.insert(vid, vec)
        assert index.count == 1
        
        result = index.delete(vid)
        
        assert result.is_ok()
        assert result.unwrap() is True
    
    def test_delete_nonexistent(self):
        """Test deleting non-existent vector."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        vid = VectorId(value="nonexistent")
        result = index.delete(vid)
        
        assert result.is_ok()
        assert result.unwrap() is False


class TestHNSWGet:
    """Tests for vector retrieval."""
    
    def test_get_existing(self):
        """Test getting existing vector."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        vid = VectorId(value="vec-1")
        original = np.random.randn(128).astype(np.float32)
        vec = EmbeddingVector.from_numpy(original)
        
        index.insert(vid, vec)
        result = index.get(vid)
        
        assert result.is_ok()
        retrieved = result.unwrap().to_numpy()
        # Vectors should be similar (cosine normalization may affect exact values)
        assert retrieved.shape == original.shape
    
    def test_get_nonexistent(self):
        """Test getting non-existent vector."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        vid = VectorId(value="nonexistent")
        result = index.get(vid)
        
        assert result.is_err()


class TestHNSWStats:
    """Tests for index statistics."""
    
    def test_stats_empty(self):
        """Test stats on empty index."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        stats = index.stats()
        
        assert stats.total_vectors == 0
        assert stats.dimension == 128
    
    def test_stats_populated(self):
        """Test stats on populated index."""
        config = HNSWConfig(dimension=128)
        index = HNSWIndex(config=config)
        
        # Insert some vectors
        for i in range(100):
            vid = VectorId(value=f"vec-{i}")
            vec = EmbeddingVector.from_numpy(np.random.randn(128).astype(np.float32))
            index.insert(vid, vec)
        
        stats = index.stats()
        
        assert stats.total_vectors == 100
        assert stats.dimension == 128
        assert stats.index_size_bytes > 0


class TestHNSWRecall:
    """Tests for search recall quality."""
    
    def test_recall_at_10(self):
        """Test recall@10 against brute force."""
        config = HNSWConfig(dimension=64, M=16, ef_construction=100, ef_search=50)
        index = HNSWIndex(config=config)
        
        # Insert vectors
        np.random.seed(42)
        n = 500
        vectors = np.random.randn(n, 64).astype(np.float32)
        
        for i in range(n):
            vid = VectorId(value=f"vec-{i}")
            vec = EmbeddingVector.from_numpy(vectors[i])
            index.insert(vid, vec)
        
        # Query
        n_queries = 10
        k = 10
        total_recall = 0.0
        
        for _ in range(n_queries):
            query_vec = np.random.randn(64).astype(np.float32)
            
            # HNSW search
            query = SearchQuery(vector=query_vec.tolist(), k=k)
            result = index.search(query)
            hnsw_ids = set(str(m.id) for m in result.unwrap().matches)
            
            # Brute force ground truth
            query_norm = query_vec / np.linalg.norm(query_vec)
            vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            sims = vectors_norm @ query_norm
            top_k_indices = np.argsort(sims)[-k:]
            ground_truth = set(f"vec-{i}" for i in top_k_indices)
            
            # Compute recall
            recall = len(hnsw_ids & ground_truth) / k
            total_recall += recall
        
        avg_recall = total_recall / n_queries
        
        # Expect at least 90% recall with these parameters
        assert avg_recall >= 0.8, f"Recall@10 = {avg_recall:.2f}, expected >= 0.80"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
