"""
Unit Tests: Core Types

Tests:
    - VectorId creation and serialization
    - EmbeddingVector from various sources
    - SearchQuery validation
    - SearchResult construction
"""

import pytest
import numpy as np

from vector_search.core.types import (
    VectorId,
    EmbeddingVector,
    SearchQuery,
    SearchResult,
    SearchResults,
    MetricType,
    IndexConfig,
)


class TestVectorId:
    """Tests for VectorId."""
    
    def test_create_default(self):
        """Test basic VectorId creation."""
        vid = VectorId(value="test-123")
        assert vid.value == "test-123"
        assert vid.namespace == "default"
    
    def test_create_with_namespace(self):
        """Test VectorId with custom namespace."""
        vid = VectorId(value="doc-1", namespace="documents")
        assert vid.value == "doc-1"
        assert vid.namespace == "documents"
    
    def test_generate(self):
        """Test random VectorId generation."""
        vid = VectorId.generate()
        assert len(vid.value) == 36  # UUID format
        assert vid.namespace == "default"
    
    def test_composite(self):
        """Test composite format (namespace:id)."""
        vid = VectorId(value="doc-1", namespace="docs")
        assert vid.to_composite() == "docs:doc-1"
        
        # Parse back
        parsed = VectorId.from_composite("docs:doc-1")
        assert parsed.value == "doc-1"
        assert parsed.namespace == "docs"
    
    def test_shard_key(self):
        """Test shard key generation."""
        vid = VectorId(value="test")
        key = vid.shard_key()
        assert len(key) == 20  # SHA1 digest
        assert isinstance(key, bytes)
    
    def test_hash_equality(self):
        """Test VectorId hashability."""
        vid1 = VectorId(value="test", namespace="ns")
        vid2 = VectorId(value="test", namespace="ns")
        
        assert vid1 == vid2
        assert hash(vid1) == hash(vid2)


class TestEmbeddingVector:
    """Tests for EmbeddingVector."""
    
    def test_from_list(self):
        """Test creation from Python list."""
        values = [0.1, 0.2, 0.3, 0.4]
        vec = EmbeddingVector.from_list(values)
        
        assert vec.dimension == 4
        assert vec.dtype == "float32"
        assert len(vec) == 4
    
    def test_from_numpy(self):
        """Test creation from numpy array."""
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vec = EmbeddingVector.from_numpy(arr)
        
        assert vec.dimension == 3
        assert vec.dtype == "float32"
    
    def test_to_numpy(self):
        """Test conversion to numpy."""
        values = [0.1, 0.2, 0.3]
        vec = EmbeddingVector.from_list(values)
        arr = vec.to_numpy()
        
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, values, rtol=1e-5)
    
    def test_to_list(self):
        """Test conversion to list."""
        values = [0.1, 0.2, 0.3]
        vec = EmbeddingVector.from_list(values)
        result = vec.to_list()
        
        assert isinstance(result, list)
        np.testing.assert_allclose(result, values, rtol=1e-5)
    
    def test_normalize(self):
        """Test L2 normalization."""
        values = [3.0, 4.0]  # Norm = 5
        vec = EmbeddingVector.from_list(values)
        normalized = vec.normalize()
        
        arr = normalized.to_numpy()
        np.testing.assert_allclose(arr, [0.6, 0.8], rtol=1e-5)
    
    def test_iteration(self):
        """Test vector iteration."""
        values = [0.1, 0.2, 0.3]
        vec = EmbeddingVector.from_list(values)
        
        result = list(vec)
        np.testing.assert_allclose(result, values, rtol=1e-5)
    
    def test_indexing(self):
        """Test element access."""
        values = [0.1, 0.2, 0.3]
        vec = EmbeddingVector.from_list(values)
        
        assert abs(vec[0] - 0.1) < 1e-5
        assert abs(vec[1] - 0.2) < 1e-5


class TestSearchQuery:
    """Tests for SearchQuery."""
    
    def test_basic_query(self):
        """Test basic query construction."""
        vec = EmbeddingVector.from_list([0.1] * 128)
        query = SearchQuery(vector=vec, k=10)
        
        assert query.k == 10
        assert query.namespace == "default"
    
    def test_query_with_filters(self):
        """Test query with metadata filters."""
        vec = EmbeddingVector.from_list([0.1] * 128)
        query = SearchQuery(
            vector=vec,
            k=5,
            filters={"category": "tech"},
            score_threshold=0.7,
        )
        
        assert query.filters == {"category": "tech"}
        assert query.score_threshold == 0.7
    
    def test_validate_k(self):
        """Test k validation."""
        vec = EmbeddingVector.from_list([0.1] * 128)
        
        # Valid k
        query = SearchQuery(vector=vec, k=10)
        assert query.validate() is None
        
        # Invalid k
        query = SearchQuery(vector=vec, k=0)
        assert query.validate() is not None
    
    def test_get_vector_from_list(self):
        """Test getting vector from list input."""
        query = SearchQuery(vector=[0.1, 0.2, 0.3], k=5)
        vec = query.get_vector()
        
        assert isinstance(vec, EmbeddingVector)
        assert vec.dimension == 3


class TestSearchResult:
    """Tests for SearchResult."""
    
    def test_basic_result(self):
        """Test basic result construction."""
        vid = VectorId(value="doc-1")
        result = SearchResult(id=vid, score=0.95, rank=0)
        
        assert result.id == vid
        assert result.score == 0.95
        assert result.rank == 0
    
    def test_distance_conversion(self):
        """Test score to distance conversion."""
        vid = VectorId(value="doc-1")
        result = SearchResult(id=vid, score=0.95, rank=0)
        
        assert result.distance == 0.05
    
    def test_to_dict(self):
        """Test serialization."""
        vid = VectorId(value="doc-1")
        result = SearchResult(
            id=vid,
            score=0.95,
            rank=0,
            metadata={"title": "Test"},
        )
        
        d = result.to_dict()
        assert d["id"] == "doc-1"
        assert d["score"] == 0.95
        assert d["metadata"]["title"] == "Test"


class TestIndexConfig:
    """Tests for IndexConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = IndexConfig(dimension=768)
        
        assert config.dimension == 768
        assert config.metric == MetricType.COSINE
        assert config.max_elements == 1_000_000
    
    def test_validation(self):
        """Test config validation."""
        # Valid
        config = IndexConfig(dimension=768)
        assert config.validate() is None
        
        # Invalid dimension
        config = IndexConfig(dimension=0)
        assert config.validate() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
