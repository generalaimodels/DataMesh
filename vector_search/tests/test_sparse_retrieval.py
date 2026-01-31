"""
Unit Tests: SOTA Sparse Retrieval Components

Tests:
    - SparseVector operations (creation, dot, normalization)
    - Partitioned Elias-Fano compression
    - PostingList with block-max scores
    - InvertedIndex with WAND search
    - Hybrid fusion algorithms
"""

import pytest
import numpy as np

from vector_search.core.sparse_types import (
    SparseVector,
    PartitionedEliasFano,
    PostingList,
    BlockMaxIndex,
    SparseSearchResult,
)
from vector_search.index.inverted_index import InvertedIndex, InvertedIndexConfig
from vector_search.index.hybrid_retriever import (
    reciprocal_rank_fusion,
    linear_fusion,
    max_fusion,
    FusionMethod,
)


# =============================================================================
# SPARSE VECTOR TESTS
# =============================================================================
class TestSparseVector:
    """Tests for SparseVector operations."""
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        sv = SparseVector.from_dict({10: 1.5, 20: 2.0, 30: 0.5})
        
        assert sv.nnz == 3
        assert not sv.is_empty
        # Should be sorted by term_id
        assert list(sv.term_ids) == [10, 20, 30]
    
    def test_from_dict_with_pruning(self):
        """Test pruning of small weights."""
        sv = SparseVector.from_dict({10: 1.0, 20: 0.001}, prune_threshold=0.01)
        
        assert sv.nnz == 1
        assert sv.term_ids[0] == 10
    
    def test_from_arrays(self):
        """Test creation from parallel arrays."""
        sv = SparseVector.from_arrays([30, 10, 20], [0.3, 0.1, 0.2])
        
        assert sv.nnz == 3
        # Should be sorted
        assert list(sv.term_ids) == [10, 20, 30]
        assert list(sv.weights) == [0.1, 0.2, 0.3]
    
    def test_from_dense(self):
        """Test extraction from dense vector."""
        dense = np.zeros(100, dtype=np.float32)
        dense[10] = 1.5
        dense[50] = 2.0
        dense[90] = 0.5
        
        sv = SparseVector.from_dense(dense)
        
        assert sv.nnz == 3
        assert 10 in sv.term_ids
        assert 50 in sv.term_ids
        assert 90 in sv.term_ids
    
    def test_dot_product(self):
        """Test dot product between sparse vectors."""
        sv1 = SparseVector.from_dict({10: 1.0, 20: 2.0, 30: 0.5})
        sv2 = SparseVector.from_dict({20: 1.5, 30: 1.0, 40: 2.0})
        
        # dot = 2.0*1.5 + 0.5*1.0 = 3.0 + 0.5 = 3.5
        assert abs(sv1.dot(sv2) - 3.5) < 1e-6
    
    def test_dot_product_empty(self):
        """Test dot product with empty vector."""
        sv1 = SparseVector.from_dict({10: 1.0})
        sv2 = SparseVector.from_dict({})
        
        assert sv1.dot(sv2) == 0.0
    
    def test_dot_product_no_overlap(self):
        """Test dot product with no overlapping terms."""
        sv1 = SparseVector.from_dict({10: 1.0, 20: 2.0})
        sv2 = SparseVector.from_dict({30: 1.5, 40: 1.0})
        
        assert sv1.dot(sv2) == 0.0
    
    def test_l2_norm(self):
        """Test L2 norm computation."""
        sv = SparseVector.from_dict({10: 3.0, 20: 4.0})
        
        # sqrt(9 + 16) = 5.0
        assert abs(sv.l2_norm() - 5.0) < 1e-6
    
    def test_normalize(self):
        """Test L2 normalization."""
        sv = SparseVector.from_dict({10: 3.0, 20: 4.0})
        normalized = sv.normalize()
        
        assert abs(normalized.l2_norm() - 1.0) < 1e-6
    
    def test_top_k(self):
        """Test top-k weight selection."""
        sv = SparseVector.from_dict({10: 0.1, 20: 0.5, 30: 0.3, 40: 0.8})
        top2 = sv.top_k(2)
        
        assert top2.nnz == 2
        # Should have highest weights
        assert 20 in top2.term_ids
        assert 40 in top2.term_ids
    
    def test_serialization(self):
        """Test bytes serialization round-trip."""
        sv = SparseVector.from_dict({10: 1.5, 20: 2.0, 30: 0.5})
        data = sv.to_bytes()
        restored = SparseVector.from_bytes(data)
        
        assert restored.nnz == sv.nnz
        assert list(restored.term_ids) == list(sv.term_ids)
        assert abs(sv.dot(restored) - sv.dot(sv)) < 1e-6


# =============================================================================
# PEF COMPRESSION TESTS
# =============================================================================
class TestPartitionedEliasFano:
    """Tests for Partitioned Elias-Fano compression."""
    
    def test_encode_decode(self):
        """Test basic encode/decode round-trip."""
        values = [10, 25, 100, 250, 500]
        pef = PartitionedEliasFano.encode(values, universe=1000)
        decoded = pef.decode_all()
        
        assert list(decoded) == values
    
    def test_compression_ratio(self):
        """Test that compression achieves space savings."""
        values = list(range(0, 10000, 10))  # 1000 values
        pef = PartitionedEliasFano.encode(values, universe=10000)
        
        raw_size = len(values) * 4  # 32-bit ints
        compressed_size = pef.size_bytes
        
        assert compressed_size < raw_size
    
    def test_empty_sequence(self):
        """Test handling of empty sequence."""
        pef = PartitionedEliasFano.encode([], universe=100)
        
        assert pef.n == 0
        assert len(pef.decode_all()) == 0
    
    def test_single_element(self):
        """Test single-element sequence."""
        pef = PartitionedEliasFano.encode([42], universe=100)
        decoded = pef.decode_all()
        
        assert list(decoded) == [42]


# =============================================================================
# POSTING LIST TESTS
# =============================================================================
class TestPostingList:
    """Tests for compressed posting lists."""
    
    def test_build_and_decode(self):
        """Test posting list construction and decoding."""
        postings = [(0, 1.5), (5, 2.0), (10, 0.8)]
        pl = PostingList.build(term_id=42, postings=postings, universe=100)
        
        assert pl.df == 3
        assert pl.term_id == 42
        
        decoded = pl.decode_all()
        assert len(decoded) == 3
        assert decoded[0].doc_id == 0
        assert decoded[0].score == 1.5
    
    def test_block_max_scores(self):
        """Test block-max score computation."""
        # Create postings across multiple blocks
        postings = [(i, float(i % 10)) for i in range(200)]
        pl = PostingList.build(term_id=1, postings=postings, universe=1000, block_size=64)
        
        # Check that block max is correctly computed
        max_score = pl.get_block_max_score(0)
        assert max_score > 0


# =============================================================================
# INVERTED INDEX TESTS
# =============================================================================
class TestInvertedIndex:
    """Tests for Block-Max WAND inverted index."""
    
    @pytest.fixture
    def simple_index(self):
        """Create index with 3 documents."""
        idx = InvertedIndex()
        idx.add_document(0, SparseVector.from_dict({10: 1.0, 20: 2.0}))
        idx.add_document(1, SparseVector.from_dict({20: 1.5, 30: 1.0}))
        idx.add_document(2, SparseVector.from_dict({10: 0.5, 30: 2.0}))
        idx.compile()
        return idx
    
    def test_add_document(self):
        """Test document indexing."""
        idx = InvertedIndex()
        sv = SparseVector.from_dict({10: 1.0, 20: 2.0})
        idx.add_document(0, sv, external_id="doc-0")
        
        assert idx.num_documents == 1
        assert idx.vocabulary_size == 2
    
    def test_search_single_term(self, simple_index):
        """Test single-term query."""
        query = SparseVector.from_dict({20: 1.0})
        results = simple_index.search(query, k=3)
        
        assert len(results.matches) == 2
        # Doc 0 has weight 2.0, doc 1 has 1.5
        assert results.matches[0].doc_id == 0
        assert results.matches[0].score == 2.0
    
    def test_search_multi_term(self, simple_index):
        """Test multi-term query."""
        query = SparseVector.from_dict({10: 1.0, 30: 1.0})
        results = simple_index.search(query, k=3)
        
        # Doc 2 matches both terms: 0.5 + 2.0 = 2.5
        # Doc 0 matches term 10: 1.0
        # Doc 1 matches term 30: 1.0
        assert results.matches[0].doc_id == 2
    
    def test_search_no_results(self, simple_index):
        """Test query with no matching terms."""
        query = SparseVector.from_dict({99: 1.0})
        results = simple_index.search(query, k=3)
        
        assert len(results.matches) == 0
    
    def test_search_empty_index(self):
        """Test search on empty index."""
        idx = InvertedIndex()
        idx.compile()
        
        query = SparseVector.from_dict({10: 1.0})
        results = idx.search(query, k=3)
        
        assert len(results.matches) == 0


# =============================================================================
# FUSION ALGORITHM TESTS
# =============================================================================
class TestFusionAlgorithms:
    """Tests for score fusion algorithms."""
    
    def test_rrf_basic(self):
        """Test Reciprocal Rank Fusion."""
        # Two rankings
        ranking1 = [(1, 0.9), (2, 0.8), (3, 0.7)]
        ranking2 = [(2, 0.95), (3, 0.85), (1, 0.75)]
        
        fused = reciprocal_rank_fusion([ranking1, ranking2], k=60)
        
        # Doc 2 is rank 2 in r1 and rank 1 in r2 → should score highest
        assert max(fused, key=fused.get) == 2
    
    def test_rrf_missing_docs(self):
        """Test RRF with non-overlapping doc sets."""
        ranking1 = [(1, 0.9), (2, 0.8)]
        ranking2 = [(3, 0.95), (4, 0.85)]
        
        fused = reciprocal_rank_fusion([ranking1, ranking2])
        
        assert len(fused) == 4
        assert all(doc in fused for doc in [1, 2, 3, 4])
    
    def test_linear_fusion(self):
        """Test weighted linear combination."""
        sparse = [(1, 0.8), (2, 0.6)]
        dense = [(2, 0.9), (3, 0.7)]
        
        fused = linear_fusion(sparse, dense, 0.5, 0.5, normalize=False)
        
        # Doc 2 appears in both: 0.5*0.6 + 0.5*0.9 = 0.75
        assert abs(fused[2] - 0.75) < 1e-6
    
    def test_max_fusion(self):
        """Test max score fusion."""
        sparse = [(1, 0.8), (2, 0.6)]
        dense = [(2, 0.9), (3, 0.7)]
        
        fused = max_fusion(sparse, dense)
        
        assert fused[1] == 0.8  # Only in sparse
        assert fused[2] == 0.9  # Max of 0.6, 0.9
        assert fused[3] == 0.7  # Only in dense


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
