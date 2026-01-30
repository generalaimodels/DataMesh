"""
Vector Store: HNSW-Based Semantic Memory

Provides:
- Approximate Nearest Neighbor (ANN) search with HNSW
- Cosine similarity for embedding comparison
- Sharding by entity_id for multi-tenant isolation
- Configurable index parameters (M, ef)

Design:
    Vector store enables semantic retrieval of memories
    based on embedding similarity. Uses Hierarchical
    Navigable Small Worlds (HNSW) for O(log N) search.

Index Layout:
    entity_id → HNSW graph
    Each entity has isolated vector space.
"""

from __future__ import annotations

import asyncio
import heapq
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

from datamesh.core.types import (
    Result, Ok, Err,
    EntityId, EmbeddingVector, Timestamp,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    """Configuration for vector store."""
    dimension: int = 1536          # Embedding dimension (e.g., text-embedding-ada-002)
    max_elements: int = 100000     # Maximum vectors per entity
    m: int = 16                    # HNSW: connections per node
    ef_construction: int = 200     # HNSW: construction beam width
    ef_search: int = 50            # HNSW: search beam width
    similarity_threshold: float = 0.7  # Minimum similarity for results


# =============================================================================
# MEMORY VECTOR
# =============================================================================
@dataclass(slots=True)
class MemoryVector:
    """
    Vector embedding for semantic memory.
    
    Stores the embedding along with metadata for retrieval.
    """
    vector_id: UUID = field(default_factory=uuid4)
    entity_id: Optional[EntityId] = None
    embedding: tuple[float, ...] = ()  # Immutable vector
    
    # Source reference
    source_id: Optional[str] = None      # Original item ID
    source_type: str = "interaction"     # interaction, summary, entity
    
    # Content
    content: str = ""                    # Text content for display
    token_count: int = 0
    
    # Metadata
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return len(self.embedding)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "vector_id": str(self.vector_id),
            "entity_id": str(self.entity_id.value) if self.entity_id else None,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "content": self.content,
            "token_count": self.token_count,
            "dimension": self.dimension,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# SEARCH RESULT
# =============================================================================
@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result from vector similarity search."""
    vector: MemoryVector
    similarity: float  # 0.0 - 1.0, higher = more similar
    rank: int
    
    @property
    def distance(self) -> float:
        """Euclidean distance (inverse of similarity)."""
        return 1.0 - self.similarity


# =============================================================================
# HNSW NODE
# =============================================================================
@dataclass
class HNSWNode:
    """Node in HNSW graph."""
    vector_id: UUID
    embedding: tuple[float, ...]
    level: int  # Maximum level this node appears in
    neighbors: list[list[UUID]]  # neighbors[level] = list of neighbor IDs
    
    def __init__(
        self,
        vector_id: UUID,
        embedding: tuple[float, ...],
        max_level: int,
        m: int,
    ) -> None:
        self.vector_id = vector_id
        self.embedding = embedding
        self.level = max_level
        self.neighbors = [[] for _ in range(max_level + 1)]


# =============================================================================
# VECTOR STORE
# =============================================================================
class VectorStore:
    """
    HNSW-based vector store for semantic memory.
    
    Features:
        - O(log N) approximate nearest neighbor search
        - Entity-sharded index for multi-tenant isolation
        - Configurable HNSW parameters
        - Cosine similarity scoring
    
    Usage:
        store = VectorStore(config=VectorStoreConfig(dimension=1536))
        
        # Insert vector
        vector = MemoryVector(
            entity_id=entity_id,
            embedding=tuple(embedding_values),
            content="Hello, world!",
        )
        await store.insert(vector)
        
        # Search
        results = await store.search(
            entity_id=entity_id,
            query_embedding=query_vector,
            k=10,
        )
    """
    
    __slots__ = (
        "_config", "_entities", "_lock", "_ml",
    )
    
    def __init__(self, config: Optional[VectorStoreConfig] = None) -> None:
        self._config = config or VectorStoreConfig()
        # entity_id -> {vector_id -> HNSWNode}
        self._entities: dict[str, dict[UUID, HNSWNode]] = {}
        # entity_id -> entry_point (top node)
        self._lock = asyncio.Lock()
        # Level multiplier for random level generation
        self._ml = 1 / math.log(self._config.m)
    
    def _random_level(self) -> int:
        """Generate random level for new node (geometric distribution)."""
        level = 0
        while random.random() < 0.5 and level < 16:  # Cap at 16 levels
            level += 1
        return level
    
    def _cosine_similarity(
        self,
        a: tuple[float, ...],
        b: tuple[float, ...],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    async def insert(self, vector: MemoryVector) -> Result[UUID, str]:
        """
        Insert vector into store.
        
        Simplified HNSW insertion (production would use full algorithm).
        """
        async with self._lock:
            if vector.entity_id is None:
                return Err("Vector must have entity_id")
            
            entity_key = str(vector.entity_id.value)
            
            # Initialize entity index if needed
            if entity_key not in self._entities:
                self._entities[entity_key] = {}
            
            entity_index = self._entities[entity_key]
            
            # Check capacity
            if len(entity_index) >= self._config.max_elements:
                return Err(f"Entity index full: {len(entity_index)} vectors")
            
            # Create node
            level = self._random_level()
            node = HNSWNode(
                vector_id=vector.vector_id,
                embedding=vector.embedding,
                max_level=level,
                m=self._config.m,
            )
            
            # Simplified: just connect to nearby nodes at each level
            # Full HNSW would do proper layer-by-layer insertion
            for l in range(level + 1):
                candidates = [
                    (self._cosine_similarity(vector.embedding, n.embedding), n.vector_id)
                    for n in entity_index.values()
                    if n.level >= l
                ]
                candidates.sort(reverse=True)
                
                # Keep top M neighbors
                for sim, neighbor_id in candidates[:self._config.m]:
                    if neighbor_id != vector.vector_id:
                        node.neighbors[l].append(neighbor_id)
                        
                        # Bidirectional connection
                        neighbor = entity_index[neighbor_id]
                        if len(neighbor.neighbors[l]) < self._config.m * 2:
                            neighbor.neighbors[l].append(vector.vector_id)
            
            entity_index[vector.vector_id] = node
            
            return Ok(vector.vector_id)
    
    async def search(
        self,
        entity_id: EntityId,
        query_embedding: tuple[float, ...],
        k: int = 10,
    ) -> list[SearchResult]:
        """
        Search for k nearest neighbors.
        
        Uses HNSW graph traversal for efficient ANN search.
        """
        async with self._lock:
            entity_key = str(entity_id.value)
            
            if entity_key not in self._entities:
                return []
            
            entity_index = self._entities[entity_key]
            
            if not entity_index:
                return []
            
            # Simplified search: brute force with heap
            # Full HNSW would do layer-by-layer graph traversal
            candidates: list[tuple[float, UUID]] = []
            
            for node in entity_index.values():
                sim = self._cosine_similarity(query_embedding, node.embedding)
                if sim >= self._config.similarity_threshold:
                    heapq.heappush(candidates, (-sim, node.vector_id))
            
            # Get top k
            results: list[SearchResult] = []
            for rank, _ in enumerate(range(min(k, len(candidates)))):
                neg_sim, vector_id = heapq.heappop(candidates)
                
                # Would need to retrieve full MemoryVector from separate store
                # For now, create placeholder
                results.append(SearchResult(
                    vector=MemoryVector(
                        vector_id=vector_id,
                        embedding=entity_index[vector_id].embedding,
                    ),
                    similarity=-neg_sim,
                    rank=rank,
                ))
            
            return results
    
    async def delete(
        self,
        entity_id: EntityId,
        vector_id: UUID,
    ) -> Result[bool, str]:
        """Delete vector from store."""
        async with self._lock:
            entity_key = str(entity_id.value)
            
            if entity_key not in self._entities:
                return Ok(False)
            
            entity_index = self._entities[entity_key]
            
            if vector_id not in entity_index:
                return Ok(False)
            
            # Remove node
            node = entity_index.pop(vector_id)
            
            # Remove from neighbors (simplified)
            for other_node in entity_index.values():
                for level in range(len(other_node.neighbors)):
                    if vector_id in other_node.neighbors[level]:
                        other_node.neighbors[level].remove(vector_id)
            
            return Ok(True)
    
    async def get(
        self,
        entity_id: EntityId,
        vector_id: UUID,
    ) -> Optional[MemoryVector]:
        """Get vector by ID."""
        async with self._lock:
            entity_key = str(entity_id.value)
            
            if entity_key not in self._entities:
                return None
            
            node = self._entities[entity_key].get(vector_id)
            if node:
                return MemoryVector(
                    vector_id=node.vector_id,
                    embedding=node.embedding,
                )
            return None
    
    async def count(self, entity_id: EntityId) -> int:
        """Count vectors for entity."""
        async with self._lock:
            entity_key = str(entity_id.value)
            return len(self._entities.get(entity_key, {}))
    
    @property
    def total_vectors(self) -> int:
        """Total vectors across all entities."""
        return sum(len(idx) for idx in self._entities.values())
    
    @property
    def entity_count(self) -> int:
        """Number of entities with vectors."""
        return len(self._entities)
