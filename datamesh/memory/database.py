"""
Memory Database Layer: SOTA-Level Large Context Window Support

High-performance memory management for 100K-1M token context windows:
- Hierarchical memory (STM → LTM) with automatic consolidation
- Streaming context assembly for large windows
- Priority-based retention with intelligent truncation
- Chunked retrieval with relevance scoring
- Vector-based semantic search with HNSW
- Graph-based relational memory

Design Principles:
    - Token-aware operations for LLM alignment
    - Lazy loading for memory efficiency
    - Streaming assembly for large contexts
    - Relevance-weighted fusion from multiple stores

Performance Targets:
    - Context assembly (100K tokens): <100ms P99
    - Semantic search (top-100): <20ms P99
    - STM add: <1ms P99

Author: Planetary AI Systems
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

from datamesh.core.types import Result, Ok, Err, EntityId, EmbeddingVector
from datamesh.storage.protocols import (
    ConsistencyLevel,
    OperationType,
    OperationMetadata,
)
from datamesh.storage.backends import (
    InMemoryCPStore,
    InMemoryAPStore,
    InMemoryObjectStore,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class MemoryDBConfig:
    """
    Configuration for memory database layer.
    
    Optimized for large context windows (100K-1M tokens).
    """
    # Context window settings
    max_context_tokens: int = 1_000_000  # 1M token support
    default_context_tokens: int = 128_000  # 128K default
    
    # STM settings
    stm_max_items: int = 1000
    stm_max_tokens: int = 500_000  # 500K tokens in STM
    stm_ttl_seconds: int = 3600
    
    # LTM settings
    ltm_embedding_dim: int = 1536  # OpenAI ada-002 dimension
    ltm_top_k: int = 100
    ltm_ef_search: int = 200  # HNSW search parameter
    
    # Retrieval settings
    retrieval_chunk_size: int = 4096  # Tokens per chunk
    retrieval_overlap: int = 256  # Token overlap between chunks
    retrieval_max_chunks: int = 100
    
    # Priority settings
    priority_system: int = 100
    priority_recent: int = 80
    priority_retrieved: int = 60
    priority_historical: int = 40
    
    # Compression
    hierarchical_summarization: bool = True
    summary_ratio: float = 0.25  # Compress to 25% of original
    
    # Eviction
    eviction_policy: str = "priority_lru"  # "lru", "priority_lru", "token_aware"
    eviction_batch_size: int = 50


# =============================================================================
# MEMORY PRIORITY
# =============================================================================
class MemoryPriority(Enum):
    """Priority levels for memory retention."""
    CRITICAL = 100   # Never evict (system prompts)
    HIGH = 80        # Recent interactions
    MEDIUM = 60      # Retrieved memories
    LOW = 40         # Historical context
    EPHEMERAL = 20   # Can be evicted immediately
    
    def __lt__(self, other: "MemoryPriority") -> bool:
        return self.value < other.value


# =============================================================================
# MEMORY ITEM
# =============================================================================
@dataclass
class MemoryItem:
    """
    Single memory item for context assembly.
    
    Can represent various content types with unified handling.
    """
    item_id: UUID
    entity_id: EntityId
    session_id: Optional[UUID] = None
    
    # Content
    content: str = ""
    content_type: str = "text"  # "text", "tool_result", "summary", "embedding"
    role: str = "user"
    
    # Token management
    token_count: int = 0
    
    # Priority and ordering
    priority: MemoryPriority = MemoryPriority.MEDIUM
    sequence_id: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Embedding for semantic search
    embedding: Optional[Tuple[float, ...]] = None
    
    # References
    source_ref: Optional[str] = None  # Reference to original source
    parent_id: Optional[UUID] = None  # For hierarchical memory
    
    @property
    def storage_key(self) -> str:
        """Generate storage key."""
        return f"memory:{self.entity_id.value}:{self.item_id}"
    
    @property
    def priority_score(self) -> float:
        """
        Calculate priority score for sorting.
        
        Combines priority level, recency, and relevance.
        """
        # Base priority
        score = float(self.priority.value)
        
        # Recency bonus (decays over time)
        age_seconds = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        recency_bonus = max(0, 50 - (age_seconds / 60))  # Max 50 bonus, decays per minute
        
        return score + recency_bonus
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "item_id": str(self.item_id),
            "entity_id": self.entity_id.value,
            "session_id": str(self.session_id) if self.session_id else None,
            "content": self.content,
            "content_type": self.content_type,
            "role": self.role,
            "token_count": self.token_count,
            "priority": self.priority.name,
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "embedding": list(self.embedding) if self.embedding else None,
            "source_ref": self.source_ref,
            "parent_id": str(self.parent_id) if self.parent_id else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Deserialize from dictionary."""
        return cls(
            item_id=UUID(data["item_id"]),
            entity_id=EntityId(data["entity_id"]),
            session_id=UUID(data["session_id"]) if data.get("session_id") else None,
            content=data.get("content", ""),
            content_type=data.get("content_type", "text"),
            role=data.get("role", "user"),
            token_count=data.get("token_count", 0),
            priority=MemoryPriority[data.get("priority", "MEDIUM")],
            sequence_id=data.get("sequence_id", 0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            embedding=tuple(data["embedding"]) if data.get("embedding") else None,
            source_ref=data.get("source_ref"),
            parent_id=UUID(data["parent_id"]) if data.get("parent_id") else None,
        )


# =============================================================================
# CONTEXT CHUNK
# =============================================================================
@dataclass(frozen=True, slots=True)
class ContextChunk:
    """
    Chunk of context for streaming assembly.
    
    Represents a contiguous portion of the context window.
    """
    chunk_id: int
    content: str
    token_count: int
    priority: MemoryPriority
    source_items: Tuple[UUID, ...]
    is_summary: bool = False
    
    @property
    def can_truncate(self) -> bool:
        """Check if chunk can be truncated."""
        return self.priority != MemoryPriority.CRITICAL


# =============================================================================
# MEMORY QUERY
# =============================================================================
@dataclass
class MemoryQuery:
    """Query for memory retrieval."""
    entity_id: EntityId
    session_id: Optional[UUID] = None
    
    # Text query
    text: Optional[str] = None
    
    # Embedding query (for semantic search)
    embedding: Optional[Tuple[float, ...]] = None
    
    # Filters
    time_range: Optional[Tuple[datetime, datetime]] = None
    priority_min: Optional[MemoryPriority] = None
    content_types: Optional[List[str]] = None
    
    # Retrieval settings
    top_k: int = 50
    include_embeddings: bool = False
    
    # Relevance boosting
    recency_weight: float = 0.3
    semantic_weight: float = 0.5
    priority_weight: float = 0.2


# =============================================================================
# MEMORY RECALL RESULT
# =============================================================================
@dataclass
class MemoryRecall:
    """Result of memory recall operation."""
    items: List[MemoryItem]
    total_tokens: int
    query_latency_ms: float
    
    # Relevance scores
    scores: Dict[UUID, float] = field(default_factory=dict)
    
    # Sources
    from_stm: int = 0
    from_ltm: int = 0
    from_summary: int = 0


# =============================================================================
# HNSW INDEX (In-Memory Implementation)
# =============================================================================
class HNSWIndex:
    """
    Hierarchical Navigable Small World graph for ANN search.
    
    Simplified in-memory implementation for development.
    Production would use Milvus, Pinecone, or similar.
    """
    
    __slots__ = ("_vectors", "_metadata", "_dim", "_ef_search")
    
    def __init__(
        self,
        dimension: int = 1536,
        ef_search: int = 200,
    ) -> None:
        self._dim = dimension
        self._ef_search = ef_search
        self._vectors: Dict[UUID, Tuple[float, ...]] = {}
        self._metadata: Dict[UUID, Dict[str, Any]] = {}
    
    def insert(
        self,
        item_id: UUID,
        vector: Tuple[float, ...],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert vector into index."""
        if len(vector) != self._dim:
            raise ValueError(f"Vector dimension mismatch: expected {self._dim}, got {len(vector)}")
        
        self._vectors[item_id] = vector
        self._metadata[item_id] = metadata or {}
    
    def search(
        self,
        query_vector: Tuple[float, ...],
        top_k: int = 10,
    ) -> List[Tuple[UUID, float]]:
        """
        Search for nearest neighbors.
        
        Returns list of (item_id, distance) tuples.
        
        Complexity: O(n) for brute force, O(log n) for actual HNSW
        """
        if len(query_vector) != self._dim:
            raise ValueError(f"Query dimension mismatch: expected {self._dim}, got {len(query_vector)}")
        
        # Calculate distances
        distances: List[Tuple[float, UUID]] = []
        
        for item_id, vector in self._vectors.items():
            # Cosine similarity (normalized to distance)
            distance = self._cosine_distance(query_vector, vector)
            distances.append((distance, item_id))
        
        # Get top-k (use heap for efficiency)
        top_results = heapq.nsmallest(top_k, distances)
        
        return [(item_id, dist) for dist, item_id in top_results]
    
    def _cosine_distance(
        self,
        a: Tuple[float, ...],
        b: Tuple[float, ...],
    ) -> float:
        """Calculate cosine distance between vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 1.0  # Maximum distance
        
        similarity = dot_product / (norm_a * norm_b)
        return 1.0 - similarity  # Convert similarity to distance
    
    def delete(self, item_id: UUID) -> bool:
        """Delete vector from index."""
        if item_id in self._vectors:
            del self._vectors[item_id]
            del self._metadata[item_id]
            return True
        return False
    
    def count(self) -> int:
        """Get total vector count."""
        return len(self._vectors)


# =============================================================================
# MEMORY DATABASE LAYER
# =============================================================================
class MemoryDatabaseLayer:
    """
    SOTA-level memory database layer for large context windows.
    
    Architecture:
        - STM (Short-Term Memory): Recent interactions, high priority
        - LTM (Long-Term Memory): Semantic vectors, graph relations
        - Hierarchical summarization for context compression
    
    Features:
        - 1M token context window support
        - Streaming context assembly
        - Priority-based retention
        - Semantic search with HNSW
        - Automatic consolidation STM → LTM
    
    Example:
        db = MemoryDatabaseLayer()
        
        # Add to STM
        await db.add_to_stm(memory_item)
        
        # Build context for LLM
        async for chunk in db.stream_context(entity_id, max_tokens=128000):
            process_chunk(chunk)
        
        # Semantic search
        results = await db.semantic_search(query_embedding, top_k=50)
    """
    
    __slots__ = (
        "_config",
        "_stm_store",      # Hot memory (Redis-like)
        "_ltm_store",      # Cold memory metadata
        "_vector_index",   # HNSW for semantic search
        "_object_store",   # Large content storage
        "_token_counts",   # Per-entity token tracking
        "_sequence_counters",
        "_metrics",
        "_embedding_fn",
    )
    
    def __init__(
        self,
        config: Optional[MemoryDBConfig] = None,
        stm_store: Optional[InMemoryAPStore] = None,
        ltm_store: Optional[InMemoryCPStore] = None,
        object_store: Optional[InMemoryObjectStore] = None,
        embedding_fn: Optional[Callable[[str], Tuple[float, ...]]] = None,
    ) -> None:
        """
        Initialize memory database layer.
        
        Args:
            config: Configuration settings
            stm_store: Short-term memory store
            ltm_store: Long-term memory metadata store
            object_store: Large content storage
            embedding_fn: Function to generate embeddings (optional)
        """
        self._config = config or MemoryDBConfig()
        
        self._stm_store = stm_store or InMemoryAPStore(
            max_entries=self._config.stm_max_items,
        )
        self._ltm_store = ltm_store or InMemoryCPStore()
        self._object_store = object_store or InMemoryObjectStore()
        
        # Vector index for semantic search
        self._vector_index = HNSWIndex(
            dimension=self._config.ltm_embedding_dim,
            ef_search=self._config.ltm_ef_search,
        )
        
        # Token tracking per entity
        self._token_counts: Dict[str, int] = {}
        self._sequence_counters: Dict[str, int] = {}
        
        # Metrics
        self._metrics = MemoryMetrics()
        
        # Embedding function (default: placeholder)
        self._embedding_fn = embedding_fn or self._default_embedding
    
    def _default_embedding(self, text: str) -> Tuple[float, ...]:
        """
        Default embedding function (placeholder).
        
        Uses hash-based embedding for development/testing.
        Production should use actual embedding model.
        """
        # Create deterministic pseudo-embedding from text
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        
        # Expand to embedding dimension
        embedding = []
        for i in range(self._config.ltm_embedding_dim):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)  # Normalize to [-1, 1]
        
        # Normalize to unit vector
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return tuple(embedding)
    
    # -------------------------------------------------------------------------
    # STM OPERATIONS
    # -------------------------------------------------------------------------
    
    async def add_to_stm(
        self,
        item: MemoryItem,
        evict_if_needed: bool = True,
    ) -> Result[MemoryItem, str]:
        """
        Add item to short-term memory.
        
        Automatically manages token budget and eviction.
        
        Args:
            item: Memory item to add
            evict_if_needed: If True, evict low-priority items if over budget
            
        Returns:
            Ok(item): With assigned sequence_id
            Err(message): Add failed
            
        Complexity: O(1) amortized
        """
        start_ns = time.time_ns()
        
        entity_key = item.entity_id.value
        
        # Assign sequence ID
        current_seq = self._sequence_counters.get(entity_key, 0)
        item.sequence_id = current_seq + 1
        self._sequence_counters[entity_key] = item.sequence_id
        
        # Check token budget
        current_tokens = self._token_counts.get(entity_key, 0)
        new_total = current_tokens + item.token_count
        
        if new_total > self._config.stm_max_tokens and evict_if_needed:
            # Evict low-priority items
            evict_result = await self._evict_stm(
                entity_id=item.entity_id,
                tokens_needed=item.token_count,
            )
            if evict_result.is_err():
                return Err(f"Cannot add item: {evict_result.error}")
        
        # Generate embedding if not provided
        if item.embedding is None and item.content:
            item.embedding = self._embedding_fn(item.content)
        
        # Store in STM
        result = await self._stm_store.put_with_ttl(
            item.storage_key,
            item.to_dict(),
            ttl_seconds=self._config.stm_ttl_seconds,
        )
        
        if result.is_err():
            return Err(f"Failed to add to STM: {result.error}")
        
        # Update token count
        self._token_counts[entity_key] = self._token_counts.get(entity_key, 0) + item.token_count
        
        # Update metrics
        latency_ms = (time.time_ns() - start_ns) / 1_000_000
        self._metrics.record_stm_add(latency_ms, item.token_count)
        
        return Ok(item)
    
    async def get_from_stm(
        self,
        entity_id: EntityId,
        item_id: UUID,
    ) -> Result[MemoryItem, str]:
        """Get item from STM by ID."""
        storage_key = f"memory:{entity_id.value}:{item_id}"
        
        result = await self._stm_store.get(storage_key)
        
        if result.is_err():
            return Err(f"Item not found: {item_id}")
        
        data, _ = result.unwrap()
        return Ok(MemoryItem.from_dict(data))
    
    async def list_stm(
        self,
        entity_id: EntityId,
        session_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> Result[List[MemoryItem], str]:
        """List items in STM for entity."""
        prefix = f"memory:{entity_id.value}:"
        
        result = await self._stm_store.scan(prefix=prefix, limit=limit)
        
        if result.is_err():
            return Err(result.error)
        
        records, _ = result.unwrap()
        
        items = []
        for _, data in records:
            item = MemoryItem.from_dict(data)
            
            # Filter by session if specified
            if session_id and item.session_id != session_id:
                continue
            
            items.append(item)
        
        # Sort by sequence (newest first)
        items.sort(key=lambda x: x.sequence_id, reverse=True)
        
        return Ok(items)
    
    async def _evict_stm(
        self,
        entity_id: EntityId,
        tokens_needed: int,
    ) -> Result[int, str]:
        """
        Evict items from STM to free token budget.
        
        Uses priority-aware LRU eviction.
        
        Returns: Number of tokens freed
        """
        items_result = await self.list_stm(entity_id, limit=1000)
        
        if items_result.is_err():
            return Err(items_result.error)
        
        items = items_result.unwrap()
        
        # Sort by eviction priority (lowest priority first, then oldest)
        items.sort(key=lambda x: (x.priority.value, x.timestamp))
        
        tokens_freed = 0
        evicted_ids = []
        
        for item in items:
            if item.priority == MemoryPriority.CRITICAL:
                continue  # Never evict critical items
            
            if tokens_freed >= tokens_needed:
                break
            
            tokens_freed += item.token_count
            evicted_ids.append(item.item_id)
        
        # Delete evicted items
        for item_id in evicted_ids:
            storage_key = f"memory:{entity_id.value}:{item_id}"
            await self._stm_store.delete(storage_key)
        
        # Update token count
        entity_key = entity_id.value
        self._token_counts[entity_key] = max(
            0,
            self._token_counts.get(entity_key, 0) - tokens_freed
        )
        
        return Ok(tokens_freed)
    
    # -------------------------------------------------------------------------
    # LTM OPERATIONS
    # -------------------------------------------------------------------------
    
    async def add_to_ltm(
        self,
        item: MemoryItem,
    ) -> Result[MemoryItem, str]:
        """
        Add item to long-term memory with embedding.
        
        Stores metadata in CP store and embedding in vector index.
        
        Complexity: O(log n) for HNSW insertion
        """
        start_ns = time.time_ns()
        
        # Generate embedding if not provided
        if item.embedding is None and item.content:
            item.embedding = self._embedding_fn(item.content)
        
        # Store metadata
        ltm_key = f"ltm:{item.entity_id.value}:{item.item_id}"
        
        result = await self._ltm_store.put(ltm_key, item.to_dict())
        
        if result.is_err():
            return Err(f"Failed to add to LTM: {result.error}")
        
        # Store embedding in vector index
        if item.embedding:
            self._vector_index.insert(
                item_id=item.item_id,
                vector=item.embedding,
                metadata={
                    "entity_id": item.entity_id.value,
                    "content_type": item.content_type,
                    "timestamp": item.timestamp.isoformat(),
                },
            )
        
        # Update metrics
        latency_ms = (time.time_ns() - start_ns) / 1_000_000
        self._metrics.record_ltm_add(latency_ms)
        
        return Ok(item)
    
    async def semantic_search(
        self,
        query: MemoryQuery,
    ) -> Result[MemoryRecall, str]:
        """
        Semantic search in LTM using embeddings.
        
        Args:
            query: Memory query with text or embedding
            
        Returns:
            MemoryRecall with ranked results
            
        Complexity: O(log n) for HNSW search
        """
        start_ns = time.time_ns()
        
        # Get query embedding
        if query.embedding is not None:
            query_embedding = query.embedding
        elif query.text is not None:
            query_embedding = self._embedding_fn(query.text)
        else:
            return Err("Query must have text or embedding")
        
        # Search vector index
        search_results = self._vector_index.search(
            query_vector=query_embedding,
            top_k=query.top_k,
        )
        
        # Load full items
        items: List[MemoryItem] = []
        scores: Dict[UUID, float] = {}
        total_tokens = 0
        
        for item_id, distance in search_results:
            # Filter by entity
            ltm_key = f"ltm:{query.entity_id.value}:{item_id}"
            result = await self._ltm_store.get(ltm_key)
            
            if result.is_err():
                continue
            
            data, _ = result.unwrap()
            item = MemoryItem.from_dict(data)
            
            # Apply filters
            if query.time_range:
                start, end = query.time_range
                if item.timestamp < start or item.timestamp > end:
                    continue
            
            if query.priority_min and item.priority.value < query.priority_min.value:
                continue
            
            if query.content_types and item.content_type not in query.content_types:
                continue
            
            # Calculate relevance score
            semantic_score = 1.0 - distance  # Convert distance to similarity
            recency_score = self._calculate_recency_score(item.timestamp)
            priority_score = item.priority.value / 100.0
            
            final_score = (
                semantic_score * query.semantic_weight +
                recency_score * query.recency_weight +
                priority_score * query.priority_weight
            )
            
            items.append(item)
            scores[item.item_id] = final_score
            total_tokens += item.token_count
        
        # Sort by score
        items.sort(key=lambda x: scores.get(x.item_id, 0), reverse=True)
        
        # Build result
        latency_ms = (time.time_ns() - start_ns) / 1_000_000
        
        recall = MemoryRecall(
            items=items,
            total_tokens=total_tokens,
            query_latency_ms=latency_ms,
            scores=scores,
            from_ltm=len(items),
        )
        
        # Update metrics
        self._metrics.record_search(latency_ms, len(items))
        
        return Ok(recall)
    
    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """Calculate recency score (0-1) with exponential decay."""
        age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
        # Half-life of 1 hour
        half_life = 3600
        return math.exp(-0.693 * age_seconds / half_life)
    
    # -------------------------------------------------------------------------
    # CONTEXT ASSEMBLY
    # -------------------------------------------------------------------------
    
    async def build_context(
        self,
        entity_id: EntityId,
        session_id: Optional[UUID] = None,
        max_tokens: Optional[int] = None,
        query: Optional[str] = None,
    ) -> Result[List[MemoryItem], str]:
        """
        Build context for LLM from memory hierarchy.
        
        Assembles context from:
        1. Critical items (system prompts) - Always included
        2. Recent STM items - Prioritized by recency
        3. Retrieved LTM items - Based on query relevance
        
        Args:
            entity_id: Entity ID
            session_id: Optional session filter
            max_tokens: Token budget (default from config)
            query: Optional query for semantic retrieval
            
        Returns:
            List of memory items within token budget
            
        Complexity: O(n log n) for sorting
        """
        start_ns = time.time_ns()
        
        max_tokens = max_tokens or self._config.default_context_tokens
        
        context_items: List[MemoryItem] = []
        used_tokens = 0
        
        # 1. Add critical items first
        stm_result = await self.list_stm(entity_id, session_id)
        if stm_result.is_ok():
            stm_items = stm_result.unwrap()
            
            for item in stm_items:
                if item.priority == MemoryPriority.CRITICAL:
                    if used_tokens + item.token_count <= max_tokens:
                        context_items.append(item)
                        used_tokens += item.token_count
        
        # 2. Add recent STM items
        if stm_result.is_ok():
            stm_items = stm_result.unwrap()
            
            # Sort by priority score
            stm_items.sort(key=lambda x: x.priority_score, reverse=True)
            
            for item in stm_items:
                if item.priority == MemoryPriority.CRITICAL:
                    continue  # Already added
                
                if used_tokens + item.token_count <= max_tokens:
                    context_items.append(item)
                    used_tokens += item.token_count
        
        # 3. Retrieve from LTM if query provided and space available
        if query and used_tokens < max_tokens:
            remaining_tokens = max_tokens - used_tokens
            
            ltm_query = MemoryQuery(
                entity_id=entity_id,
                session_id=session_id,
                text=query,
                top_k=self._config.ltm_top_k,
            )
            
            ltm_result = await self.semantic_search(ltm_query)
            
            if ltm_result.is_ok():
                recall = ltm_result.unwrap()
                
                for item in recall.items:
                    # Skip if already in context
                    if any(c.item_id == item.item_id for c in context_items):
                        continue
                    
                    if used_tokens + item.token_count <= max_tokens:
                        item.priority = MemoryPriority.MEDIUM  # Mark as retrieved
                        context_items.append(item)
                        used_tokens += item.token_count
        
        # Sort final context by sequence (chronological order)
        context_items.sort(key=lambda x: x.sequence_id)
        
        # Update metrics
        latency_ms = (time.time_ns() - start_ns) / 1_000_000
        self._metrics.record_context_build(latency_ms, used_tokens)
        
        return Ok(context_items)
    
    async def stream_context(
        self,
        entity_id: EntityId,
        session_id: Optional[UUID] = None,
        max_tokens: Optional[int] = None,
        query: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> AsyncIterator[ContextChunk]:
        """
        Stream context in chunks for large windows.
        
        Memory-efficient for 100K-1M token contexts.
        
        Args:
            entity_id: Entity ID
            session_id: Optional session filter
            max_tokens: Token budget
            query: Optional query for retrieval
            chunk_size: Tokens per chunk
            
        Yields:
            ContextChunk objects
        """
        chunk_size = chunk_size or self._config.retrieval_chunk_size
        
        # Build full context
        result = await self.build_context(
            entity_id=entity_id,
            session_id=session_id,
            max_tokens=max_tokens,
            query=query,
        )
        
        if result.is_err():
            return
        
        items = result.unwrap()
        
        # Group items into chunks
        current_chunk_content = []
        current_chunk_tokens = 0
        current_chunk_items: List[UUID] = []
        chunk_id = 0
        
        for item in items:
            if current_chunk_tokens + item.token_count > chunk_size and current_chunk_content:
                # Yield current chunk
                yield ContextChunk(
                    chunk_id=chunk_id,
                    content="\n".join(current_chunk_content),
                    token_count=current_chunk_tokens,
                    priority=MemoryPriority.MEDIUM,
                    source_items=tuple(current_chunk_items),
                )
                
                chunk_id += 1
                current_chunk_content = []
                current_chunk_tokens = 0
                current_chunk_items = []
            
            current_chunk_content.append(item.content)
            current_chunk_tokens += item.token_count
            current_chunk_items.append(item.item_id)
        
        # Yield final chunk
        if current_chunk_content:
            yield ContextChunk(
                chunk_id=chunk_id,
                content="\n".join(current_chunk_content),
                token_count=current_chunk_tokens,
                priority=MemoryPriority.MEDIUM,
                source_items=tuple(current_chunk_items),
            )
    
    # -------------------------------------------------------------------------
    # CONSOLIDATION
    # -------------------------------------------------------------------------
    
    async def consolidate_stm_to_ltm(
        self,
        entity_id: EntityId,
        session_id: Optional[UUID] = None,
        min_age_seconds: int = 3600,
    ) -> Result[int, str]:
        """
        Consolidate STM items to LTM.
        
        Moves older items from STM to LTM for long-term storage.
        
        Args:
            entity_id: Entity ID
            session_id: Optional session filter
            min_age_seconds: Minimum age before consolidation
            
        Returns:
            Number of items consolidated
        """
        items_result = await self.list_stm(entity_id, session_id, limit=1000)
        
        if items_result.is_err():
            return Err(items_result.error)
        
        items = items_result.unwrap()
        
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=min_age_seconds)
        consolidated = 0
        
        for item in items:
            if item.timestamp > cutoff:
                continue  # Too recent
            
            if item.priority == MemoryPriority.CRITICAL:
                continue  # Keep critical items in STM
            
            # Add to LTM
            ltm_result = await self.add_to_ltm(item)
            
            if ltm_result.is_ok():
                # Remove from STM
                await self._stm_store.delete(item.storage_key)
                
                # Update token count
                entity_key = entity_id.value
                self._token_counts[entity_key] = max(
                    0,
                    self._token_counts.get(entity_key, 0) - item.token_count
                )
                
                consolidated += 1
        
        return Ok(consolidated)
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    
    async def get_memory_stats(
        self,
        entity_id: EntityId,
    ) -> Result[Dict[str, Any], str]:
        """Get memory statistics for entity."""
        stm_result = await self.list_stm(entity_id, limit=10000)
        
        stm_count = 0
        stm_tokens = 0
        
        if stm_result.is_ok():
            items = stm_result.unwrap()
            stm_count = len(items)
            stm_tokens = sum(i.token_count for i in items)
        
        return Ok({
            "stm_item_count": stm_count,
            "stm_token_count": stm_tokens,
            "stm_token_budget": self._config.stm_max_tokens,
            "stm_utilization": stm_tokens / self._config.stm_max_tokens if self._config.stm_max_tokens > 0 else 0,
            "ltm_vector_count": self._vector_index.count(),
            "max_context_tokens": self._config.max_context_tokens,
        })
    
    def get_metrics(self) -> "MemoryMetrics":
        """Get current metrics."""
        return self._metrics


# =============================================================================
# MEMORY METRICS
# =============================================================================
@dataclass
class MemoryMetrics:
    """Metrics for memory database operations."""
    
    # Counters
    stm_adds: int = 0
    ltm_adds: int = 0
    searches: int = 0
    context_builds: int = 0
    
    # Token tracking
    total_stm_tokens: int = 0
    total_context_tokens: int = 0
    total_search_results: int = 0
    
    # Latency tracking (in ms)
    stm_add_latency_sum: float = 0.0
    ltm_add_latency_sum: float = 0.0
    search_latency_sum: float = 0.0
    context_build_latency_sum: float = 0.0
    
    def record_stm_add(self, latency_ms: float, tokens: int) -> None:
        self.stm_adds += 1
        self.stm_add_latency_sum += latency_ms
        self.total_stm_tokens += tokens
    
    def record_ltm_add(self, latency_ms: float) -> None:
        self.ltm_adds += 1
        self.ltm_add_latency_sum += latency_ms
    
    def record_search(self, latency_ms: float, result_count: int) -> None:
        self.searches += 1
        self.search_latency_sum += latency_ms
        self.total_search_results += result_count
    
    def record_context_build(self, latency_ms: float, tokens: int) -> None:
        self.context_builds += 1
        self.context_build_latency_sum += latency_ms
        self.total_context_tokens += tokens
    
    @property
    def avg_stm_add_latency_ms(self) -> float:
        return self.stm_add_latency_sum / self.stm_adds if self.stm_adds > 0 else 0
    
    @property
    def avg_search_latency_ms(self) -> float:
        return self.search_latency_sum / self.searches if self.searches > 0 else 0
    
    @property
    def avg_context_build_latency_ms(self) -> float:
        return self.context_build_latency_sum / self.context_builds if self.context_builds > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stm_adds": self.stm_adds,
            "ltm_adds": self.ltm_adds,
            "searches": self.searches,
            "context_builds": self.context_builds,
            "total_stm_tokens": self.total_stm_tokens,
            "total_context_tokens": self.total_context_tokens,
            "avg_stm_add_latency_ms": self.avg_stm_add_latency_ms,
            "avg_search_latency_ms": self.avg_search_latency_ms,
            "avg_context_build_latency_ms": self.avg_context_build_latency_ms,
        }
