"""
Memory Hierarchy: Unified Retrieval Interface

Provides:
- Unified query interface across STM, LTM stores
- Relevance-weighted result fusion
- Multi-modal recall (semantic, temporal, relational)
- Adaptive retrieval strategies

Design:
    Memory hierarchy provides a single API for memory retrieval
    that automatically queries all stores and fuses results
    based on relevance and recency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional, Sequence
from uuid import UUID

from datamesh.core.types import Result, Ok, Err, EntityId, Timestamp


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class HierarchyConfig:
    """Configuration for memory hierarchy."""
    stm_weight: float = 0.4         # Weight for short-term memory
    semantic_weight: float = 0.3    # Weight for vector similarity
    relational_weight: float = 0.2  # Weight for graph relationships
    episodic_weight: float = 0.1    # Weight for episode relevance
    max_results: int = 20
    min_relevance: float = 0.3


# =============================================================================
# MEMORY QUERY
# =============================================================================
class QueryType(Enum):
    """Type of memory query."""
    SEMANTIC = auto()    # Vector similarity search
    TEMPORAL = auto()    # Time-range query
    RELATIONAL = auto()  # Graph traversal
    KEYWORD = auto()     # Text search
    HYBRID = auto()      # Combination of above


@dataclass(slots=True)
class MemoryQuery:
    """
    Query for memory retrieval.
    
    Supports multiple query types and fusion strategies.
    """
    entity_id: EntityId
    query_type: QueryType = QueryType.HYBRID
    
    # Query content
    query_text: Optional[str] = None
    query_embedding: Optional[tuple[float, ...]] = None
    
    # Temporal filters
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Relational filters
    related_to: Optional[UUID] = None
    relationship_type: Optional[str] = None
    
    # Result configuration
    max_results: int = 20
    min_relevance: float = 0.3
    include_stm: bool = True
    include_ltm: bool = True
    
    # Metadata
    metadata_filters: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MEMORY RECALL
# =============================================================================
@dataclass(slots=True)
class MemoryItem:
    """Single memory item from recall."""
    item_id: str
    source: str  # stm, vector, graph, episodic
    content: str
    relevance: float  # 0.0-1.0
    
    # Context
    created_at: Optional[datetime] = None
    token_count: int = 0
    
    # Source-specific data
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryRecall:
    """
    Result of memory retrieval.
    
    Contains fused results from multiple stores.
    """
    query: MemoryQuery
    items: list[MemoryItem] = field(default_factory=list)
    
    # Statistics
    stm_count: int = 0
    vector_count: int = 0
    graph_count: int = 0
    episodic_count: int = 0
    
    # Timing
    latency_ms: float = 0.0
    
    @property
    def total_count(self) -> int:
        """Total items retrieved."""
        return len(self.items)
    
    @property
    def avg_relevance(self) -> float:
        """Average relevance score."""
        if not self.items:
            return 0.0
        return sum(i.relevance for i in self.items) / len(self.items)
    
    def get_context_string(self, max_tokens: int = 4000) -> str:
        """Build context string within token budget."""
        context_parts = []
        total_tokens = 0
        
        for item in self.items:
            if total_tokens + item.token_count > max_tokens:
                break
            context_parts.append(item.content)
            total_tokens += item.token_count
        
        return "\n\n".join(context_parts)


# =============================================================================
# MEMORY HIERARCHY
# =============================================================================
class MemoryHierarchy:
    """
    Unified memory retrieval interface.
    
    Features:
        - Multi-store query execution
        - Relevance-weighted result fusion
        - Adaptive retrieval strategies
        - Context assembly for prompts
    
    Usage:
        hierarchy = MemoryHierarchy(
            working_set=working_set,
            vector_store=vector_store,
            graph_store=graph_store,
            episodic_memory=episodic_memory,
        )
        
        # Query memory
        query = MemoryQuery(
            entity_id=entity_id,
            query_text="What did we discuss about the project?",
            query_type=QueryType.HYBRID,
        )
        recall = await hierarchy.recall(query)
        
        # Build context
        context = recall.get_context_string(max_tokens=4000)
    """
    
    __slots__ = (
        "_config", "_working_set", "_vector_store",
        "_graph_store", "_episodic_memory", "_embedding_fn",
    )
    
    def __init__(
        self,
        working_set: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        graph_store: Optional[Any] = None,
        episodic_memory: Optional[Any] = None,
        config: Optional[HierarchyConfig] = None,
        embedding_fn: Optional[Any] = None,
    ) -> None:
        self._config = config or HierarchyConfig()
        self._working_set = working_set
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._episodic_memory = episodic_memory
        self._embedding_fn = embedding_fn
    
    async def recall(self, query: MemoryQuery) -> MemoryRecall:
        """
        Execute memory recall query.
        
        Queries all applicable stores and fuses results.
        """
        import time
        start_time = time.time()
        
        recall = MemoryRecall(query=query)
        all_items: list[MemoryItem] = []
        
        # Query STM (working set)
        if query.include_stm and self._working_set:
            stm_items = await self._query_stm(query)
            for item in stm_items:
                item.relevance *= self._config.stm_weight
            all_items.extend(stm_items)
            recall.stm_count = len(stm_items)
        
        # Query LTM stores
        if query.include_ltm:
            # Vector store (semantic)
            if self._vector_store and query.query_type in (
                QueryType.SEMANTIC, QueryType.HYBRID
            ):
                vector_items = await self._query_vectors(query)
                for item in vector_items:
                    item.relevance *= self._config.semantic_weight
                all_items.extend(vector_items)
                recall.vector_count = len(vector_items)
            
            # Graph store (relational)
            if self._graph_store and query.query_type in (
                QueryType.RELATIONAL, QueryType.HYBRID
            ):
                graph_items = await self._query_graph(query)
                for item in graph_items:
                    item.relevance *= self._config.relational_weight
                all_items.extend(graph_items)
                recall.graph_count = len(graph_items)
            
            # Episodic memory
            if self._episodic_memory:
                episodic_items = await self._query_episodes(query)
                for item in episodic_items:
                    item.relevance *= self._config.episodic_weight
                all_items.extend(episodic_items)
                recall.episodic_count = len(episodic_items)
        
        # Filter by minimum relevance
        all_items = [
            i for i in all_items
            if i.relevance >= query.min_relevance * self._config.stm_weight
        ]
        
        # Sort by relevance descending
        all_items.sort(key=lambda x: x.relevance, reverse=True)
        
        # Deduplicate by content hash
        seen_content: set[str] = set()
        unique_items: list[MemoryItem] = []
        for item in all_items:
            content_key = item.content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_items.append(item)
        
        # Limit results
        recall.items = unique_items[:query.max_results]
        recall.latency_ms = (time.time() - start_time) * 1000
        
        return recall
    
    async def _query_stm(self, query: MemoryQuery) -> list[MemoryItem]:
        """Query short-term memory."""
        items: list[MemoryItem] = []
        
        if self._working_set is None:
            return items
        
        try:
            recent = await self._working_set.get_recent(limit=query.max_results)
            
            for ws_item in recent:
                content = ws_item.decompress().decode("utf-8", errors="replace")
                
                # Calculate relevance based on recency
                relevance = 1.0  # STM is always relevant
                if query.query_text:
                    # Simple text match scoring
                    query_lower = query.query_text.lower()
                    if query_lower in content.lower():
                        relevance = 1.0
                    else:
                        relevance = 0.5
                
                items.append(MemoryItem(
                    item_id=ws_item.item_id,
                    source="stm",
                    content=content,
                    relevance=relevance,
                    created_at=ws_item.created_at,
                    token_count=ws_item.token_count,
                    metadata={"role": ws_item.role},
                ))
        except Exception:
            pass
        
        return items
    
    async def _query_vectors(self, query: MemoryQuery) -> list[MemoryItem]:
        """Query vector store for semantic similarity."""
        items: list[MemoryItem] = []
        
        if self._vector_store is None:
            return items
        
        try:
            # Generate embedding for query
            if query.query_embedding:
                embedding = query.query_embedding
            elif query.query_text and self._embedding_fn:
                embedding = self._embedding_fn(query.query_text)
            else:
                return items
            
            results = await self._vector_store.search(
                entity_id=query.entity_id,
                query_embedding=embedding,
                k=query.max_results,
            )
            
            for result in results:
                items.append(MemoryItem(
                    item_id=str(result.vector.vector_id),
                    source="vector",
                    content=result.vector.content,
                    relevance=result.similarity,
                    token_count=result.vector.token_count,
                    metadata={"rank": result.rank},
                ))
        except Exception:
            pass
        
        return items
    
    async def _query_graph(self, query: MemoryQuery) -> list[MemoryItem]:
        """Query graph store for relational memory."""
        items: list[MemoryItem] = []
        
        if self._graph_store is None:
            return items
        
        try:
            if query.query_text:
                nodes = await self._graph_store.query_by_label(
                    label_pattern=query.query_text,
                    entity_id=query.entity_id,
                    limit=query.max_results,
                )
                
                for node in nodes:
                    items.append(MemoryItem(
                        item_id=str(node.node_id),
                        source="graph",
                        content=f"{node.label}: {node.value}",
                        relevance=node.confidence,
                        created_at=node.created_at,
                        metadata={
                            "node_type": node.node_type.name,
                            "attributes": node.attributes,
                        },
                    ))
        except Exception:
            pass
        
        return items
    
    async def _query_episodes(self, query: MemoryQuery) -> list[MemoryItem]:
        """Query episodic memory."""
        items: list[MemoryItem] = []
        
        if self._episodic_memory is None:
            return items
        
        try:
            if query.query_text:
                episodes = await self._episodic_memory.search(
                    entity_id=query.entity_id,
                    query=query.query_text,
                    limit=query.max_results,
                )
            else:
                episodes = await self._episodic_memory.get_recent(
                    entity_id=query.entity_id,
                    limit=query.max_results,
                )
            
            for episode in episodes:
                items.append(MemoryItem(
                    item_id=str(episode.episode_id),
                    source="episodic",
                    content=f"{episode.title}\n{episode.summary}",
                    relevance=episode.importance,
                    created_at=episode.started_at,
                    token_count=episode.token_count,
                    metadata={
                        "episode_type": episode.episode_type.name,
                        "turn_count": episode.turn_count,
                        "tags": episode.tags,
                    },
                ))
        except Exception:
            pass
        
        return items
    
    async def get_context(
        self,
        entity_id: EntityId,
        query_text: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> str:
        """
        Convenience method to get context string.
        
        Combines STM and relevant LTM into prompt context.
        """
        query = MemoryQuery(
            entity_id=entity_id,
            query_text=query_text,
            query_type=QueryType.HYBRID if query_text else QueryType.TEMPORAL,
            max_results=50,
        )
        
        recall = await self.recall(query)
        return recall.get_context_string(max_tokens=max_tokens)
