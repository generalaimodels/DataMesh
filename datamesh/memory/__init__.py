"""
Memory Module: Hierarchical Cognitive Storage

Provides:
- Short-Term Memory (STM): Sliding window context buffer
- Long-Term Memory (LTM): Persistent semantic storage
- Memory Hierarchy: Unified retrieval interface

Architecture:
    STM ←→ LTM Consolidation Pipeline
    
    STM (Redis-like):
        - Sliding window buffer (last N turns)
        - Token-aware context management
        - LRU eviction with consolidation triggers
    
    LTM (Vector + Graph):
        - HNSW vector index for semantic search
        - Graph store for relational memory
        - Episodic memory for artifact storage
"""

from datamesh.memory.stm import (
    WorkingSet,
    WorkingSetConfig,
    ContextWindow,
    ContextWindowConfig,
    TruncationStrategy,
    EvictionPolicy,
    EvictionEvent,
)
from datamesh.memory.ltm import (
    VectorStore,
    VectorStoreConfig,
    MemoryVector,
    SearchResult,
    GraphStore,
    MemoryNode,
    MemoryEdge,
    EpisodicMemory,
    Episode,
    ConsolidationPipeline,
    ConsolidationConfig,
)
from datamesh.memory.hierarchy import (
    MemoryHierarchy,
    MemoryQuery,
    MemoryRecall,
    HierarchyConfig,
)

__all__ = [
    # STM
    "WorkingSet",
    "WorkingSetConfig",
    "ContextWindow",
    "ContextWindowConfig",
    "TruncationStrategy",
    "EvictionPolicy",
    "EvictionEvent",
    # LTM
    "VectorStore",
    "VectorStoreConfig",
    "MemoryVector",
    "SearchResult",
    "GraphStore",
    "MemoryNode",
    "MemoryEdge",
    "EpisodicMemory",
    "Episode",
    "ConsolidationPipeline",
    "ConsolidationConfig",
    # Hierarchy
    "MemoryHierarchy",
    "MemoryQuery",
    "MemoryRecall",
    "HierarchyConfig",
]
