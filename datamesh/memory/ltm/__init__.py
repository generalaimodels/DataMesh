"""
Long-Term Memory (LTM) Module

Provides:
- VectorStore: HNSW-based semantic search
- GraphStore: Entity-relationship graph
- EpisodicMemory: Artifact storage
- ConsolidationPipeline: STM → LTM migration
"""

from datamesh.memory.ltm.vector_store import (
    VectorStore,
    VectorStoreConfig,
    MemoryVector,
    SearchResult,
)
from datamesh.memory.ltm.graph_store import (
    GraphStore,
    MemoryNode,
    MemoryEdge,
    NodeType,
    EdgeType,
)
from datamesh.memory.ltm.episodic import (
    EpisodicMemory,
    Episode,
    EpisodeType,
)
from datamesh.memory.ltm.consolidation import (
    ConsolidationPipeline,
    ConsolidationConfig,
    ConsolidationJob,
)

__all__ = [
    "VectorStore",
    "VectorStoreConfig",
    "MemoryVector",
    "SearchResult",
    "GraphStore",
    "MemoryNode",
    "MemoryEdge",
    "NodeType",
    "EdgeType",
    "EpisodicMemory",
    "Episode",
    "EpisodeType",
    "ConsolidationPipeline",
    "ConsolidationConfig",
    "ConsolidationJob",
]
