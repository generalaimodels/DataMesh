"""
Graph Store: Entity-Relationship Memory Graph

Provides:
- Entity-Attribute-Value (EAV) model for flexible schema
- Temporal edges with validity periods
- Cypher-like traversal patterns
- Memory relationship tracking

Design:
    Graph store captures relational memory:
    - Entities mentioned in conversations
    - Relationships between entities
    - Temporal validity of facts
    - User preferences and patterns
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

from datamesh.core.types import Result, Ok, Err, EntityId, Timestamp


# =============================================================================
# NODE AND EDGE TYPES
# =============================================================================
class NodeType(Enum):
    """Types of memory graph nodes."""
    ENTITY = auto()      # Named entity (person, place, thing)
    CONCEPT = auto()     # Abstract concept or topic
    PREFERENCE = auto()  # User preference
    FACT = auto()        # Factual assertion
    INTERACTION = auto() # Conversation reference
    
    @property
    def color(self) -> str:
        """Visualization color hint."""
        return {
            NodeType.ENTITY: "#4CAF50",
            NodeType.CONCEPT: "#2196F3",
            NodeType.PREFERENCE: "#FF9800",
            NodeType.FACT: "#9C27B0",
            NodeType.INTERACTION: "#607D8B",
        }[self]


class EdgeType(Enum):
    """Types of memory graph edges."""
    RELATES_TO = auto()      # General relationship
    MENTIONED_IN = auto()    # Entity in interaction
    HAS_ATTRIBUTE = auto()   # EAV attribute
    PREFERS = auto()         # User preference
    SIMILAR_TO = auto()      # Semantic similarity
    FOLLOWS = auto()         # Temporal sequence
    REFERENCES = auto()      # Cross-reference
    
    @property
    def bidirectional(self) -> bool:
        """Whether edge implies reverse relationship."""
        return self in (EdgeType.RELATES_TO, EdgeType.SIMILAR_TO)


# =============================================================================
# MEMORY NODE
# =============================================================================
@dataclass(slots=True)
class MemoryNode:
    """
    Node in memory graph.
    
    Represents an entity, concept, or fact in long-term memory.
    """
    node_id: UUID = field(default_factory=uuid4)
    entity_id: Optional[EntityId] = None  # Owner entity
    
    # Node content
    node_type: NodeType = NodeType.ENTITY
    label: str = ""          # Display label
    value: str = ""          # Primary value
    
    # Attributes (EAV model)
    attributes: dict[str, Any] = field(default_factory=dict)
    
    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    confidence: float = 1.0  # 0.0-1.0, confidence in this memory
    source_ids: list[str] = field(default_factory=list)  # Source references
    
    @property
    def is_valid(self) -> bool:
        """Check if node is currently valid."""
        now = datetime.now(timezone.utc)
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_to and now > self.valid_to:
            return False
        return True
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": str(self.node_id),
            "entity_id": str(self.entity_id.value) if self.entity_id else None,
            "node_type": self.node_type.name,
            "label": self.label,
            "value": self.value,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# MEMORY EDGE
# =============================================================================
@dataclass(slots=True)
class MemoryEdge:
    """
    Edge in memory graph.
    
    Represents a relationship between two nodes.
    """
    edge_id: UUID = field(default_factory=uuid4)
    source_id: UUID = field(default_factory=uuid4)
    target_id: UUID = field(default_factory=uuid4)
    
    # Edge content
    edge_type: EdgeType = EdgeType.RELATES_TO
    weight: float = 1.0      # Relationship strength
    label: Optional[str] = None
    
    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    properties: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if edge is currently valid."""
        now = datetime.now(timezone.utc)
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_to and now > self.valid_to:
            return False
        return True


# =============================================================================
# GRAPH STORE
# =============================================================================
class GraphStore:
    """
    Entity-relationship graph for long-term memory.
    
    Features:
        - EAV model for flexible schema
        - Temporal edge validity
        - Pattern-based traversal
        - Multi-tenant isolation by entity_id
    
    Usage:
        store = GraphStore()
        
        # Create nodes
        person = MemoryNode(
            entity_id=entity_id,
            node_type=NodeType.ENTITY,
            label="Alice",
            attributes={"role": "colleague"},
        )
        await store.add_node(person)
        
        # Create edge
        edge = MemoryEdge(
            source_id=person.node_id,
            target_id=project.node_id,
            edge_type=EdgeType.RELATES_TO,
            label="works on",
        )
        await store.add_edge(edge)
        
        # Query
        related = await store.get_neighbors(person.node_id, EdgeType.RELATES_TO)
    """
    
    __slots__ = ("_nodes", "_edges", "_lock")
    
    def __init__(self) -> None:
        # node_id -> MemoryNode
        self._nodes: dict[UUID, MemoryNode] = {}
        # edge_id -> MemoryEdge
        self._edges: dict[UUID, MemoryEdge] = {}
        self._lock = asyncio.Lock()
    
    async def add_node(self, node: MemoryNode) -> Result[UUID, str]:
        """Add node to graph."""
        async with self._lock:
            if node.node_id in self._nodes:
                return Err(f"Node {node.node_id} already exists")
            
            self._nodes[node.node_id] = node
            return Ok(node.node_id)
    
    async def get_node(self, node_id: UUID) -> Optional[MemoryNode]:
        """Get node by ID."""
        async with self._lock:
            return self._nodes.get(node_id)
    
    async def update_node(
        self,
        node_id: UUID,
        updates: dict[str, Any],
    ) -> Result[MemoryNode, str]:
        """Update node attributes."""
        async with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return Err(f"Node {node_id} not found")
            
            for key, value in updates.items():
                if hasattr(node, key):
                    setattr(node, key, value)
                else:
                    node.attributes[key] = value
            
            node.updated_at = datetime.now(timezone.utc)
            return Ok(node)
    
    async def delete_node(self, node_id: UUID) -> Result[bool, str]:
        """Delete node and connected edges."""
        async with self._lock:
            if node_id not in self._nodes:
                return Ok(False)
            
            # Remove connected edges
            edges_to_remove = [
                eid for eid, edge in self._edges.items()
                if edge.source_id == node_id or edge.target_id == node_id
            ]
            for eid in edges_to_remove:
                del self._edges[eid]
            
            del self._nodes[node_id]
            return Ok(True)
    
    async def add_edge(self, edge: MemoryEdge) -> Result[UUID, str]:
        """Add edge to graph."""
        async with self._lock:
            if edge.source_id not in self._nodes:
                return Err(f"Source node {edge.source_id} not found")
            if edge.target_id not in self._nodes:
                return Err(f"Target node {edge.target_id} not found")
            
            self._edges[edge.edge_id] = edge
            return Ok(edge.edge_id)
    
    async def get_edge(self, edge_id: UUID) -> Optional[MemoryEdge]:
        """Get edge by ID."""
        async with self._lock:
            return self._edges.get(edge_id)
    
    async def delete_edge(self, edge_id: UUID) -> bool:
        """Delete edge."""
        async with self._lock:
            if edge_id in self._edges:
                del self._edges[edge_id]
                return True
            return False
    
    async def get_neighbors(
        self,
        node_id: UUID,
        edge_type: Optional[EdgeType] = None,
        direction: str = "outgoing",  # outgoing, incoming, both
    ) -> list[tuple[MemoryNode, MemoryEdge]]:
        """
        Get neighboring nodes connected by edges.
        
        Returns list of (node, edge) tuples.
        """
        async with self._lock:
            neighbors: list[tuple[MemoryNode, MemoryEdge]] = []
            
            for edge in self._edges.values():
                if not edge.is_valid:
                    continue
                
                if edge_type and edge.edge_type != edge_type:
                    continue
                
                neighbor_id = None
                
                if direction in ("outgoing", "both"):
                    if edge.source_id == node_id:
                        neighbor_id = edge.target_id
                
                if direction in ("incoming", "both"):
                    if edge.target_id == node_id:
                        neighbor_id = edge.source_id
                
                if neighbor_id:
                    neighbor = self._nodes.get(neighbor_id)
                    if neighbor and neighbor.is_valid:
                        neighbors.append((neighbor, edge))
            
            return neighbors
    
    async def find_path(
        self,
        start_id: UUID,
        end_id: UUID,
        max_depth: int = 5,
    ) -> list[list[UUID]]:
        """
        Find paths between two nodes (BFS).
        
        Returns list of paths (each path is list of node IDs).
        """
        async with self._lock:
            if start_id not in self._nodes or end_id not in self._nodes:
                return []
            
            # BFS
            queue: list[list[UUID]] = [[start_id]]
            visited: set[UUID] = {start_id}
            paths: list[list[UUID]] = []
            
            while queue:
                path = queue.pop(0)
                current = path[-1]
                
                if len(path) > max_depth:
                    continue
                
                if current == end_id:
                    paths.append(path)
                    continue
                
                # Get neighbors
                for edge in self._edges.values():
                    if edge.source_id == current and edge.target_id not in visited:
                        visited.add(edge.target_id)
                        queue.append(path + [edge.target_id])
                    elif edge.target_id == current and edge.source_id not in visited:
                        visited.add(edge.source_id)
                        queue.append(path + [edge.source_id])
            
            return paths
    
    async def query_by_label(
        self,
        label_pattern: str,
        entity_id: Optional[EntityId] = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        """Query nodes by label pattern."""
        async with self._lock:
            matches: list[MemoryNode] = []
            pattern_lower = label_pattern.lower()
            
            for node in self._nodes.values():
                if not node.is_valid:
                    continue
                
                if entity_id and (
                    node.entity_id is None
                    or node.entity_id.value != entity_id.value
                ):
                    continue
                
                if pattern_lower in node.label.lower():
                    matches.append(node)
                    if len(matches) >= limit:
                        break
            
            return matches
    
    async def query_by_type(
        self,
        node_type: NodeType,
        entity_id: Optional[EntityId] = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        """Query nodes by type."""
        async with self._lock:
            matches: list[MemoryNode] = []
            
            for node in self._nodes.values():
                if not node.is_valid:
                    continue
                
                if node.node_type != node_type:
                    continue
                
                if entity_id and (
                    node.entity_id is None
                    or node.entity_id.value != entity_id.value
                ):
                    continue
                
                matches.append(node)
                if len(matches) >= limit:
                    break
            
            return matches
    
    @property
    def node_count(self) -> int:
        """Total node count."""
        return len(self._nodes)
    
    @property
    def edge_count(self) -> int:
        """Total edge count."""
        return len(self._edges)
