"""
Session Affinity: Geo-Affinity and Consistent Hashing

Provides session routing and placement:
- Consistent hashing on entity_id for session affinity
- Geo-affinity routing to minimize latency
- Health-based failover with automatic migration
- Load balancing across healthy nodes

Design:
    Layer 7 ingress uses consistent hashing to route requests
    to the same node for a given entity_id. This ensures:
    - Session cache locality
    - Reduced cross-node coordination
    - Predictable routing for debugging

Failover:
    When a node becomes unhealthy:
    1. Health check fails (timeout/error threshold)
    2. Node marked as draining
    3. New sessions route to next ring node
    4. Existing sessions migrate lazily on next access
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Sequence
from uuid import UUID

from datamesh.core.types import Result, Ok, Err, EntityId, Timestamp, GeoRegion
from datamesh.sharding.consistent_hash import ConsistentHashRing, VirtualNode


# =============================================================================
# CONSTANTS
# =============================================================================
VIRTUAL_NODES_PER_NODE: int = 256
HEALTH_CHECK_INTERVAL_MS: int = 5000
UNHEALTHY_THRESHOLD: int = 3  # Consecutive failures before marking unhealthy
DRAIN_TIMEOUT_MS: int = 60000  # Time to drain before removal


# =============================================================================
# NODE HEALTH STATUS
# =============================================================================
class NodeStatus(Enum):
    """Node health status."""
    HEALTHY = auto()      # Accepting requests
    DEGRADED = auto()     # Accepting but slow/error-prone
    DRAINING = auto()     # Not accepting new, serving existing
    UNHEALTHY = auto()    # Not accepting any requests


@dataclass
class NodeHealth:
    """
    Health information for a routing node.
    
    Tracks health check results and manages status transitions.
    """
    node_id: str
    address: str
    region: GeoRegion
    
    # Health tracking
    status: NodeStatus = NodeStatus.HEALTHY
    consecutive_failures: int = 0
    last_success: Optional[Timestamp] = None
    last_failure: Optional[Timestamp] = None
    
    # Metrics
    total_requests: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    
    # Drain state
    drain_started_at: Optional[Timestamp] = None
    
    @property
    def is_available(self) -> bool:
        """Check if node can accept new requests."""
        return self.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED)
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is fully healthy."""
        return self.status == NodeStatus.HEALTHY
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests
    
    def record_success(self, latency_ms: float) -> None:
        """Record successful request."""
        self.last_success = Timestamp.now()
        self.consecutive_failures = 0
        self.total_requests += 1
        
        # Exponential moving average for latency
        alpha = 0.1
        self.avg_latency_ms = (alpha * latency_ms + 
                              (1 - alpha) * self.avg_latency_ms)
        
        # Potentially recover from degraded
        if self.status == NodeStatus.DEGRADED and self.consecutive_failures == 0:
            self.status = NodeStatus.HEALTHY
    
    def record_failure(self) -> None:
        """Record failed request."""
        self.last_failure = Timestamp.now()
        self.consecutive_failures += 1
        self.total_requests += 1
        self.total_failures += 1
        
        # Status transitions
        if self.consecutive_failures >= UNHEALTHY_THRESHOLD:
            if self.status == NodeStatus.HEALTHY:
                self.status = NodeStatus.DEGRADED
            elif self.status == NodeStatus.DEGRADED:
                self.status = NodeStatus.UNHEALTHY
    
    def start_drain(self) -> None:
        """Start draining the node."""
        self.status = NodeStatus.DRAINING
        self.drain_started_at = Timestamp.now()
    
    def mark_unhealthy(self) -> None:
        """Force node to unhealthy status."""
        self.status = NodeStatus.UNHEALTHY


# =============================================================================
# AFFINITY CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class AffinityConfig:
    """Configuration for session affinity."""
    vnodes_per_node: int = VIRTUAL_NODES_PER_NODE
    health_check_interval_ms: int = HEALTH_CHECK_INTERVAL_MS
    unhealthy_threshold: int = UNHEALTHY_THRESHOLD
    prefer_same_region: bool = True
    allow_cross_region: bool = True  # Fallback to other regions if same unavailable


# =============================================================================
# SESSION AFFINITY MANAGER
# =============================================================================
class SessionAffinity:
    """
    Session affinity and routing manager.
    
    Combines consistent hashing with health-aware routing:
    - Primary routing via consistent hash on entity_id
    - Geo-affinity preference for co-located requests
    - Automatic failover to healthy backup nodes
    - Session migration support for node removal
    
    Usage:
        affinity = SessionAffinity()
        
        # Add nodes
        affinity.add_node("node-1", "10.0.0.1:8080", GeoRegion.US_EAST)
        affinity.add_node("node-2", "10.0.0.2:8080", GeoRegion.US_WEST)
        
        # Route session
        result = affinity.route(entity_id, preferred_region=GeoRegion.US_WEST)
        if result.is_ok():
            node = result.unwrap()
            forward_request_to(node.address)
    """
    
    __slots__ = (
        "_ring", "_nodes", "_config", "_lock",
        "_health_check_task",
    )
    
    def __init__(self, config: Optional[AffinityConfig] = None) -> None:
        self._config = config or AffinityConfig()
        self._ring: ConsistentHashRing = ConsistentHashRing(
            vnodes_per_node=self._config.vnodes_per_node
        )
        self._nodes: dict[str, NodeHealth] = {}
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task[None]] = None
    
    async def add_node(
        self,
        node_id: str,
        address: str,
        region: GeoRegion,
    ) -> Result[None, str]:
        """Add node to routing ring."""
        async with self._lock:
            if node_id in self._nodes:
                return Err(f"Node {node_id} already exists")
            
            # Parse address to get host:port
            if ":" in address:
                host, port_str = address.rsplit(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    host = address
                    port = 8080
            else:
                host = address
                port = 8080
            
            # Add to hash ring with PhysicalNode
            from datamesh.sharding.consistent_hash import PhysicalNode
            physical_node = PhysicalNode(node_id=node_id, host=host, port=port)
            self._ring.add_node(physical_node)
            
            # Create health tracking
            self._nodes[node_id] = NodeHealth(
                node_id=node_id,
                address=address,
                region=region,
            )
            
            return Ok(None)
    
    async def remove_node(
        self,
        node_id: str,
        graceful: bool = True,
    ) -> Result[None, str]:
        """Remove node from routing ring."""
        async with self._lock:
            if node_id not in self._nodes:
                return Err(f"Node {node_id} not found")
            
            node = self._nodes[node_id]
            
            if graceful and node.status != NodeStatus.DRAINING:
                # Start drain instead of immediate removal
                node.start_drain()
                return Ok(None)
            
            # Remove from ring and tracking
            self._ring.remove_node(node_id)
            del self._nodes[node_id]
            
            return Ok(None)
    
    async def route(
        self,
        entity_id: EntityId,
        preferred_region: Optional[GeoRegion] = None,
    ) -> Result[NodeHealth, str]:
        """
        Route entity to appropriate node.
        
        Strategy:
        1. Get primary node from consistent hash
        2. If healthy and region matches, use it
        3. If unhealthy or wrong region, find backup in preferred region
        4. Fall back to any healthy node
        """
        async with self._lock:
            if not self._nodes:
                return Err("No nodes available")
            
            # Get primary node from hash ring
            key = entity_id.value.bytes
            primary_id = self._ring.get_node(key)
            
            if primary_id is None:
                return Err("Hash ring is empty")
            
            primary = self._nodes.get(primary_id)
            
            # Check primary health and region
            if primary and primary.is_available:
                if not self._config.prefer_same_region:
                    return Ok(primary)
                if preferred_region is None or primary.region == preferred_region:
                    return Ok(primary)
            
            # Find backup nodes
            backups = self._ring.get_replicas(key, count=3)
            
            # Try backup in preferred region
            if preferred_region and self._config.prefer_same_region:
                for backup_id in backups:
                    backup = self._nodes.get(backup_id)
                    if backup and backup.is_available and backup.region == preferred_region:
                        return Ok(backup)
            
            # Fall back to any healthy backup
            if self._config.allow_cross_region:
                for backup_id in backups:
                    backup = self._nodes.get(backup_id)
                    if backup and backup.is_available:
                        return Ok(backup)
            
            # Last resort: any healthy node
            for node in self._nodes.values():
                if node.is_available:
                    return Ok(node)
            
            return Err("No healthy nodes available")
    
    async def record_request(
        self,
        node_id: str,
        success: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """Record request result for health tracking."""
        async with self._lock:
            node = self._nodes.get(node_id)
            if node:
                if success:
                    node.record_success(latency_ms)
                else:
                    node.record_failure()
    
    async def get_migration_targets(
        self,
        source_node_id: str,
    ) -> list[tuple[EntityId, str]]:
        """
        Get session migration targets when node is draining.
        
        Returns list of (entity_id, target_node_id) pairs.
        This is a placeholder - actual implementation would query
        active sessions on the source node.
        """
        # In production, this would query the session registry
        # for sessions on the source node and compute new targets
        return []
    
    async def start_health_checks(
        self,
        check_fn: Optional[Any] = None,
    ) -> None:
        """Start background health check task."""
        if self._health_check_task:
            return
        
        async def health_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(
                        self._config.health_check_interval_ms / 1000
                    )
                    # In production, would ping each node
                    # and update health status
                except asyncio.CancelledError:
                    break
        
        self._health_check_task = asyncio.create_task(health_loop())
    
    async def stop_health_checks(self) -> None:
        """Stop background health check task."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
    
    def get_node(self, node_id: str) -> Optional[NodeHealth]:
        """Get node health information."""
        return self._nodes.get(node_id)
    
    def list_nodes(
        self,
        status_filter: Optional[NodeStatus] = None,
        region_filter: Optional[GeoRegion] = None,
    ) -> list[NodeHealth]:
        """List nodes with optional filtering."""
        nodes = list(self._nodes.values())
        
        if status_filter:
            nodes = [n for n in nodes if n.status == status_filter]
        if region_filter:
            nodes = [n for n in nodes if n.region == region_filter]
        
        return nodes
    
    @property
    def healthy_count(self) -> int:
        """Count of healthy nodes."""
        return sum(1 for n in self._nodes.values() if n.is_healthy)
    
    @property
    def total_nodes(self) -> int:
        """Total node count."""
        return len(self._nodes)
