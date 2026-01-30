"""
Episodic Memory: Artifact and Event Storage

Provides:
- Episode storage for complex interactions
- Artifact references (files, images, code)
- Temporal episode chains
- Episode search and retrieval

Design:
    Episodic memory stores rich interaction sequences:
    - Multi-turn conversation episodes
    - Tool use sequences
    - Artifact generation events
    - Referenced as chapters in entity history
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
# EPISODE TYPES
# =============================================================================
class EpisodeType(Enum):
    """Types of episodic memories."""
    CONVERSATION = auto()   # Multi-turn conversation
    TASK = auto()           # Task execution sequence
    TOOL_USE = auto()       # Tool invocation episode
    ARTIFACT = auto()       # Artifact generation
    LEARNING = auto()       # User correction/feedback
    MILESTONE = auto()      # Significant event
    
    @property
    def retention_priority(self) -> int:
        """Priority for retention (higher = keep longer)."""
        return {
            EpisodeType.CONVERSATION: 3,
            EpisodeType.TASK: 5,
            EpisodeType.TOOL_USE: 2,
            EpisodeType.ARTIFACT: 7,
            EpisodeType.LEARNING: 10,
            EpisodeType.MILESTONE: 10,
        }[self]


# =============================================================================
# ARTIFACT REFERENCE
# =============================================================================
@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference to stored artifact."""
    artifact_id: str
    artifact_type: str  # file, image, code, audio, video
    storage_path: str   # S3 or filesystem path
    mime_type: str
    size_bytes: int
    checksum: Optional[str] = None


# =============================================================================
# EPISODE
# =============================================================================
@dataclass(slots=True)
class Episode:
    """
    Episodic memory entry.
    
    Stores a coherent sequence of interactions with metadata.
    """
    episode_id: UUID = field(default_factory=uuid4)
    entity_id: Optional[EntityId] = None
    
    # Episode classification
    episode_type: EpisodeType = EpisodeType.CONVERSATION
    title: str = ""
    summary: str = ""
    
    # Content
    turns: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    
    # Temporal bounds
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Statistics
    turn_count: int = 0
    token_count: int = 0
    tool_calls: int = 0
    
    # Linking
    parent_episode_id: Optional[UUID] = None
    related_episode_ids: list[UUID] = field(default_factory=list)
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0-1.0
    consolidation_status: str = "pending"  # pending, processing, complete
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if episode is complete."""
        return self.ended_at is not None
    
    @property
    def artifact_count(self) -> int:
        """Number of artifacts in episode."""
        return len(self.artifacts)
    
    def add_turn(
        self,
        role: str,
        content: str,
        token_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add turn to episode."""
        self.turns.append({
            "role": role,
            "content": content,
            "token_count": token_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        })
        self.turn_count += 1
        self.token_count += token_count
        
        if role == "tool":
            self.tool_calls += 1
    
    def add_artifact(self, artifact: ArtifactRef) -> None:
        """Add artifact reference to episode."""
        self.artifacts.append(artifact)
    
    def complete(self, summary: Optional[str] = None) -> None:
        """Mark episode as complete."""
        self.ended_at = datetime.now(timezone.utc)
        self.duration_seconds = (
            self.ended_at - self.started_at
        ).total_seconds()
        if summary:
            self.summary = summary
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "episode_id": str(self.episode_id),
            "entity_id": str(self.entity_id.value) if self.entity_id else None,
            "episode_type": self.episode_type.name,
            "title": self.title,
            "summary": self.summary,
            "turn_count": self.turn_count,
            "token_count": self.token_count,
            "artifact_count": self.artifact_count,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "importance": self.importance,
            "tags": self.tags,
        }


# =============================================================================
# EPISODIC MEMORY STORE
# =============================================================================
class EpisodicMemory:
    """
    Episodic memory store.
    
    Features:
        - Episode CRUD operations
        - Temporal queries
        - Artifact management
        - Episode chaining
    
    Usage:
        store = EpisodicMemory()
        
        # Create episode
        episode = Episode(
            entity_id=entity_id,
            episode_type=EpisodeType.TASK,
            title="Code Review Task",
        )
        await store.create(episode)
        
        # Add turns
        episode.add_turn("user", "Review this code", token_count=10)
        episode.add_turn("assistant", "The code looks good", token_count=50)
        await store.update(episode)
        
        # Query recent episodes
        recent = await store.get_recent(entity_id, limit=10)
    """
    
    __slots__ = ("_episodes", "_lock")
    
    def __init__(self) -> None:
        self._episodes: dict[UUID, Episode] = {}
        self._lock = asyncio.Lock()
    
    async def create(self, episode: Episode) -> Result[UUID, str]:
        """Create new episode."""
        async with self._lock:
            if episode.episode_id in self._episodes:
                return Err(f"Episode {episode.episode_id} already exists")
            
            self._episodes[episode.episode_id] = episode
            return Ok(episode.episode_id)
    
    async def get(self, episode_id: UUID) -> Optional[Episode]:
        """Get episode by ID."""
        async with self._lock:
            return self._episodes.get(episode_id)
    
    async def update(self, episode: Episode) -> Result[None, str]:
        """Update existing episode."""
        async with self._lock:
            if episode.episode_id not in self._episodes:
                return Err(f"Episode {episode.episode_id} not found")
            
            self._episodes[episode.episode_id] = episode
            return Ok(None)
    
    async def delete(self, episode_id: UUID) -> bool:
        """Delete episode."""
        async with self._lock:
            if episode_id in self._episodes:
                del self._episodes[episode_id]
                return True
            return False
    
    async def get_recent(
        self,
        entity_id: EntityId,
        limit: int = 10,
        episode_type: Optional[EpisodeType] = None,
    ) -> list[Episode]:
        """Get recent episodes for entity."""
        async with self._lock:
            episodes = [
                e for e in self._episodes.values()
                if e.entity_id and e.entity_id.value == entity_id.value
            ]
            
            if episode_type:
                episodes = [e for e in episodes if e.episode_type == episode_type]
            
            # Sort by started_at descending
            episodes.sort(key=lambda e: e.started_at, reverse=True)
            
            return episodes[:limit]
    
    async def get_by_tag(
        self,
        entity_id: EntityId,
        tag: str,
        limit: int = 100,
    ) -> list[Episode]:
        """Get episodes by tag."""
        async with self._lock:
            episodes = [
                e for e in self._episodes.values()
                if e.entity_id
                and e.entity_id.value == entity_id.value
                and tag in e.tags
            ]
            
            episodes.sort(key=lambda e: e.started_at, reverse=True)
            return episodes[:limit]
    
    async def get_time_range(
        self,
        entity_id: EntityId,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Episode]:
        """Get episodes in time range."""
        async with self._lock:
            episodes = [
                e for e in self._episodes.values()
                if e.entity_id
                and e.entity_id.value == entity_id.value
                and start_time <= e.started_at <= end_time
            ]
            
            episodes.sort(key=lambda e: e.started_at)
            return episodes
    
    async def get_chain(
        self,
        episode_id: UUID,
    ) -> list[Episode]:
        """Get episode chain (parent and related)."""
        async with self._lock:
            episode = self._episodes.get(episode_id)
            if not episode:
                return []
            
            chain: list[Episode] = [episode]
            
            # Get parent chain
            current = episode
            while current.parent_episode_id:
                parent = self._episodes.get(current.parent_episode_id)
                if parent:
                    chain.insert(0, parent)
                    current = parent
                else:
                    break
            
            # Get related
            for related_id in episode.related_episode_ids:
                related = self._episodes.get(related_id)
                if related and related not in chain:
                    chain.append(related)
            
            return chain
    
    async def search(
        self,
        entity_id: EntityId,
        query: str,
        limit: int = 20,
    ) -> list[Episode]:
        """Search episodes by title/summary."""
        async with self._lock:
            query_lower = query.lower()
            matches: list[tuple[float, Episode]] = []
            
            for episode in self._episodes.values():
                if not episode.entity_id:
                    continue
                if episode.entity_id.value != entity_id.value:
                    continue
                
                score = 0.0
                if query_lower in episode.title.lower():
                    score += 1.0
                if query_lower in episode.summary.lower():
                    score += 0.5
                for tag in episode.tags:
                    if query_lower in tag.lower():
                        score += 0.3
                
                if score > 0:
                    matches.append((score, episode))
            
            matches.sort(key=lambda x: (-x[0], x[1].started_at), reverse=True)
            return [e for _, e in matches[:limit]]
    
    async def get_with_artifacts(
        self,
        entity_id: EntityId,
        artifact_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[Episode]:
        """Get episodes containing artifacts."""
        async with self._lock:
            episodes = [
                e for e in self._episodes.values()
                if e.entity_id
                and e.entity_id.value == entity_id.value
                and e.artifact_count > 0
            ]
            
            if artifact_type:
                episodes = [
                    e for e in episodes
                    if any(a.artifact_type == artifact_type for a in e.artifacts)
                ]
            
            episodes.sort(key=lambda e: e.started_at, reverse=True)
            return episodes[:limit]
    
    @property
    def total_episodes(self) -> int:
        """Total episode count."""
        return len(self._episodes)
