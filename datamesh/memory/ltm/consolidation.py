"""
Memory Consolidation Pipeline: STM → LTM Migration

Provides:
- Event-driven consolidation from STM eviction
- NLP extraction for entities and relationships
- Embedding generation for semantic storage
- Dual-write coordination to Vector/Graph stores

Pipeline Stages:
    1. Receive eviction events from STM
    2. Extract entities and facts (NER, relation extraction)
    3. Generate embeddings for semantic search
    4. Write to Vector store and Graph store
    5. Create episodic memory entries
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from datamesh.core.types import Result, Ok, Err, EntityId, Timestamp
from datamesh.memory.stm.eviction import EvictionEvent
from datamesh.memory.ltm.vector_store import VectorStore, MemoryVector
from datamesh.memory.ltm.graph_store import GraphStore, MemoryNode, MemoryEdge, NodeType
from datamesh.memory.ltm.episodic import EpisodicMemory, Episode, EpisodeType


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class ConsolidationConfig:
    """Configuration for consolidation pipeline."""
    max_concurrent_jobs: int = 4
    batch_size: int = 10
    embedding_dimension: int = 1536
    extract_entities: bool = True
    create_episodes: bool = True
    min_tokens_for_embedding: int = 20


# =============================================================================
# CONSOLIDATION JOB
# =============================================================================
class JobStatus(Enum):
    """Consolidation job status."""
    PENDING = auto()
    EXTRACTING = auto()
    EMBEDDING = auto()
    WRITING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class ConsolidationJob:
    """
    Single consolidation job.
    
    Processes one eviction event through the pipeline.
    """
    job_id: UUID = field(default_factory=uuid4)
    event: Optional[EvictionEvent] = None
    
    # Status
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Results
    vectors_created: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    episode_id: Optional[UUID] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Job duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


# =============================================================================
# CONSOLIDATION PIPELINE
# =============================================================================
class ConsolidationPipeline:
    """
    STM to LTM memory consolidation pipeline.
    
    Features:
        - Event-driven processing from STM eviction
        - Entity extraction and embedding generation
        - Dual-write to Vector and Graph stores
        - Episode creation for complex interactions
    
    Usage:
        pipeline = ConsolidationPipeline(
            vector_store=vector_store,
            graph_store=graph_store,
            episodic_memory=episodic_memory,
        )
        
        # Connect to eviction events
        eviction_manager.on_eviction(pipeline.enqueue)
        
        # Start processing
        await pipeline.start()
        
        # Process events
        await pipeline.process_pending()
    """
    
    __slots__ = (
        "_config", "_vector_store", "_graph_store",
        "_episodic_memory", "_queue", "_active_jobs",
        "_completed_jobs", "_lock", "_embedding_fn",
        "_running",
    )
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        graph_store: Optional[GraphStore] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        config: Optional[ConsolidationConfig] = None,
        embedding_fn: Optional[Callable[[str], tuple[float, ...]]] = None,
    ) -> None:
        self._config = config or ConsolidationConfig()
        self._vector_store = vector_store or VectorStore()
        self._graph_store = graph_store or GraphStore()
        self._episodic_memory = episodic_memory or EpisodicMemory()
        self._queue: list[ConsolidationJob] = []
        self._active_jobs: dict[UUID, ConsolidationJob] = {}
        self._completed_jobs: list[ConsolidationJob] = []
        self._lock = asyncio.Lock()
        self._embedding_fn = embedding_fn or self._default_embedding
        self._running = False
    
    @staticmethod
    def _default_embedding(text: str) -> tuple[float, ...]:
        """Default embedding (placeholder using hash)."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Create pseudo-embedding from hash bytes
        return tuple(b / 255.0 - 0.5 for b in h[:64])
    
    def enqueue(self, event: EvictionEvent) -> UUID:
        """
        Enqueue eviction event for consolidation.
        
        Called by eviction manager when items are evicted from STM.
        """
        job = ConsolidationJob(event=event)
        self._queue.append(job)
        return job.job_id
    
    async def process_pending(self) -> list[ConsolidationJob]:
        """Process pending consolidation jobs."""
        completed: list[ConsolidationJob] = []
        
        async with self._lock:
            available_slots = self._config.max_concurrent_jobs - len(self._active_jobs)
            jobs_to_start = self._queue[:available_slots]
            self._queue = self._queue[available_slots:]
            
            for job in jobs_to_start:
                self._active_jobs[job.job_id] = job
        
        # Process jobs outside lock
        tasks = [self._process_job(job) for job in jobs_to_start]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
            async with self._lock:
                for job in jobs_to_start:
                    self._active_jobs.pop(job.job_id, None)
                    self._completed_jobs.append(job)
                    completed.append(job)
        
        return completed
    
    async def _process_job(self, job: ConsolidationJob) -> None:
        """Process single consolidation job."""
        job.status = JobStatus.EXTRACTING
        job.started_at = datetime.now(timezone.utc)
        
        try:
            if job.event is None:
                raise ValueError("Job has no eviction event")
            
            event = job.event
            entity_id = event.entity_id
            
            # Stage 1: Extract content from evicted items
            # In production, would retrieve actual content
            content = f"Evicted {event.item_count} items with {event.total_tokens} tokens"
            
            # Stage 2: Entity extraction (simplified)
            if self._config.extract_entities:
                job.status = JobStatus.EXTRACTING
                extracted_entities = await self._extract_entities(content)
                
                # Create graph nodes
                for entity in extracted_entities:
                    node = MemoryNode(
                        entity_id=entity_id,
                        node_type=NodeType.ENTITY,
                        label=entity,
                        value=entity,
                    )
                    result = await self._graph_store.add_node(node)
                    if result.is_ok():
                        job.nodes_created += 1
            
            # Stage 3: Generate embeddings
            if event.total_tokens >= self._config.min_tokens_for_embedding:
                job.status = JobStatus.EMBEDDING
                embedding = self._embedding_fn(content)
                
                vector = MemoryVector(
                    entity_id=entity_id,
                    embedding=embedding,
                    content=content[:500],  # Truncate for storage
                    token_count=event.total_tokens,
                    source_type="eviction",
                )
                result = await self._vector_store.insert(vector)
                if result.is_ok():
                    job.vectors_created += 1
            
            # Stage 4: Create episode
            if self._config.create_episodes:
                job.status = JobStatus.WRITING
                episode = Episode(
                    entity_id=entity_id,
                    episode_type=EpisodeType.CONVERSATION,
                    title=f"Consolidated memory from {event.item_count} items",
                    summary=content,
                    token_count=event.total_tokens,
                    turn_count=event.item_count,
                )
                episode.complete()
                
                result = await self._episodic_memory.create(episode)
                if result.is_ok():
                    job.episode_id = episode.episode_id
            
            job.status = JobStatus.COMPLETED
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
        
        finally:
            job.completed_at = datetime.now(timezone.utc)
    
    async def _extract_entities(self, content: str) -> list[str]:
        """
        Extract entities from content.
        
        Simplified extraction - production would use NER model.
        """
        # Placeholder: extract capitalized words as potential entities
        words = content.split()
        entities = [
            w for w in words
            if w and w[0].isupper() and len(w) > 2
        ]
        return entities[:10]
    
    async def get_job(self, job_id: UUID) -> Optional[ConsolidationJob]:
        """Get job by ID."""
        async with self._lock:
            if job_id in self._active_jobs:
                return self._active_jobs[job_id]
            
            for job in self._completed_jobs:
                if job.job_id == job_id:
                    return job
            
            for job in self._queue:
                if job.job_id == job_id:
                    return job
            
            return None
    
    async def start(self) -> None:
        """Start background processing loop."""
        self._running = True
        
        while self._running:
            try:
                if self._queue:
                    await self.process_pending()
                await asyncio.sleep(0.1)  # Poll interval
            except asyncio.CancelledError:
                break
    
    async def stop(self) -> None:
        """Stop background processing."""
        self._running = False
    
    @property
    def queue_size(self) -> int:
        """Pending job count."""
        return len(self._queue)
    
    @property
    def active_count(self) -> int:
        """Active job count."""
        return len(self._active_jobs)
    
    @property
    def total_completed(self) -> int:
        """Total completed jobs."""
        return len(self._completed_jobs)
