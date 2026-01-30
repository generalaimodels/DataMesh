"""
Comprehensive Test Suite for Database Layer Components

Tests for:
- SessionDatabaseLayer: Session CRUD, state transitions, leases
- HistoryDatabaseLayer: Timeline operations, bucketing, archival
- MemoryDatabaseLayer: STM/LTM, semantic search, context assembly

Run with: python -m datamesh.tests.test_database_layers

Author: Planetary AI Systems
License: MIT
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("\\", 3)[0])

from datamesh.core.types import EntityId
from datamesh.session.database import (
    SessionDatabaseLayer,
    SessionDBConfig,
    SessionMetadata,
    SessionPayload,
    SessionState,
)
from datamesh.history.database import (
    HistoryDatabaseLayer,
    HistoryDBConfig,
    Interaction,
    InteractionType,
    StorageTier,
)
from datamesh.memory.database import (
    MemoryDatabaseLayer,
    MemoryDBConfig,
    MemoryItem,
    MemoryPriority,
    MemoryQuery,
)


# =============================================================================
# TEST UTILITIES
# =============================================================================

def create_test_entity() -> EntityId:
    """Create a test entity ID."""
    return EntityId(f"test-entity-{uuid4().hex[:8]}")


def assert_ok(result, message: str = "Expected Ok result"):
    """Assert that result is Ok."""
    if result.is_err():
        raise AssertionError(f"{message}: {result.error}")
    return result.unwrap()


def assert_err(result, message: str = "Expected Err result"):
    """Assert that result is Err."""
    if result.is_ok():
        raise AssertionError(f"{message}: Got Ok({result.unwrap()})")
    return result.error


class TestResults:
    """Accumulate test results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
    
    def record(self, name: str, passed: bool, error: Optional[str] = None):
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {error}")
            print(f"  ✗ {name}: {error}")
    
    def summary(self) -> str:
        total = self.passed + self.failed
        return f"Passed: {self.passed}/{total}"


# =============================================================================
# SESSION DATABASE LAYER TESTS
# =============================================================================

async def test_session_create(results: TestResults):
    """Test session creation."""
    db = SessionDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        result = await db.create_session(entity_id, ttl_seconds=3600)
        
        if result.is_err():
            results.record("session_create", False, result.error)
            return
        
        metadata, payload = result.unwrap()
        
        assert metadata.entity_id == entity_id
        assert metadata.state == SessionState.INITIATED
        assert metadata.session_id is not None
        assert payload.session_id == metadata.session_id
        
        results.record("session_create", True)
    except Exception as e:
        results.record("session_create", False, str(e))


async def test_session_get(results: TestResults):
    """Test session retrieval."""
    db = SessionDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Create session
        create_result = await db.create_session(entity_id)
        metadata, _ = assert_ok(create_result)
        
        # Get session
        get_result = await db.get_session(entity_id, metadata.session_id)
        
        if get_result.is_err():
            results.record("session_get", False, get_result.error)
            return
        
        got_metadata, got_payload = get_result.unwrap()
        
        assert got_metadata.session_id == metadata.session_id
        assert got_payload is not None
        
        results.record("session_get", True)
    except Exception as e:
        results.record("session_get", False, str(e))


async def test_session_update(results: TestResults):
    """Test session update."""
    db = SessionDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Create session
        create_result = await db.create_session(entity_id)
        metadata, _ = assert_ok(create_result)
        
        # Update session
        update_result = await db.update_session(
            entity_id,
            metadata.session_id,
            metadata_updates={"interaction_count": 5},
        )
        
        if update_result.is_err():
            results.record("session_update", False, update_result.error)
            return
        
        updated_metadata = update_result.unwrap()
        
        assert updated_metadata.interaction_count == 5
        assert updated_metadata.version == 2
        
        results.record("session_update", True)
    except Exception as e:
        results.record("session_update", False, str(e))


async def test_session_state_transition(results: TestResults):
    """Test session state transitions."""
    db = SessionDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Create session
        create_result = await db.create_session(entity_id)
        metadata, _ = assert_ok(create_result)
        
        # Valid transition: INITIATED -> ACTIVE
        result = await db.transition_state(
            entity_id, metadata.session_id, SessionState.ACTIVE
        )
        
        if result.is_err():
            results.record("session_state_transition", False, result.error)
            return
        
        updated = result.unwrap()
        assert updated.state == SessionState.ACTIVE
        
        # Valid transition: ACTIVE -> PAUSED
        result = await db.transition_state(
            entity_id, metadata.session_id, SessionState.PAUSED
        )
        assert result.is_ok()
        
        # Valid transition: PAUSED -> TERMINATED
        result = await db.transition_state(
            entity_id, metadata.session_id, SessionState.TERMINATED
        )
        assert result.is_ok()
        
        results.record("session_state_transition", True)
    except Exception as e:
        results.record("session_state_transition", False, str(e))


async def test_session_invalid_transition(results: TestResults):
    """Test invalid state transition is rejected."""
    db = SessionDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Create session
        create_result = await db.create_session(entity_id)
        metadata, _ = assert_ok(create_result)
        
        # Invalid transition: INITIATED -> PAUSED (must go through ACTIVE)
        result = await db.transition_state(
            entity_id, metadata.session_id, SessionState.PAUSED
        )
        
        assert result.is_err()
        assert "Invalid transition" in result.error
        
        results.record("session_invalid_transition", True)
    except Exception as e:
        results.record("session_invalid_transition", False, str(e))


async def test_session_lease(results: TestResults):
    """Test lease acquire/release."""
    db = SessionDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Create session
        create_result = await db.create_session(entity_id)
        metadata, _ = assert_ok(create_result)
        
        holder_id = "node-001"
        
        # Acquire lease
        acquire_result = await db.acquire_lease(
            entity_id, metadata.session_id, holder_id
        )
        
        if acquire_result.is_err():
            results.record("session_lease", False, acquire_result.error)
            return
        
        fencing_token = acquire_result.unwrap()
        assert fencing_token == 1
        
        # Renew lease
        renew_result = await db.renew_lease(
            entity_id, metadata.session_id, holder_id, fencing_token
        )
        assert renew_result.is_ok()
        
        # Release lease
        release_result = await db.release_lease(
            entity_id, metadata.session_id, holder_id, fencing_token
        )
        assert release_result.is_ok()
        
        results.record("session_lease", True)
    except Exception as e:
        results.record("session_lease", False, str(e))


async def test_session_delete(results: TestResults):
    """Test session deletion."""
    db = SessionDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Create session
        create_result = await db.create_session(entity_id)
        metadata, _ = assert_ok(create_result)
        
        # Soft delete
        delete_result = await db.delete_session(
            entity_id, metadata.session_id, hard_delete=False
        )
        assert delete_result.is_ok()
        
        # Verify state is TERMINATED
        get_result = await db.get_session(entity_id, metadata.session_id)
        if get_result.is_ok():
            got_metadata, _ = get_result.unwrap()
            assert got_metadata.state == SessionState.TERMINATED
        
        results.record("session_delete", True)
    except Exception as e:
        results.record("session_delete", False, str(e))


async def test_session_list(results: TestResults):
    """Test listing sessions."""
    db = SessionDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Create multiple sessions
        for _ in range(3):
            await db.create_session(entity_id)
        
        # List sessions
        list_result = await db.list_sessions(entity_id)
        
        if list_result.is_err():
            results.record("session_list", False, list_result.error)
            return
        
        sessions, cursor = list_result.unwrap()
        assert len(sessions) == 3
        
        results.record("session_list", True)
    except Exception as e:
        results.record("session_list", False, str(e))


# =============================================================================
# HISTORY DATABASE LAYER TESTS
# =============================================================================

async def test_history_append(results: TestResults):
    """Test appending interactions."""
    db = HistoryDatabaseLayer()
    entity_id = create_test_entity()
    session_id = uuid4()
    
    try:
        interaction = Interaction(
            interaction_id=uuid4(),
            entity_id=entity_id,
            session_id=session_id,
            sequence_id=0,
            timestamp=datetime.now(timezone.utc),
            interaction_type=InteractionType.USER_MESSAGE,
            role="user",
            content="Hello, how are you?",
            token_count=5,
        )
        
        result = await db.append(interaction)
        
        if result.is_err():
            results.record("history_append", False, result.error)
            return
        
        appended = result.unwrap()
        assert appended.sequence_id == 1  # Auto-assigned
        
        results.record("history_append", True)
    except Exception as e:
        results.record("history_append", False, str(e))


async def test_history_batch_append(results: TestResults):
    """Test batch appending interactions."""
    db = HistoryDatabaseLayer()
    entity_id = create_test_entity()
    session_id = uuid4()
    
    try:
        interactions = []
        for i in range(10):
            interactions.append(Interaction(
                interaction_id=uuid4(),
                entity_id=entity_id,
                session_id=session_id,
                sequence_id=0,
                timestamp=datetime.now(timezone.utc),
                interaction_type=InteractionType.USER_MESSAGE,
                role="user",
                content=f"Message {i}",
                token_count=3,
            ))
        
        result = await db.batch_append(interactions)
        
        if result.is_err():
            results.record("history_batch_append", False, result.error)
            return
        
        appended = result.unwrap()
        assert len(appended) == 10
        assert all(i.sequence_id > 0 for i in appended)
        
        results.record("history_batch_append", True)
    except Exception as e:
        results.record("history_batch_append", False, str(e))


async def test_history_get_interaction(results: TestResults):
    """Test getting single interaction."""
    db = HistoryDatabaseLayer()
    entity_id = create_test_entity()
    session_id = uuid4()
    
    try:
        interaction = Interaction(
            interaction_id=uuid4(),
            entity_id=entity_id,
            session_id=session_id,
            sequence_id=0,
            timestamp=datetime.now(timezone.utc),
            interaction_type=InteractionType.USER_MESSAGE,
            role="user",
            content="Test message",
            token_count=3,
        )
        
        append_result = await db.append(interaction)
        appended = assert_ok(append_result)
        
        # Get by sequence ID
        get_result = await db.get_interaction(
            entity_id, session_id, appended.sequence_id
        )
        
        if get_result.is_err():
            results.record("history_get_interaction", False, get_result.error)
            return
        
        got = get_result.unwrap()
        assert got.content == "Test message"
        
        results.record("history_get_interaction", True)
    except Exception as e:
        results.record("history_get_interaction", False, str(e))


async def test_history_query_range(results: TestResults):
    """Test querying interaction range."""
    db = HistoryDatabaseLayer()
    entity_id = create_test_entity()
    session_id = uuid4()
    
    try:
        # Insert 20 interactions
        for i in range(20):
            await db.append(Interaction(
                interaction_id=uuid4(),
                entity_id=entity_id,
                session_id=session_id,
                sequence_id=0,
                timestamp=datetime.now(timezone.utc),
                interaction_type=InteractionType.USER_MESSAGE,
                role="user",
                content=f"Message {i}",
                token_count=3,
            ))
        
        # Query with limit
        result = await db.query_range(entity_id, session_id, limit=10)
        
        if result.is_err():
            results.record("history_query_range", False, result.error)
            return
        
        interactions, cursor = result.unwrap()
        assert len(interactions) == 10
        assert cursor is not None  # More results available
        
        results.record("history_query_range", True)
    except Exception as e:
        results.record("history_query_range", False, str(e))


async def test_history_get_recent(results: TestResults):
    """Test getting recent interactions."""
    db = HistoryDatabaseLayer()
    entity_id = create_test_entity()
    session_id = uuid4()
    
    try:
        # Insert interactions
        for i in range(10):
            await db.append(Interaction(
                interaction_id=uuid4(),
                entity_id=entity_id,
                session_id=session_id,
                sequence_id=0,
                timestamp=datetime.now(timezone.utc),
                interaction_type=InteractionType.USER_MESSAGE,
                role="user",
                content=f"Message {i}",
                token_count=3,
            ))
        
        result = await db.get_recent(entity_id, session_id, count=5)
        
        if result.is_err():
            results.record("history_get_recent", False, result.error)
            return
        
        interactions = result.unwrap()
        assert len(interactions) == 5
        
        results.record("history_get_recent", True)
    except Exception as e:
        results.record("history_get_recent", False, str(e))


async def test_history_streaming(results: TestResults):
    """Test streaming history."""
    db = HistoryDatabaseLayer()
    entity_id = create_test_entity()
    session_id = uuid4()
    
    try:
        # Insert interactions
        for i in range(20):
            await db.append(Interaction(
                interaction_id=uuid4(),
                entity_id=entity_id,
                session_id=session_id,
                sequence_id=0,
                timestamp=datetime.now(timezone.utc),
                interaction_type=InteractionType.USER_MESSAGE,
                role="user",
                content=f"Message {i}",
                token_count=3,
            ))
        
        # Stream all
        count = 0
        async for interaction in db.stream_history(entity_id, session_id, batch_size=5):
            count += 1
        
        assert count == 20
        
        results.record("history_streaming", True)
    except Exception as e:
        results.record("history_streaming", False, str(e))


async def test_history_buckets(results: TestResults):
    """Test bucket management."""
    db = HistoryDatabaseLayer()
    entity_id = create_test_entity()
    session_id = uuid4()
    
    try:
        # Insert interaction to create bucket
        await db.append(Interaction(
            interaction_id=uuid4(),
            entity_id=entity_id,
            session_id=session_id,
            sequence_id=0,
            timestamp=datetime.now(timezone.utc),
            interaction_type=InteractionType.USER_MESSAGE,
            role="user",
            content="Test",
            token_count=1,
        ))
        
        result = await db.list_buckets(entity_id, session_id)
        
        if result.is_err():
            results.record("history_buckets", False, result.error)
            return
        
        buckets = result.unwrap()
        assert len(buckets) >= 1
        assert buckets[0].tier == StorageTier.HOT
        
        results.record("history_buckets", True)
    except Exception as e:
        results.record("history_buckets", False, str(e))


async def test_history_stats(results: TestResults):
    """Test session statistics."""
    db = HistoryDatabaseLayer()
    entity_id = create_test_entity()
    session_id = uuid4()
    
    try:
        # Insert interactions
        total_tokens = 0
        for i in range(10):
            tokens = i + 1
            total_tokens += tokens
            await db.append(Interaction(
                interaction_id=uuid4(),
                entity_id=entity_id,
                session_id=session_id,
                sequence_id=0,
                timestamp=datetime.now(timezone.utc),
                interaction_type=InteractionType.USER_MESSAGE,
                role="user",
                content=f"Message {i}",
                token_count=tokens,
            ))
        
        result = await db.get_session_stats(entity_id, session_id)
        
        if result.is_err():
            results.record("history_stats", False, result.error)
            return
        
        stats = result.unwrap()
        assert stats["total_interactions"] == 10
        assert stats["total_tokens"] == total_tokens
        
        results.record("history_stats", True)
    except Exception as e:
        results.record("history_stats", False, str(e))


# =============================================================================
# MEMORY DATABASE LAYER TESTS
# =============================================================================

async def test_memory_stm_add(results: TestResults):
    """Test adding to STM."""
    db = MemoryDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        item = MemoryItem(
            item_id=uuid4(),
            entity_id=entity_id,
            content="Hello, this is a test message.",
            token_count=10,
            priority=MemoryPriority.HIGH,
        )
        
        result = await db.add_to_stm(item)
        
        if result.is_err():
            results.record("memory_stm_add", False, result.error)
            return
        
        added = result.unwrap()
        assert added.sequence_id == 1
        
        results.record("memory_stm_add", True)
    except Exception as e:
        results.record("memory_stm_add", False, str(e))


async def test_memory_stm_get(results: TestResults):
    """Test getting from STM."""
    db = MemoryDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        item = MemoryItem(
            item_id=uuid4(),
            entity_id=entity_id,
            content="Test content",
            token_count=5,
        )
        
        await db.add_to_stm(item)
        
        result = await db.get_from_stm(entity_id, item.item_id)
        
        if result.is_err():
            results.record("memory_stm_get", False, result.error)
            return
        
        got = result.unwrap()
        assert got.content == "Test content"
        
        results.record("memory_stm_get", True)
    except Exception as e:
        results.record("memory_stm_get", False, str(e))


async def test_memory_stm_list(results: TestResults):
    """Test listing STM items."""
    db = MemoryDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Add multiple items
        for i in range(5):
            await db.add_to_stm(MemoryItem(
                item_id=uuid4(),
                entity_id=entity_id,
                content=f"Item {i}",
                token_count=3,
            ))
        
        result = await db.list_stm(entity_id)
        
        if result.is_err():
            results.record("memory_stm_list", False, result.error)
            return
        
        items = result.unwrap()
        assert len(items) == 5
        
        results.record("memory_stm_list", True)
    except Exception as e:
        results.record("memory_stm_list", False, str(e))


async def test_memory_ltm_add(results: TestResults):
    """Test adding to LTM."""
    db = MemoryDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        item = MemoryItem(
            item_id=uuid4(),
            entity_id=entity_id,
            content="Long-term memory item with semantic content",
            token_count=10,
        )
        
        result = await db.add_to_ltm(item)
        
        if result.is_err():
            results.record("memory_ltm_add", False, result.error)
            return
        
        added = result.unwrap()
        assert added.embedding is not None  # Should have embedding
        
        results.record("memory_ltm_add", True)
    except Exception as e:
        results.record("memory_ltm_add", False, str(e))


async def test_memory_semantic_search(results: TestResults):
    """Test semantic search in LTM."""
    db = MemoryDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Add items to LTM
        for i, topic in enumerate(["Python programming", "JavaScript coding", "Database design"]):
            await db.add_to_ltm(MemoryItem(
                item_id=uuid4(),
                entity_id=entity_id,
                content=f"Content about {topic}",
                token_count=5,
            ))
        
        # Search
        query = MemoryQuery(
            entity_id=entity_id,
            text="programming languages",
            top_k=10,
        )
        
        result = await db.semantic_search(query)
        
        if result.is_err():
            results.record("memory_semantic_search", False, result.error)
            return
        
        recall = result.unwrap()
        assert len(recall.items) > 0
        assert recall.query_latency_ms > 0
        
        results.record("memory_semantic_search", True)
    except Exception as e:
        results.record("memory_semantic_search", False, str(e))


async def test_memory_build_context(results: TestResults):
    """Test context building."""
    db = MemoryDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Add items with different priorities
        await db.add_to_stm(MemoryItem(
            item_id=uuid4(),
            entity_id=entity_id,
            content="System prompt: You are a helpful assistant.",
            token_count=10,
            priority=MemoryPriority.CRITICAL,
        ))
        
        await db.add_to_stm(MemoryItem(
            item_id=uuid4(),
            entity_id=entity_id,
            content="User: Hello!",
            token_count=5,
            priority=MemoryPriority.HIGH,
        ))
        
        await db.add_to_stm(MemoryItem(
            item_id=uuid4(),
            entity_id=entity_id,
            content="Historical context...",
            token_count=20,
            priority=MemoryPriority.LOW,
        ))
        
        result = await db.build_context(entity_id, max_tokens=1000)
        
        if result.is_err():
            results.record("memory_build_context", False, result.error)
            return
        
        context = result.unwrap()
        assert len(context) == 3
        
        # Critical items should be included
        assert any(i.priority == MemoryPriority.CRITICAL for i in context)
        
        results.record("memory_build_context", True)
    except Exception as e:
        results.record("memory_build_context", False, str(e))


async def test_memory_stream_context(results: TestResults):
    """Test streaming context assembly."""
    db = MemoryDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Add items
        for i in range(10):
            await db.add_to_stm(MemoryItem(
                item_id=uuid4(),
                entity_id=entity_id,
                content=f"Message {i}: " + "x" * 100,
                token_count=50,
            ))
        
        # Stream context
        chunks = []
        async for chunk in db.stream_context(
            entity_id,
            max_tokens=10000,
            chunk_size=100,
        ):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        total_tokens = sum(c.token_count for c in chunks)
        assert total_tokens == 500  # 10 items * 50 tokens
        
        results.record("memory_stream_context", True)
    except Exception as e:
        results.record("memory_stream_context", False, str(e))


async def test_memory_eviction(results: TestResults):
    """Test STM eviction when over budget."""
    # Create DB with small token budget
    config = MemoryDBConfig(stm_max_tokens=100)
    db = MemoryDatabaseLayer(config=config)
    entity_id = create_test_entity()
    
    try:
        # Add items that exceed budget
        for i in range(10):
            await db.add_to_stm(MemoryItem(
                item_id=uuid4(),
                entity_id=entity_id,
                content=f"Item {i}",
                token_count=20,
                priority=MemoryPriority.LOW,
            ))
        
        result = await db.list_stm(entity_id)
        items = result.unwrap()
        
        # Should have evicted some items
        total_tokens = sum(i.token_count for i in items)
        assert total_tokens <= 100
        
        results.record("memory_eviction", True)
    except Exception as e:
        results.record("memory_eviction", False, str(e))


async def test_memory_consolidation(results: TestResults):
    """Test STM to LTM consolidation."""
    db = MemoryDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Add old item
        old_item = MemoryItem(
            item_id=uuid4(),
            entity_id=entity_id,
            content="Old content",
            token_count=10,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        await db.add_to_stm(old_item)
        
        # Add recent item
        await db.add_to_stm(MemoryItem(
            item_id=uuid4(),
            entity_id=entity_id,
            content="Recent content",
            token_count=10,
        ))
        
        # Consolidate (min age 1 hour)
        result = await db.consolidate_stm_to_ltm(entity_id, min_age_seconds=3600)
        
        if result.is_err():
            results.record("memory_consolidation", False, result.error)
            return
        
        consolidated = result.unwrap()
        assert consolidated == 1  # Only old item
        
        results.record("memory_consolidation", True)
    except Exception as e:
        results.record("memory_consolidation", False, str(e))


async def test_memory_stats(results: TestResults):
    """Test memory statistics."""
    db = MemoryDatabaseLayer()
    entity_id = create_test_entity()
    
    try:
        # Add items
        for i in range(5):
            await db.add_to_stm(MemoryItem(
                item_id=uuid4(),
                entity_id=entity_id,
                content=f"Item {i}",
                token_count=10,
            ))
        
        result = await db.get_memory_stats(entity_id)
        
        if result.is_err():
            results.record("memory_stats", False, result.error)
            return
        
        stats = result.unwrap()
        assert stats["stm_item_count"] == 5
        assert stats["stm_token_count"] == 50
        
        results.record("memory_stats", True)
    except Exception as e:
        results.record("memory_stats", False, str(e))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

async def test_full_conversation_flow(results: TestResults):
    """Test full conversation flow across all layers."""
    session_db = SessionDatabaseLayer()
    history_db = HistoryDatabaseLayer()
    memory_db = MemoryDatabaseLayer()
    
    entity_id = create_test_entity()
    
    try:
        # 1. Create session
        session_result = await session_db.create_session(entity_id)
        metadata, payload = assert_ok(session_result)
        session_id = metadata.session_id
        
        # 2. Activate session
        await session_db.transition_state(entity_id, session_id, SessionState.ACTIVE)
        
        # 3. Add system prompt to memory
        await memory_db.add_to_stm(MemoryItem(
            item_id=uuid4(),
            entity_id=entity_id,
            session_id=session_id,
            content="You are a helpful AI assistant.",
            token_count=10,
            priority=MemoryPriority.CRITICAL,
            role="system",
        ))
        
        # 4. Simulate conversation
        messages = [
            ("user", "Hello! How are you?"),
            ("assistant", "I'm doing great! How can I help you today?"),
            ("user", "Tell me about Python."),
            ("assistant", "Python is a versatile programming language..."),
        ]
        
        for role, content in messages:
            # Add to history
            await history_db.append(Interaction(
                interaction_id=uuid4(),
                entity_id=entity_id,
                session_id=session_id,
                sequence_id=0,
                timestamp=datetime.now(timezone.utc),
                interaction_type=InteractionType.USER_MESSAGE if role == "user" else InteractionType.ASSISTANT_MESSAGE,
                role=role,
                content=content,
                token_count=len(content.split()),
            ))
            
            # Add to memory
            await memory_db.add_to_stm(MemoryItem(
                item_id=uuid4(),
                entity_id=entity_id,
                session_id=session_id,
                content=f"{role}: {content}",
                token_count=len(content.split()),
                role=role,
            ))
        
        # 5. Build context
        context_result = await memory_db.build_context(entity_id, session_id)
        context = assert_ok(context_result)
        
        assert len(context) >= 5  # System + 4 messages
        
        # 6. Get history
        history_result = await history_db.get_recent(entity_id, session_id, count=10)
        history = assert_ok(history_result)
        
        assert len(history) == 4
        
        # 7. Terminate session
        await session_db.transition_state(entity_id, session_id, SessionState.TERMINATED)
        
        # Verify final state
        final_result = await session_db.get_session(entity_id, session_id, include_payload=False)
        final_metadata, _ = assert_ok(final_result)
        
        assert final_metadata.state == SessionState.TERMINATED
        
        results.record("full_conversation_flow", True)
    except Exception as e:
        results.record("full_conversation_flow", False, str(e))


async def test_large_context_window(results: TestResults):
    """Test large context window support (simulated)."""
    config = MemoryDBConfig(
        max_context_tokens=1_000_000,
        default_context_tokens=128_000,
        stm_max_tokens=200_000,
    )
    db = MemoryDatabaseLayer(config=config)
    entity_id = create_test_entity()
    
    try:
        # Simulate adding many items (representing large context)
        total_tokens = 0
        for i in range(100):
            tokens = 1000  # 1K tokens each
            total_tokens += tokens
            await db.add_to_stm(MemoryItem(
                item_id=uuid4(),
                entity_id=entity_id,
                content=f"Content block {i}: " + "x" * 500,
                token_count=tokens,
            ))
        
        # Build large context
        result = await db.build_context(entity_id, max_tokens=50000)
        context = assert_ok(result)
        
        context_tokens = sum(i.token_count for i in context)
        assert context_tokens <= 50000
        
        results.record("large_context_window", True)
    except Exception as e:
        results.record("large_context_window", False, str(e))


# =============================================================================
# MAIN
# =============================================================================

async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DATABASE LAYER TESTS")
    print("=" * 60)
    
    results = TestResults()
    
    # Session tests
    print("\n[Session Database Layer]")
    await test_session_create(results)
    await test_session_get(results)
    await test_session_update(results)
    await test_session_state_transition(results)
    await test_session_invalid_transition(results)
    await test_session_lease(results)
    await test_session_delete(results)
    await test_session_list(results)
    
    # History tests
    print("\n[History Database Layer]")
    await test_history_append(results)
    await test_history_batch_append(results)
    await test_history_get_interaction(results)
    await test_history_query_range(results)
    await test_history_get_recent(results)
    await test_history_streaming(results)
    await test_history_buckets(results)
    await test_history_stats(results)
    
    # Memory tests
    print("\n[Memory Database Layer]")
    await test_memory_stm_add(results)
    await test_memory_stm_get(results)
    await test_memory_stm_list(results)
    await test_memory_ltm_add(results)
    await test_memory_semantic_search(results)
    await test_memory_build_context(results)
    await test_memory_stream_context(results)
    await test_memory_eviction(results)
    await test_memory_consolidation(results)
    await test_memory_stats(results)
    
    # Integration tests
    print("\n[Integration Tests]")
    await test_full_conversation_flow(results)
    await test_large_context_window(results)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {results.summary()}")
    print("=" * 60)
    
    if results.errors:
        print("\nFailed tests:")
        for error in results.errors:
            print(f"  - {error}")
    
    return results.failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
