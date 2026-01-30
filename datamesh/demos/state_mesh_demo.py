"""
Session, History, and Memory Demo

Demonstrates the complete lifecycle of:
1. Session Management - State machine, registry, cache
2. History Management - Timeline, partitioning, snapshots
3. Memory Management - STM working set, LTM vector store, hierarchy

Run: python -m datamesh.demos.state_mesh_demo
"""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

# Core types
from datamesh.core.types import EntityId, GeoRegion

# Session Manager
from datamesh.session import (
    SessionStateMachine,
    SessionState,
    SessionRegistry,
    SessionCache,
    SessionPayload,
    DistributedLease,
    SessionAffinity,
)

# History Manager
from datamesh.history import (
    ConversationTimeline,
    HistoryPartitioner,
    SnapshotManager,
    HistoryCursor,
)

# Memory Manager
from datamesh.memory.stm import (
    WorkingSet,
    WorkingSetConfig,
    WorkingSetItem,
    ContextWindow,
    ContextWindowConfig,
)
from datamesh.memory.ltm import (
    VectorStore,
    VectorStoreConfig,
    MemoryVector,
    GraphStore,
    MemoryNode,
    NodeType,
    EpisodicMemory,
    Episode,
    EpisodeType,
)
from datamesh.memory import MemoryHierarchy


async def demo_session_manager():
    """Demonstrate session lifecycle management."""
    print("\n" + "="*60)
    print("SESSION MANAGER DEMO")
    print("="*60)
    
    entity_id = EntityId.generate()
    session_id = str(uuid4())
    
    # Create context and state machine
    from datamesh.session.state_machine import SessionContext
    context = SessionContext(entity_id=entity_id, session_id=session_id)
    fsm = SessionStateMachine(context)
    print(f"✓ Created session FSM: {session_id}")
    print(f"  Initial state: {fsm.state.name}")
    
    # Record interaction first (required by guard for certain transitions)
    fsm.record_interaction(token_count=100)
    
    # Activate session with trigger
    result = fsm.transition("FIRST_INTERACTION")
    if result.is_ok():
        print(f"  Transitioned to: {fsm.state.name}")
    
    # Create registry and cache
    registry = SessionRegistry()
    cache = SessionCache()
    
    # Register session with SessionMetadata
    from datamesh.session.registry import SessionMetadata
    session_uuid = uuid4()
    session_meta = SessionMetadata(entity_id=entity_id, session_id=session_uuid)
    reg_result = await registry.create(session_meta)
    if reg_result.is_ok():
        print(f"✓ Registered session in CP registry")
    
    # Cache payload
    payload = SessionPayload(
        entity_id=entity_id,
        session_id=session_uuid,
        context_data=b"Recent conversation context...",
    )
    await cache.put(payload)
    print(f"✓ Cached session payload ({len(payload.context_data)} bytes)")
    
    # Demonstrate lease
    lease = DistributedLease()
    lock_result = await lease.acquire(
        resource_id=f"session:{session_id}",
    )
    if lock_result.is_ok():
        handle = lock_result.unwrap()
        print(f"✓ Acquired distributed lease (fencing token: {handle.fencing_token})")
        await lease.release(handle.resource_id, handle.fencing_token)
    
    # Session affinity (simplified - full routing requires deeper integration)
    affinity = SessionAffinity()
    print(f"✓ Created session affinity manager (ready for node registration)")
    
    return entity_id, session_id


async def demo_history_manager(entity_id: EntityId):
    """Demonstrate history timeline and partitioning."""
    print("\n" + "="*60)
    print("HISTORY MANAGER DEMO")
    print("="*60)
    
    session_id = uuid4()
    
    # Create timeline
    timeline = ConversationTimeline(entity_id=entity_id)
    
    # Import required types
    from datamesh.history.timeline import Interaction, InteractionType
    
    # Append interactions
    for i in range(5):
        interaction_type = InteractionType.PROMPT if i % 2 == 0 else InteractionType.RESPONSE
        interaction = Interaction(
            entity_id=entity_id,
            session_id=session_id,
            sequence_id=i + 1,
            interaction_type=interaction_type,
            content=f"Message {i+1} in the conversation".encode(),
            token_count=10 + i * 5,
        )
        result = await timeline.append(interaction)
        if result.is_ok():
            seq_id = result.unwrap()
            print(f"✓ Appended interaction #{seq_id}")
    
    # Query recent
    recent_result = await timeline.get_recent(limit=3)
    if recent_result.is_ok():
        recent = recent_result.unwrap()
        print(f"✓ Retrieved {len(recent)} recent interactions")
    
    # Demonstrate partitioner
    partitioner = HistoryPartitioner()
    bucket_id = partitioner.compute_bucket_id(datetime.now(timezone.utc))
    bucket = partitioner.get_or_create_bucket(entity_id, datetime.now(timezone.utc))
    print(f"✓ Current bucket: {bucket.bucket_id} (tier: {bucket.tier.name})")
    
    # Create snapshot
    snapshot_mgr = SnapshotManager()
    snapshot_result = await snapshot_mgr.create(
        entity_id=entity_id,
        source_bucket_ids=[bucket_id],
        description="Demo snapshot",
    )
    if snapshot_result.is_ok():
        snap = snapshot_result.unwrap()
        print(f"✓ Created snapshot: {snap.snapshot_id}")
        print(f"  Content hash: {snap.content_hash[:16]}...")
    
    return session_id


async def demo_memory_manager(entity_id: EntityId):
    """Demonstrate STM and LTM memory systems."""
    print("\n" + "="*60)
    print("MEMORY MANAGER DEMO")
    print("="*60)
    
    # === STM: Working Set ===
    print("\n--- Short-Term Memory ---")
    
    ws_config = WorkingSetConfig(max_items=10, max_tokens=500)
    working_set = WorkingSet(entity_id=entity_id, config=ws_config)
    
    # Add items to working set
    for i in range(5):
        item = WorkingSetItem(
            item_id=f"msg-{i+1}",
            sequence_id=i + 1,
            content=f"Conversation turn {i+1}".encode(),
            token_count=20 + i * 10,
            role="user" if i % 2 == 0 else "assistant",
        )
        await working_set.add(item)
    
    print(f"✓ Working set: {working_set.count} items, {working_set.total_tokens} tokens")
    
    # Build context window
    context_config = ContextWindowConfig(max_tokens=200, reserve_tokens=50)
    context_window = ContextWindow(working_set, context_config)
    context_window.set_system_context("You are a helpful AI assistant.")
    
    context_result = await context_window.build_context()
    if context_result.is_ok():
        items = context_result.unwrap()
        print(f"✓ Built context window: {len(items)} items")
        token_usage = await context_window.get_token_usage()
        print(f"  Token usage: {token_usage.get('total', 0)}/{context_config.available_tokens}")
    
    # === LTM: Vector Store ===
    print("\n--- Long-Term Memory ---")
    
    vector_store = VectorStore(config=VectorStoreConfig(dimension=64))
    
    # Insert vectors
    for i in range(3):
        # Create simple embedding (in production, use real embeddings)
        embedding = tuple((i + j) / 100.0 for j in range(64))
        vector = MemoryVector(
            entity_id=entity_id,
            embedding=embedding,
            content=f"Memory fragment {i+1}",
            token_count=50,
        )
        await vector_store.insert(vector)
    
    print(f"✓ Vector store: {await vector_store.count(entity_id)} vectors")
    
    # Search
    query_embedding = tuple(j / 100.0 for j in range(64))
    results = await vector_store.search(entity_id, query_embedding, k=2)
    print(f"✓ Semantic search returned {len(results)} results")
    
    # === LTM: Graph Store ===
    graph_store = GraphStore()
    
    # Create nodes
    user_node = MemoryNode(
        entity_id=entity_id,
        node_type=NodeType.ENTITY,
        label="User",
        value="Research Assistant User",
        confidence=0.95,
    )
    await graph_store.add_node(user_node)
    
    topic_node = MemoryNode(
        entity_id=entity_id,
        node_type=NodeType.CONCEPT,
        label="Machine Learning",
        value="AI/ML Research Topic",
    )
    await graph_store.add_node(topic_node)
    
    print(f"✓ Graph store: {graph_store.node_count} nodes, {graph_store.edge_count} edges")
    
    # === LTM: Episodic Memory ===
    episodic = EpisodicMemory()
    
    episode = Episode(
        entity_id=entity_id,
        episode_type=EpisodeType.TASK,
        title="ML Research Discussion",
        importance=0.8,
    )
    episode.add_turn("user", "What are the latest advances in LLMs?", 15)
    episode.add_turn("assistant", "Recent advances include...", 150)
    episode.complete(summary="Discussed LLM research trends")
    
    await episodic.create(episode)
    print(f"✓ Episodic memory: {episodic.total_episodes} episodes")
    
    # === Memory Hierarchy ===
    print("\n--- Memory Hierarchy ---")
    
    hierarchy = MemoryHierarchy(
        working_set=working_set,
        vector_store=vector_store,
        graph_store=graph_store,
        episodic_memory=episodic,
    )
    
    # Unified recall
    from datamesh.memory.hierarchy import MemoryQuery, QueryType
    
    query = MemoryQuery(
        entity_id=entity_id,
        query_text="machine learning",
        query_type=QueryType.HYBRID,
        max_results=10,
    )
    
    recall = await hierarchy.recall(query)
    print(f"✓ Memory recall: {recall.total_count} items")
    print(f"  STM: {recall.stm_count}, Vector: {recall.vector_count}, "
          f"Graph: {recall.graph_count}, Episodic: {recall.episodic_count}")
    print(f"  Latency: {recall.latency_ms:.2f}ms")


async def main():
    """Run all demos."""
    print("="*60)
    print("DATAMESH STATE MESH DEMO")
    print("Session, History, and Memory Management")
    print("="*60)
    
    # Session Manager Demo
    entity_id, session_id = await demo_session_manager()
    
    # History Manager Demo
    conversation_id = await demo_history_manager(entity_id)
    
    # Memory Manager Demo
    await demo_memory_manager(entity_id)
    
    print("\n" + "="*60)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
