#!/usr/bin/env python3
"""
Planetary-Scale Conversational Data Mesh

Main entry point demonstrating complete system initialization
and basic operations.

Usage:
    python -m datamesh
    
    # Or with custom config
    DATAMESH_CP_HOST=postgres.example.com python -m datamesh
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from datamesh.core.config import DataMeshConfig
from datamesh.core.types import EntityId, ConversationId, EmbeddingVector, GeoRegion
from datamesh.observability.logging import setup_logging, LogLevel
from datamesh.observability.tracing import Tracer
from datamesh.observability.metrics import MetricsCollector
from datamesh.pipeline.ingestion import IngestionPipeline, IngestionRequest
from datamesh.api.router import DataMeshRouter, Request, Response
from datamesh.api.handlers import IngestHandler, QueryHandler
from datamesh.api.middleware import TracingMiddleware, RateLimitMiddleware


async def demo_local_mode() -> None:
    """
    Demonstrate data mesh with local storage (no external deps).
    
    Uses SQLite for both CP and AP subsystems.
    """
    print("\n" + "=" * 60)
    print("Planetary-Scale Conversational Data Mesh - Local Demo")
    print("=" * 60 + "\n")
    
    # Load config
    config_result = DataMeshConfig.from_env()
    if config_result.is_err():
        print(f"Configuration error: {config_result.error}")
        sys.exit(1)
    
    config = config_result.unwrap()
    
    # Validate config
    validation = config.validate()
    if validation.is_err():
        print(f"Validation error: {validation.error}")
        sys.exit(1)
    
    print("✓ Configuration loaded and validated")
    print(f"  CP: {config.cp.host}:{config.cp.port}")
    print(f"  AP: {config.ap.data_dir}")
    print(f"  Objects: {config.object_store.data_dir}")
    
    # Initialize observability
    setup_logging(LogLevel.INFO, json_output=False)
    tracer = Tracer.get_instance("datamesh-demo")
    metrics = MetricsCollector.get_instance()
    
    print("\n✓ Observability initialized")
    
    # Initialize AP engine directly (skip CP for local demo)
    from datamesh.storage.ap.engine import APEngine
    from datamesh.storage.ap.schema import APSchema
    from datamesh.storage.ap.repositories import ResponseRepository, ResponseDataframe
    from datamesh.storage.object_store.cas import ContentAddressableStorage
    
    # Create AP engine
    ap_engine = APEngine(config.ap)
    result = await ap_engine.initialize()
    if result.is_err():
        print(f"AP engine error: {result.error}")
        sys.exit(1)
    
    print("✓ AP engine (SQLite) initialized")
    
    # Create schema
    schema = APSchema(ap_engine)
    await schema.create_all()
    print("✓ Schema created")
    
    # Create CAS
    cas = ContentAddressableStorage(config.object_store)
    print("✓ Content-addressed storage initialized")
    
    # Demo operations
    print("\n--- Demo Operations ---\n")
    
    # 1. Store content in CAS
    with tracer.start_span("store_content") as span:
        content = b"Hello, Planetary Data Mesh!"
        cas_result = await cas.put(content)
        
        if cas_result.is_ok():
            obj = cas_result.unwrap()
            print(f"1. Stored content: {obj.hash.to_hex()[:16]}... ({obj.size_bytes} bytes)")
            span.set_attribute("content_hash", obj.hash.to_hex())
        else:
            print(f"   Error: {cas_result.error}")
    
    # 2. Write to AP subsystem
    entity_id = EntityId.generate()
    conversation_id = ConversationId.generate()
    
    response = ResponseDataframe(
        entity_id=entity_id,
        conversation_id=conversation_id,
        sequence_id=0,
        payload_ref=f"cas://{obj.hash.to_hex()}" if cas_result.is_ok() else None,
        embedding=None,
        token_length=len(content),
        quality_score=0.95,
        metadata={"model": "demo", "temperature": 0.7},
    )
    
    resp_repo = ResponseRepository(ap_engine)
    insert_result = await resp_repo.insert(response)
    
    if insert_result.is_ok():
        print(f"2. Inserted response: entity={str(entity_id)[:8]}... conv={str(conversation_id)[:8]}...")
    else:
        print(f"   Error: {insert_result.error}")
    
    # 3. Read back
    read_result = await resp_repo.get_by_id(entity_id, conversation_id, 0)
    
    if read_result.is_ok() and read_result.unwrap():
        r = read_result.unwrap()
        print(f"3. Retrieved response: quality_score={r.quality_score}, metadata={r.metadata}")
    else:
        print(f"   Error: {read_result.error if read_result.is_err() else 'Not found'}")
    
    # 4. Show stats
    stats = ap_engine.stats
    print(f"\n4. AP Engine Stats:")
    print(f"   Total writes: {stats.total_writes}")
    print(f"   Total reads: {stats.total_reads}")
    print(f"   WAL size: {stats.wal_size_bytes} bytes")
    
    # Cleanup
    await ap_engine.close()
    
    print("\n✓ Demo complete")
    print("=" * 60 + "\n")


async def main() -> None:
    """Main entry point."""
    try:
        await demo_local_mode()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        raise


def run() -> None:
    """Synchronous entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
