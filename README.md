# Planetary-Scale Conversational Data Mesh

A polyglot persistence architecture for conversational AI workloads, governed by CAP theorem partitioning.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway (Layer 7)                       │
│   Rate Limiting │ Authentication │ Tracing │ Load Balancing     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
    ┌───────────────────────▼───────────────────────┐
    │              Ingestion Pipeline               │
    │  Backpressure │ Deduplication │ Saga Pattern  │
    └───────────────────────┬───────────────────────┘
                            │
    ┌───────────┬───────────┴───────────┬───────────┐
    ▼           ▼                       ▼           ▼
┌───────┐  ┌───────────┐       ┌──────────────┐  ┌───────┐
│  CP   │  │    AP     │       │   Object     │  │ Index │
│Tier   │  │   Tier    │       │   Storage    │  │ Tier  │
└───┬───┘  └─────┬─────┘       └──────┬───────┘  └───────┘
    │            │                    │
PostgreSQL   SQLite+LSM           S3/FS (CAS)
(Metadata)   (Content)           (Large Blobs)
```

## Components

### CP Subsystem (Consistency Priority)
- **Engine**: PostgreSQL with serializable isolation
- **Purpose**: Metadata, provenance chains, access control
- **Guarantees**: RPO=0 via synchronous replication

### AP Subsystem (Availability Priority)  
- **Engine**: SQLite with WAL mode + custom LSM-tree
- **Purpose**: Content storage, embeddings, response data
- **Guarantees**: High throughput (1M rows/sec), eventual consistency

### Object Storage
- **Engine**: Content-addressable storage (CAS) with SHA-256
- **Backend**: Local filesystem or S3-compatible
- **Features**: Automatic deduplication, integrity verification

### Pipeline
- **Saga Pattern**: Distributed transactions without 2PC
- **Backpressure**: Token bucket rate limiting
- **Deduplication**: Idempotency key management

### Observability
- **Metrics**: Prometheus-compatible counters/gauges/histograms
- **Tracing**: OpenTelemetry-compatible distributed tracing  
- **Logging**: JSON structured logs with trace correlation

## Quick Start

```bash
# Install (local mode, no external dependencies)
pip install -e .

# Run demo
python -m datamesh

# With PostgreSQL support
pip install -e ".[postgresql]"

# With S3 support
pip install -e ".[s3]"

# Full installation
pip install -e ".[full]"
```

## Usage Example

```python
import asyncio
from datamesh.core.config import DataMeshConfig
from datamesh.core.types import EntityId, GeoRegion
from datamesh.pipeline.ingestion import IngestionPipeline, IngestionRequest

async def main():
    # Load config from environment
    config = DataMeshConfig.from_env().unwrap()
    
    # Create pipeline
    pipeline = await IngestionPipeline.create(config)
    if pipeline.is_err():
        print(f"Error: {pipeline.error}")
        return
    
    async with pipeline.unwrap() as p:
        # Ingest document
        request = IngestionRequest(
            entity_id=EntityId.generate(),
            prompt=b"What is the capital of France?",
            response=b"The capital of France is Paris.",
            geo_region=GeoRegion.EU_WEST,
        )
        
        result = await p.ingest(request)
        if result.is_ok():
            resp = result.unwrap()
            print(f"Ingested: {resp.conversation_id}")
            print(f"Latency: {resp.latency_ms:.2f}ms")

asyncio.run(main())
```

## Design Principles

### Error Handling
- Return `Result` types, never exceptions for control flow
- Exhaustive pattern matching for all error variants
- Atomic state rollback on failures

### Memory Efficiency
- Struct members ordered by descending size
- Arena allocators for short-lived objects
- Zero-copy I/O where possible

### Concurrency
- Lock-free algorithms for high-contention paths
- Explicit acquire/release memory ordering
- Exponential backoff with full jitter for retries

### Observability
- Nanosecond-precision latency metrics
- 100% error sampling, 1% success sampling
- Structured JSON logs with correlation IDs

## License

MIT
