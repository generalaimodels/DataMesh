# DataMesh Storage API

> **Planetary-Scale Conversational State Mesh**  
> SOTA-level database abstraction layer for session, history, and memory management supporting 100K-1M token context windows.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [Basic CRUD Operations](#basic-crud-operations)
  - [Batch Operations](#batch-operations)
  - [Transactions](#transactions)
  - [Versioning (OCC)](#versioning-occ)
  - [TTL Operations](#ttl-operations)
  - [Streaming](#streaming)
  - [Object Store](#object-store)
- [Advanced Usage](#advanced-usage)
- [Performance](#performance)
- [Architecture](#architecture)

## Overview

The DataMesh Storage API provides a unified interface for managing conversational state across distributed systems. It implements:

- **CP (Control Plane)**: Strong consistency via Spanner-compatible backends
- **AP (Availability Plane)**: High availability via Redis-compatible backends
- **Object Store**: S3-compatible blob storage for snapshots and artifacts

### Key Features

| Feature | Description |
|---------|-------------|
| **Zero-Exception Control Flow** | All operations return `Result[T, E]` monad |
| **ACID Transactions** | Optimistic Concurrency Control (OCC) |
| **TTL Support** | Automatic expiration for session data |
| **Streaming** | Memory-efficient iteration for large datasets |
| **Sharding** | Consistent hashing for horizontal scaling |
| **CDC** | Change Data Capture for event-driven systems |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/datamesh.git
cd datamesh

# Install dependencies
pip install -e .
```

## Quick Start

```python
import asyncio
from datamesh.storage import InMemoryCPStore, InMemoryAPStore

async def main():
    # Create a CP store (strong consistency)
    store = InMemoryCPStore[str, dict]()
    
    # INSERT: Create a new record
    result = await store.put("user:001", {"name": "Alice", "email": "alice@example.com"})
    if result.is_ok():
        print(f"Created with version: {result.unwrap().version}")
    
    # SELECT: Retrieve the record
    result = await store.get("user:001")
    if result.is_ok():
        user, metadata = result.unwrap()
        print(f"Found: {user['name']} (version {metadata.version})")
    
    # UPDATE: Modify the record
    result = await store.put("user:001", {"name": "Alice Smith", "email": "alice@example.com"})
    if result.is_ok():
        print(f"Updated to version: {result.unwrap().version}")
    
    # DELETE: Remove the record
    result = await store.delete("user:001")
    if result.is_ok():
        print(f"Deleted {result.unwrap().affected_rows} record(s)")

asyncio.run(main())
```

## Core Concepts

### Result Monad

All operations return a `Result[T, E]` type for zero-exception control flow:

```python
from datamesh.core.types import Result, Ok, Err

async def safe_get(store, key: str):
    result = await store.get(key)
    
    if result.is_ok():
        value, metadata = result.unwrap()
        return value
    else:
        # Handle error without exceptions
        print(f"Error: {result.error}")
        return None
```

### Consistency Levels

```python
from datamesh.storage import ConsistencyLevel

# Read from any replica (fastest)
await store.get(key, consistency=ConsistencyLevel.ONE)

# Read from majority (balanced)
await store.get(key, consistency=ConsistencyLevel.QUORUM)

# Read from all replicas (strongest)
await store.get(key, consistency=ConsistencyLevel.ALL)

# Linearizable read (Paxos/Raft)
await store.get(key, consistency=ConsistencyLevel.SERIAL)
```

## API Reference

### Basic CRUD Operations

#### CREATE: Insert a new record

```python
from datamesh.storage import InMemoryCPStore

store = InMemoryCPStore[str, dict]()

# Simple insert
result = await store.put("user:001", {
    "user_id": "001",
    "name": "Alice",
    "email": "alice@example.com",
    "created_at": "2024-01-15T10:30:00Z"
})

if result.is_ok():
    metadata = result.unwrap()
    print(f"Created record:")
    print(f"  Version: {metadata.version}")
    print(f"  Latency: {metadata.latency_ms:.3f}ms")
    print(f"  Operation: {metadata.operation.value}")
else:
    print(f"Error: {result.error}")
```

#### READ: Retrieve a record

```python
# Get by key
result = await store.get("user:001")

if result.is_ok():
    user, metadata = result.unwrap()
    print(f"User: {user['name']}")
    print(f"Email: {user['email']}")
    print(f"Version: {metadata.version}")
else:
    print(f"Not found: {result.error}")

# Check existence without fetching
exists = await store.exists("user:001")
if exists.is_ok() and exists.unwrap():
    print("User exists")
```

#### UPDATE: Modify a record

```python
# Upsert semantics (insert or update)
result = await store.put("user:001", {
    "user_id": "001",
    "name": "Alice Smith",  # Updated name
    "email": "alice.smith@example.com",  # Updated email
    "updated_at": "2024-01-16T14:00:00Z"
})

if result.is_ok():
    metadata = result.unwrap()
    print(f"Updated to version: {metadata.version}")
    # version will be 2 if record existed
```

#### DELETE: Remove a record

```python
result = await store.delete("user:001")

if result.is_ok():
    metadata = result.unwrap()
    print(f"Deleted {metadata.affected_rows} record(s)")
    # affected_rows = 1 if existed, 0 if not
else:
    print(f"Delete failed: {result.error}")
```

### Batch Operations

#### MULTI-PUT: Bulk insert/update

```python
users = {
    "user:001": {"name": "Alice", "role": "admin"},
    "user:002": {"name": "Bob", "role": "user"},
    "user:003": {"name": "Charlie", "role": "user"},
    "user:004": {"name": "Diana", "role": "moderator"},
}

result = await store.multi_put(users)

if result.is_ok():
    metadata = result.unwrap()
    print(f"Inserted/updated {metadata.affected_rows} records")
    print(f"Total latency: {metadata.latency_ms:.3f}ms")
```

#### MULTI-GET: Bulk retrieve

```python
keys = ["user:001", "user:002", "user:003", "user:999"]

result = await store.multi_get(keys)

if result.is_ok():
    found = result.unwrap()
    print(f"Found {len(found)} of {len(keys)} records:")
    for key, user in found.items():
        print(f"  {key}: {user['name']}")
    # Note: user:999 won't be in results if not found
```

#### MULTI-DELETE: Bulk delete

```python
keys_to_delete = ["user:002", "user:004"]

result = await store.multi_delete(keys_to_delete)

if result.is_ok():
    metadata = result.unwrap()
    print(f"Deleted {metadata.affected_rows} records")
```

### Scan Operations

#### List with prefix filter

```python
# Scan all sessions
result = await store.scan(prefix="session:", limit=100)

if result.is_ok():
    records, next_cursor = result.unwrap()
    for key, value in records:
        print(f"{key}: {value}")
    
    if next_cursor:
        print(f"More records available, cursor: {next_cursor}")
```

#### Pagination

```python
async def paginate_all(store, prefix: str, page_size: int = 100):
    """Iterate through all records with pagination."""
    cursor = None
    total = 0
    
    while True:
        result = await store.scan(
            prefix=prefix,
            limit=page_size,
            cursor=cursor
        )
        
        if result.is_err():
            break
        
        records, next_cursor = result.unwrap()
        total += len(records)
        
        for key, value in records:
            yield key, value
        
        if next_cursor is None:
            break
        cursor = next_cursor
    
    print(f"Total records: {total}")

# Usage
async for key, value in paginate_all(store, "user:"):
    process(key, value)
```

### Transactions

#### Basic transaction

```python
from datamesh.storage import IsolationLevel

# Begin transaction
txn_result = await store.begin_transaction(
    isolation=IsolationLevel.SERIALIZABLE,
    timeout_ms=30000
)

if txn_result.is_ok():
    txn = txn_result.unwrap()
    
    try:
        # Read current state
        alice = await store.get("account:alice")
        bob = await store.get("account:bob")
        
        if alice.is_ok() and bob.is_ok():
            alice_data, _ = alice.unwrap()
            bob_data, _ = bob.unwrap()
            
            # Transfer $100
            alice_data["balance"] -= 100
            bob_data["balance"] += 100
            
            # Record operations
            from datamesh.storage import OperationType
            txn.record_operation(OperationType.UPDATE, "account:alice", alice_data)
            txn.record_operation(OperationType.UPDATE, "account:bob", bob_data)
            
            # Commit
            commit_result = await store.commit(txn)
            if commit_result.is_ok():
                print("Transfer successful!")
            else:
                print(f"Commit failed: {commit_result.error}")
                await store.rollback(txn)
    except Exception as e:
        await store.rollback(txn)
        raise
```

#### Context manager pattern (recommended)

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def transaction(store, isolation=IsolationLevel.SNAPSHOT):
    """Transaction context manager with automatic cleanup."""
    txn_result = await store.begin_transaction(isolation=isolation)
    if txn_result.is_err():
        raise RuntimeError(txn_result.error)
    
    txn = txn_result.unwrap()
    try:
        yield txn
        await store.commit(txn)
    except Exception:
        await store.rollback(txn)
        raise

# Usage
async with transaction(store) as txn:
    txn.record_operation(OperationType.CREATE, "order:001", order_data)
    txn.record_operation(OperationType.UPDATE, "inventory:item-1", updated_inventory)
```

### Versioning (OCC)

#### Optimistic concurrency control

```python
# Read with version
result = await store.get_with_version("counter:001")

if result.is_ok():
    value, version = result.unwrap()
    print(f"Current value: {value}, version: {version}")
    
    # Update only if version unchanged (CAS)
    new_value = {"count": value["count"] + 1}
    update_result = await store.put_if_version(
        "counter:001",
        new_value,
        expected_version=version
    )
    
    if update_result.is_ok():
        new_version = update_result.unwrap()
        print(f"Updated to version: {new_version}")
    elif "version_mismatch" in update_result.error:
        print("Conflict! Retry with fresh version")
    else:
        print(f"Error: {update_result.error}")
```

#### Retry pattern for conflicts

```python
async def atomic_increment(store, key: str, max_retries: int = 5):
    """Increment counter with automatic retry on conflict."""
    for attempt in range(max_retries):
        result = await store.get_with_version(key)
        
        if result.is_err():
            # Key doesn't exist, create it
            create_result = await store.put_if_version(key, {"count": 1}, 0)
            if create_result.is_ok():
                return 1
            continue
        
        value, version = result.unwrap()
        new_count = value["count"] + 1
        
        update_result = await store.put_if_version(
            key,
            {"count": new_count},
            expected_version=version
        )
        
        if update_result.is_ok():
            return new_count
        
        # Exponential backoff on conflict
        await asyncio.sleep(0.01 * (2 ** attempt))
    
    raise RuntimeError(f"Failed after {max_retries} attempts")
```

### TTL Operations

#### Session with expiration

```python
from datetime import datetime, timedelta

session_data = {
    "session_id": "sess-abc123",
    "user_id": "user-001",
    "token": "jwt-token-here",
    "created_at": datetime.now().isoformat()
}

# Create session with 1 hour TTL
result = await store.put_with_ttl(
    "session:abc123",
    session_data,
    ttl_seconds=3600  # 1 hour
)

if result.is_ok():
    print("Session created with 1 hour expiration")
```

#### Check and extend TTL

```python
# Check remaining TTL
ttl_result = await store.get_ttl("session:abc123")

if ttl_result.is_ok():
    remaining = ttl_result.unwrap()
    if remaining is not None:
        print(f"Session expires in {remaining} seconds")
        
        # Extend if less than 5 minutes remaining
        if remaining < 300:
            extend_result = await store.extend_ttl("session:abc123", 1800)
            if extend_result.is_ok():
                new_ttl = extend_result.unwrap()
                print(f"Extended to {new_ttl} seconds")
    else:
        print("Session has no expiration")
```

#### Remove TTL (make permanent)

```python
result = await store.remove_ttl("session:abc123")

if result.is_ok():
    print("Session is now permanent")
```

### Streaming

#### Memory-efficient iteration

```python
# Stream all records without loading into memory
async for key, value in store.stream_scan(prefix="log:", batch_size=100):
    # Process one record at a time
    process_log_entry(key, value)
    
    # Backpressure: slow down if needed
    if should_throttle():
        await asyncio.sleep(0.01)
```

#### Change Data Capture (CDC)

```python
from datetime import datetime, timedelta

# Get changes since 1 hour ago
since = datetime.now() - timedelta(hours=1)

async for op_type, key, value in store.stream_changes(since=since):
    if op_type == OperationType.CREATE:
        print(f"Created: {key}")
    elif op_type == OperationType.UPDATE:
        print(f"Updated: {key} -> {value}")
    elif op_type == OperationType.DELETE:
        print(f"Deleted: {key}")
```

### Object Store

#### Store large objects

```python
from datamesh.storage import InMemoryObjectStore

object_store = InMemoryObjectStore()

# Store a snapshot
snapshot_data = b"... binary parquet data ..."

result = await object_store.put_object(
    key="snapshots/2024-01/entity-001.parquet",
    data=snapshot_data,
    content_type="application/x-parquet",
    metadata={
        "entity_id": "001",
        "version": "1.0",
        "compression": "snappy"
    }
)

if result.is_ok():
    meta = result.unwrap()
    print(f"Stored: {meta.key}")
    print(f"Size: {meta.size_bytes} bytes")
    print(f"ETag: {meta.etag}")
```

#### Retrieve objects

```python
# Full object
result = await object_store.get_object("snapshots/2024-01/entity-001.parquet")

if result.is_ok():
    data, metadata = result.unwrap()
    print(f"Retrieved {len(data)} bytes")
    print(f"Content-Type: {metadata.content_type}")

# Metadata only (no download)
result = await object_store.head_object("snapshots/2024-01/entity-001.parquet")

if result.is_ok():
    meta = result.unwrap()
    print(f"Size: {meta.size_bytes}, ETag: {meta.etag}")

# Partial read (range request)
result = await object_store.get_object_range(
    "snapshots/2024-01/entity-001.parquet",
    start_byte=0,
    end_byte=1024
)

if result.is_ok():
    header_bytes = result.unwrap()
    print(f"Read first {len(header_bytes)} bytes")
```

#### List and manage objects

```python
# List objects with prefix
result = await object_store.list_objects(
    prefix="snapshots/2024-01/",
    limit=100
)

if result.is_ok():
    objects, continuation_token = result.unwrap()
    for obj in objects:
        print(f"{obj.key}: {obj.size_bytes} bytes")

# Copy object
await object_store.copy_object(
    source_key="snapshots/2024-01/entity-001.parquet",
    dest_key="backups/entity-001.parquet"
)

# Delete object
await object_store.delete_object("snapshots/2024-01/entity-001.parquet")
```

## Advanced Usage

### AP Store with LRU Eviction

```python
from datamesh.storage import InMemoryAPStore

# Create store with capacity limits
cache = InMemoryAPStore[str, dict](
    max_entries=10000,      # Maximum 10K entries
    max_bytes=100_000_000,  # Maximum 100MB
)

# LRU eviction happens automatically when limits exceeded
for i in range(15000):
    await cache.put(f"key:{i}", {"value": i})

count = await cache.count()
print(f"Entries after insertion: {count}")  # Will be ~10000
```

### Pub/Sub for Real-time Updates

```python
# Subscribe to changes matching pattern
def on_session_change(key, value):
    print(f"Session changed: {key} -> {value}")

await cache.subscribe("session:*", on_session_change)

# Changes trigger callbacks
await cache.put("session:001", {"status": "active"})
# Callback prints: Session changed: session:001 -> {'status': 'active'}

# Unsubscribe when done
await cache.unsubscribe("session:*", on_session_change)
```

## Performance

### Complexity Analysis

| Operation | Average | Worst Case |
|-----------|---------|------------|
| get | O(1) | O(1) |
| put | O(1) | O(1) |
| delete | O(1) | O(1) |
| exists | O(1) | O(1) |
| scan | O(k) | O(n) |
| multi_get | O(k) | O(k) |
| multi_put | O(k) | O(k) |
| transaction commit | O(n) | O(n) |

Where:
- k = result/batch size
- n = total records (for scan) or operations (for transactions)

### Benchmarks

```bash
# Run benchmarks
python -m datamesh.benchmarks.storage

# Expected results (in-memory):
# - get: ~50μs p99
# - put: ~100μs p99
# - batch_put(100): ~500μs p99
# - scan(100): ~200μs p99
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
├─────────────────────────────────────────────────────────────────┤
│                     Storage Protocols                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Database │ │  Batch   │ │   TTL    │ │ Version  │            │
│  │ Protocol │ │ Protocol │ │ Protocol │ │ Protocol │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                     Backend Implementations                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  InMemoryCPStore │  │  InMemoryAPStore │  │  InMemoryObject │  │
│  │  (Spanner-like)  │  │  (Redis-like)    │  │  Store (S3-like)│  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Production Backends                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │     Spanner     │  │   Redis/Valkey  │  │    S3 / GCS     │  │
│  │   CockroachDB   │  │   DragonflyDB   │  │   MinIO         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Running Tests

```bash
# Run all storage tests
python -m datamesh.tests.test_storage_api

# Run with pytest
python -m pytest datamesh/tests/test_storage_api.py -v

# Run with coverage
python -m pytest datamesh/tests/ --cov=datamesh.storage
```

## License

MIT License - see LICENSE file for details.
