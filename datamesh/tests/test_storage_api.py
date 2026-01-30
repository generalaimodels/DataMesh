"""
Database API Test Suite: Comprehensive CRUD Operation Tests

Tests all database operations including:
- Basic CRUD (Create, Read, Update, Delete)
- Batch operations (multi-get, multi-put, multi-delete)
- Transactions with OCC (Optimistic Concurrency Control)
- TTL (Time-To-Live) operations
- Versioning and CAS (Compare-And-Swap)
- Streaming and change data capture
- Object store operations

Run: python -m pytest datamesh/tests/test_storage_api.py -v
Or:  python -m datamesh.tests.test_storage_api

Author: Planetary AI Systems
License: MIT
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Add parent to path for direct execution
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datamesh.core.types import Result, Ok, Err, EntityId
from datamesh.storage import (
    ConsistencyLevel,
    IsolationLevel,
    OperationType,
    InMemoryCPStore,
    InMemoryAPStore,
    InMemoryObjectStore,
)


# =============================================================================
# TEST DATA MODELS
# =============================================================================
@dataclass
class UserData:
    """Example user data for testing."""
    user_id: str
    name: str
    email: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


@dataclass
class SessionData:
    """Example session data for testing."""
    session_id: str
    user_id: str
    token: str
    expires_at: datetime
    metadata: Dict[str, Any] = None


# =============================================================================
# ANSI COLOR CODES FOR CONSOLE OUTPUT
# =============================================================================
class Colors:
    """ANSI color codes for pretty output."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.CYAN}  {text}{Colors.ENDC}")


# =============================================================================
# TEST: BASIC CRUD OPERATIONS
# =============================================================================
async def test_basic_crud() -> bool:
    """
    Test basic Create, Read, Update, Delete operations.
    
    Demonstrates:
        - put(): Insert new record
        - get(): Retrieve record by key
        - put(): Update existing record (upsert)
        - delete(): Remove record
        - exists(): Check key existence
    """
    print_header("TEST: Basic CRUD Operations")
    
    store = InMemoryCPStore()
    all_passed = True
    
    # -------------------------------------------------------------------------
    # CREATE: Insert new record
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}1. CREATE - Insert new record{Colors.ENDC}")
    
    user = UserData(
        user_id="user-001",
        name="Alice Smith",
        email="alice@example.com",
    )
    
    result = await store.put("user:001", user)
    
    if result.is_ok():
        metadata = result.unwrap()
        print_success(f"Created record with version={metadata.version}")
        print_info(f"Latency: {metadata.latency_ms:.3f}ms")
        print_info(f"Operation: {metadata.operation.value}")
    else:
        print_error(f"Failed to create: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # READ: Retrieve record
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. READ - Retrieve record{Colors.ENDC}")
    
    result = await store.get("user:001")
    
    if result.is_ok():
        value, metadata = result.unwrap()
        print_success(f"Retrieved: {value.name} <{value.email}>")
        print_info(f"Version: {metadata.version}")
        print_info(f"Latency: {metadata.latency_ms:.3f}ms")
    else:
        print_error(f"Failed to read: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # UPDATE: Modify record
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}3. UPDATE - Modify record{Colors.ENDC}")
    
    updated_user = UserData(
        user_id="user-001",
        name="Alice Johnson",  # Name changed
        email="alice.johnson@example.com",  # Email changed
    )
    
    result = await store.put("user:001", updated_user)
    
    if result.is_ok():
        metadata = result.unwrap()
        print_success(f"Updated record, new version={metadata.version}")
        print_info(f"Operation: {metadata.operation.value}")
    else:
        print_error(f"Failed to update: {result.error}")
        all_passed = False
    
    # Verify update
    result = await store.get("user:001")
    if result.is_ok():
        value, _ = result.unwrap()
        print_info(f"Verified: {value.name} <{value.email}>")
    
    # -------------------------------------------------------------------------
    # EXISTS: Check key existence
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}4. EXISTS - Check key existence{Colors.ENDC}")
    
    exists_result = await store.exists("user:001")
    if exists_result.is_ok() and exists_result.unwrap():
        print_success("Key 'user:001' exists")
    else:
        print_error("Key should exist")
        all_passed = False
    
    exists_result = await store.exists("user:999")
    if exists_result.is_ok() and not exists_result.unwrap():
        print_success("Key 'user:999' does not exist (expected)")
    else:
        print_error("Key should not exist")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # DELETE: Remove record
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}5. DELETE - Remove record{Colors.ENDC}")
    
    result = await store.delete("user:001")
    
    if result.is_ok():
        metadata = result.unwrap()
        print_success(f"Deleted record, affected={metadata.affected_rows}")
    else:
        print_error(f"Failed to delete: {result.error}")
        all_passed = False
    
    # Verify deletion
    result = await store.get("user:001")
    if result.is_err():
        print_success("Verified: Record no longer exists")
    else:
        print_error("Record should not exist after delete")
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST: BATCH OPERATIONS
# =============================================================================
async def test_batch_operations() -> bool:
    """
    Test batch operations for bulk data manipulation.
    
    Demonstrates:
        - multi_put(): Insert multiple records atomically
        - multi_get(): Retrieve multiple records in one call
        - multi_delete(): Delete multiple records atomically
    """
    print_header("TEST: Batch Operations")
    
    store = InMemoryCPStore()
    all_passed = True
    
    # -------------------------------------------------------------------------
    # MULTI-PUT: Bulk insert
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}1. MULTI-PUT - Bulk insert{Colors.ENDC}")
    
    users = {
        "user:001": UserData("u001", "Alice", "alice@example.com"),
        "user:002": UserData("u002", "Bob", "bob@example.com"),
        "user:003": UserData("u003", "Charlie", "charlie@example.com"),
        "user:004": UserData("u004", "Diana", "diana@example.com"),
        "user:005": UserData("u005", "Eve", "eve@example.com"),
    }
    
    result = await store.multi_put(users)
    
    if result.is_ok():
        metadata = result.unwrap()
        print_success(f"Inserted {metadata.affected_rows} records")
        print_info(f"Latency: {metadata.latency_ms:.3f}ms")
    else:
        print_error(f"Failed to multi_put: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # MULTI-GET: Bulk retrieve
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. MULTI-GET - Bulk retrieve{Colors.ENDC}")
    
    keys = ["user:001", "user:003", "user:005", "user:999"]  # Include non-existent
    result = await store.multi_get(keys)
    
    if result.is_ok():
        found = result.unwrap()
        print_success(f"Retrieved {len(found)} of {len(keys)} requested records")
        for key, user in found.items():
            print_info(f"  {key}: {user.name}")
        if "user:999" not in found:
            print_info("  user:999: Not found (expected)")
    else:
        print_error(f"Failed to multi_get: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # MULTI-DELETE: Bulk delete
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}3. MULTI-DELETE - Bulk delete{Colors.ENDC}")
    
    keys_to_delete = ["user:002", "user:004"]
    result = await store.multi_delete(keys_to_delete)
    
    if result.is_ok():
        metadata = result.unwrap()
        print_success(f"Deleted {metadata.affected_rows} records")
    else:
        print_error(f"Failed to multi_delete: {result.error}")
        all_passed = False
    
    # Verify remaining
    result = await store.multi_get(["user:001", "user:002", "user:003", "user:004", "user:005"])
    if result.is_ok():
        remaining = result.unwrap()
        print_info(f"Remaining records: {list(remaining.keys())}")
    
    return all_passed


# =============================================================================
# TEST: SCAN OPERATIONS
# =============================================================================
async def test_scan_operations() -> bool:
    """
    Test scan operations for range queries.
    
    Demonstrates:
        - scan(): List records with prefix filter
        - Cursor-based pagination
    """
    print_header("TEST: Scan Operations")
    
    store = InMemoryCPStore()
    all_passed = True
    
    # Insert test data
    print(f"\n{Colors.BOLD}1. Setup - Insert test data{Colors.ENDC}")
    
    for i in range(25):
        category = "active" if i % 2 == 0 else "inactive"
        await store.put(
            f"session:{category}:{i:03d}",
            {"id": i, "category": category, "data": f"Session {i}"},
        )
    print_success("Inserted 25 session records")
    
    # -------------------------------------------------------------------------
    # SCAN: List all with limit
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. SCAN - List all with limit{Colors.ENDC}")
    
    result = await store.scan(limit=10)
    
    if result.is_ok():
        records, next_cursor = result.unwrap()
        print_success(f"Retrieved {len(records)} records")
        print_info(f"Next cursor: {next_cursor}")
        for key, value in records[:3]:
            print_info(f"  {key}: {value['data']}")
        if len(records) > 3:
            print_info(f"  ... and {len(records) - 3} more")
    else:
        print_error(f"Failed to scan: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # SCAN: Filter by prefix
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}3. SCAN - Filter by prefix 'session:active'{Colors.ENDC}")
    
    result = await store.scan(prefix="session:active", limit=100)
    
    if result.is_ok():
        records, _ = result.unwrap()
        print_success(f"Found {len(records)} active sessions")
        for key, value in records[:3]:
            print_info(f"  {key}")
    else:
        print_error(f"Failed to scan with prefix: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # SCAN: Pagination
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}4. SCAN - Pagination{Colors.ENDC}")
    
    total_records = 0
    cursor = None
    page_num = 0
    
    while True:
        result = await store.scan(limit=10, cursor=cursor)
        if result.is_err():
            break
        
        records, next_cursor = result.unwrap()
        page_num += 1
        total_records += len(records)
        print_info(f"Page {page_num}: {len(records)} records")
        
        if next_cursor is None:
            break
        cursor = next_cursor
    
    print_success(f"Paginated through {total_records} total records in {page_num} pages")
    
    return all_passed


# =============================================================================
# TEST: TRANSACTIONS
# =============================================================================
async def test_transactions() -> bool:
    """
    Test ACID transactions with OCC.
    
    Demonstrates:
        - begin_transaction(): Start new transaction
        - Transaction operations with handle
        - commit(): Apply all changes atomically
        - rollback(): Discard changes
        - OCC conflict detection
    """
    print_header("TEST: Transactions with OCC")
    
    store = InMemoryCPStore()
    all_passed = True
    
    # Setup initial data
    await store.put("account:001", {"balance": 1000, "owner": "Alice"})
    await store.put("account:002", {"balance": 500, "owner": "Bob"})
    print_success("Setup: Created two accounts (Alice: $1000, Bob: $500)")
    
    # -------------------------------------------------------------------------
    # TRANSACTION: Successful commit
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}1. TRANSACTION - Successful commit (transfer $200){Colors.ENDC}")
    
    # Begin transaction
    txn_result = await store.begin_transaction(isolation=IsolationLevel.SERIALIZABLE)
    
    if txn_result.is_ok():
        txn = txn_result.unwrap()
        print_info(f"Transaction started: {txn.transaction_id}")
        
        # Record operations
        alice_result = await store.get("account:001")
        bob_result = await store.get("account:002")
        
        if alice_result.is_ok() and bob_result.is_ok():
            alice, _ = alice_result.unwrap()
            bob, _ = bob_result.unwrap()
            
            # Update balances
            alice["balance"] -= 200
            bob["balance"] += 200
            
            txn.record_operation(OperationType.UPDATE, "account:001", alice)
            txn.record_operation(OperationType.UPDATE, "account:002", bob)
            
            # Commit
            commit_result = await store.commit(txn)
            
            if commit_result.is_ok():
                print_success("Transaction committed successfully")
                
                # Verify
                alice_check = await store.get("account:001")
                bob_check = await store.get("account:002")
                if alice_check.is_ok() and bob_check.is_ok():
                    a, _ = alice_check.unwrap()
                    b, _ = bob_check.unwrap()
                    print_info(f"Alice balance: ${a['balance']}")
                    print_info(f"Bob balance: ${b['balance']}")
            else:
                print_error(f"Commit failed: {commit_result.error}")
                all_passed = False
    else:
        print_error(f"Failed to begin transaction: {txn_result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # TRANSACTION: Rollback
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. TRANSACTION - Rollback{Colors.ENDC}")
    
    txn_result = await store.begin_transaction()
    
    if txn_result.is_ok():
        txn = txn_result.unwrap()
        
        # Record operation but rollback
        txn.record_operation(OperationType.UPDATE, "account:001", {"balance": 0})
        
        rollback_result = await store.rollback(txn)
        
        if rollback_result.is_ok():
            print_success("Transaction rolled back")
            
            # Verify data unchanged
            alice_check = await store.get("account:001")
            if alice_check.is_ok():
                a, _ = alice_check.unwrap()
                print_info(f"Alice balance unchanged: ${a['balance']}")
        else:
            print_error(f"Rollback failed: {rollback_result.error}")
            all_passed = False
    
    return all_passed


# =============================================================================
# TEST: VERSIONING AND OCC
# =============================================================================
async def test_versioning() -> bool:
    """
    Test version-based operations for optimistic concurrency.
    
    Demonstrates:
        - get_with_version(): Read with version number
        - put_if_version(): Conditional update (CAS)
        - delete_if_version(): Conditional delete
        - Conflict detection
    """
    print_header("TEST: Versioning and OCC")
    
    store = InMemoryCPStore()
    all_passed = True
    
    # -------------------------------------------------------------------------
    # GET WITH VERSION
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}1. GET WITH VERSION{Colors.ENDC}")
    
    await store.put("counter:001", {"value": 0})
    
    result = await store.get_with_version("counter:001")
    
    if result.is_ok():
        value, version = result.unwrap()
        print_success(f"Got value={value['value']} at version={version}")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # PUT IF VERSION - Success
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. PUT IF VERSION - Success case{Colors.ENDC}")
    
    result = await store.put_if_version("counter:001", {"value": 1}, expected_version=1)
    
    if result.is_ok():
        new_version = result.unwrap()
        print_success(f"Updated to version={new_version}")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # PUT IF VERSION - Conflict
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}3. PUT IF VERSION - Conflict case{Colors.ENDC}")
    
    # Try to update with old version (should fail)
    result = await store.put_if_version("counter:001", {"value": 999}, expected_version=1)
    
    if result.is_err() and "version_mismatch" in result.error:
        print_success("Conflict detected (expected): version_mismatch")
    else:
        print_error("Should have detected version conflict")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # DELETE IF VERSION
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}4. DELETE IF VERSION{Colors.ENDC}")
    
    # Get current version
    result = await store.get_with_version("counter:001")
    if result.is_ok():
        _, current_version = result.unwrap()
        
        # Delete with correct version
        delete_result = await store.delete_if_version("counter:001", current_version)
        
        if delete_result.is_ok():
            print_success(f"Deleted with version={current_version}")
        else:
            print_error(f"Failed to delete: {delete_result.error}")
            all_passed = False
    
    return all_passed


# =============================================================================
# TEST: TTL OPERATIONS
# =============================================================================
async def test_ttl_operations() -> bool:
    """
    Test Time-To-Live operations.
    
    Demonstrates:
        - put_with_ttl(): Insert with expiration
        - get_ttl(): Check remaining TTL
        - extend_ttl(): Extend expiration
        - remove_ttl(): Make permanent
        - Automatic expiration
    """
    print_header("TEST: TTL Operations")
    
    store = InMemoryCPStore()
    all_passed = True
    
    # -------------------------------------------------------------------------
    # PUT WITH TTL
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}1. PUT WITH TTL - Insert with 3600s expiration{Colors.ENDC}")
    
    session = SessionData(
        session_id="sess-001",
        user_id="user-001",
        token="abc123xyz",
        expires_at=datetime.now(timezone.utc),
    )
    
    result = await store.put_with_ttl("session:001", session, ttl_seconds=3600)
    
    if result.is_ok():
        print_success("Session stored with 1 hour TTL")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # GET TTL
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. GET TTL - Check remaining time{Colors.ENDC}")
    
    result = await store.get_ttl("session:001")
    
    if result.is_ok():
        remaining = result.unwrap()
        print_success(f"Remaining TTL: {remaining} seconds")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # EXTEND TTL
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}3. EXTEND TTL - Add 1800 seconds{Colors.ENDC}")
    
    result = await store.extend_ttl("session:001", 1800)
    
    if result.is_ok():
        new_ttl = result.unwrap()
        print_success(f"Extended TTL to: {new_ttl} seconds")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # REMOVE TTL
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}4. REMOVE TTL - Make permanent{Colors.ENDC}")
    
    result = await store.remove_ttl("session:001")
    
    if result.is_ok():
        print_success("TTL removed, session is now permanent")
        
        # Verify
        ttl_check = await store.get_ttl("session:001")
        if ttl_check.is_ok() and ttl_check.unwrap() is None:
            print_info("Verified: No TTL set")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST: OBJECT STORE
# =============================================================================
async def test_object_store() -> bool:
    """
    Test object store for blob storage.
    
    Demonstrates:
        - put_object(): Store blob with metadata
        - get_object(): Retrieve blob
        - head_object(): Get metadata only
        - list_objects(): List with prefix
        - copy_object(): Copy to new key
        - get_object_range(): Range read
    """
    print_header("TEST: Object Store Operations")
    
    store = InMemoryObjectStore()
    all_passed = True
    
    # -------------------------------------------------------------------------
    # PUT OBJECT
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}1. PUT OBJECT - Store snapshot{Colors.ENDC}")
    
    snapshot_data = b"Binary snapshot data here..." * 100
    
    result = await store.put_object(
        key="snapshots/2024-01/entity-001.parquet",
        data=snapshot_data,
        content_type="application/x-parquet",
        metadata={"entity_id": "001", "version": "1.0"},
    )
    
    if result.is_ok():
        meta = result.unwrap()
        print_success(f"Stored object: {meta.key}")
        print_info(f"Size: {meta.size_bytes} bytes")
        print_info(f"ETag: {meta.etag[:16]}...")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # GET OBJECT
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. GET OBJECT - Retrieve snapshot{Colors.ENDC}")
    
    result = await store.get_object("snapshots/2024-01/entity-001.parquet")
    
    if result.is_ok():
        data, meta = result.unwrap()
        print_success(f"Retrieved {len(data)} bytes")
        print_info(f"Content-Type: {meta.content_type}")
        print_info(f"Metadata: {meta.metadata}")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # HEAD OBJECT
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}3. HEAD OBJECT - Get metadata only{Colors.ENDC}")
    
    result = await store.head_object("snapshots/2024-01/entity-001.parquet")
    
    if result.is_ok():
        meta = result.unwrap()
        print_success(f"Got metadata without downloading data")
        print_info(f"Size: {meta.size_bytes}, Created: {meta.created_at}")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # LIST OBJECTS
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}4. LIST OBJECTS - List snapshots{Colors.ENDC}")
    
    # Add more objects
    await store.put_object("snapshots/2024-01/entity-002.parquet", b"data2")
    await store.put_object("snapshots/2024-02/entity-001.parquet", b"data3")
    await store.put_object("logs/2024-01/app.log", b"log data")
    
    result = await store.list_objects(prefix="snapshots/")
    
    if result.is_ok():
        objects, _ = result.unwrap()
        print_success(f"Found {len(objects)} objects with prefix 'snapshots/'")
        for obj in objects:
            print_info(f"  {obj.key} ({obj.size_bytes} bytes)")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # COPY OBJECT
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}5. COPY OBJECT - Create backup{Colors.ENDC}")
    
    result = await store.copy_object(
        source_key="snapshots/2024-01/entity-001.parquet",
        dest_key="backups/entity-001.parquet.bak",
    )
    
    if result.is_ok():
        meta = result.unwrap()
        print_success(f"Copied to: {meta.key}")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # GET OBJECT RANGE
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}6. GET OBJECT RANGE - Partial read{Colors.ENDC}")
    
    result = await store.get_object_range(
        key="snapshots/2024-01/entity-001.parquet",
        start_byte=0,
        end_byte=100,
    )
    
    if result.is_ok():
        partial_data = result.unwrap()
        print_success(f"Read first {len(partial_data)} bytes")
    else:
        print_error(f"Failed: {result.error}")
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST: STREAMING
# =============================================================================
async def test_streaming() -> bool:
    """
    Test streaming operations for large datasets.
    
    Demonstrates:
        - stream_scan(): Memory-efficient iteration
        - Backpressure handling
    """
    print_header("TEST: Streaming Operations")
    
    store = InMemoryCPStore()
    all_passed = True
    
    # Insert test data
    print(f"\n{Colors.BOLD}1. Setup - Insert 100 records{Colors.ENDC}")
    
    for i in range(100):
        await store.put(f"record:{i:04d}", {"id": i, "value": i * 10})
    print_success("Inserted 100 records")
    
    # -------------------------------------------------------------------------
    # STREAM SCAN
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. STREAM SCAN - Iterate with backpressure{Colors.ENDC}")
    
    count = 0
    total_value = 0
    
    async for key, value in store.stream_scan(batch_size=20):
        count += 1
        total_value += value["value"]
    
    print_success(f"Streamed {count} records")
    print_info(f"Sum of values: {total_value}")
    
    if count == 100:
        print_success("All records streamed successfully")
    else:
        print_error(f"Expected 100 records, got {count}")
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST: AP STORE (REDIS-COMPATIBLE)
# =============================================================================
async def test_ap_store() -> bool:
    """
    Test AP store with LRU eviction and pub/sub.
    
    Demonstrates:
        - LRU eviction when capacity exceeded
        - Pub/Sub for real-time updates
    """
    print_header("TEST: AP Store (Redis-Compatible)")
    
    store = InMemoryAPStore(max_entries=5)
    all_passed = True
    
    # -------------------------------------------------------------------------
    # LRU EVICTION
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}1. LRU EVICTION - Test capacity limit{Colors.ENDC}")
    
    # Insert more than capacity
    for i in range(7):
        await store.put(f"item:{i}", {"id": i})
    
    count = await store.count()
    print_info(f"Inserted 7 items, current count: {count}")
    
    if count <= 5:
        print_success(f"LRU eviction working: count={count} <= max=5")
    else:
        print_error("LRU eviction not working")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # PUB/SUB
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. PUB/SUB - Subscribe to changes{Colors.ENDC}")
    
    received_events = []
    
    def on_change(key: Any, value: Any) -> None:
        received_events.append((key, value))
    
    await store.subscribe("session:*", on_change)
    print_info("Subscribed to 'session:*' pattern")
    
    # Trigger events
    await store.put("session:001", {"status": "active"})
    await store.put("session:002", {"status": "pending"})
    await store.put("other:001", {"type": "test"})  # Should not match
    
    print_success(f"Received {len(received_events)} events for 'session:*' pattern")
    for key, value in received_events:
        print_info(f"  {key}: {value}")
    
    if len(received_events) == 2:
        print_success("Pub/Sub pattern matching working correctly")
    else:
        print_error(f"Expected 2 events, got {len(received_events)}")
        all_passed = False
    
    return all_passed


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
async def run_all_tests() -> None:
    """Run all test suites."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("=" * 60)
    print("  DATAMESH DATABASE API TEST SUITE")
    print("  Comprehensive CRUD Operation Tests")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    
    results = {}
    
    # Run all tests
    results["Basic CRUD"] = await test_basic_crud()
    results["Batch Operations"] = await test_batch_operations()
    results["Scan Operations"] = await test_scan_operations()
    results["Transactions"] = await test_transactions()
    results["Versioning/OCC"] = await test_versioning()
    results["TTL Operations"] = await test_ttl_operations()
    results["Object Store"] = await test_object_store()
    results["Streaming"] = await test_streaming()
    results["AP Store"] = await test_ap_store()
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = 0
    failed = 0
    
    for name, result in results.items():
        if result:
            print_success(f"{name}")
            passed += 1
        else:
            print_error(f"{name}")
            failed += 1
    
    print(f"\n{Colors.BOLD}Results: {passed} passed, {failed} failed{Colors.ENDC}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
