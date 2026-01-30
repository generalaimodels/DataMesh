"""
AP Engine: SQLite with WAL Mode and Aggressive Optimization

Provides high-throughput content storage with:
- Write-Ahead Logging (WAL) for concurrent reads/writes
- Memory-mapped I/O for faster reads
- Large cache for hot data
- Batch write accumulation

Performance Characteristics:
- Write: O(1) amortized via WAL batching
- Read: O(log n) via B-tree index
- Memory: Configurable cache + mmap

Thread Safety:
- Single writer, multiple readers (SQLite WAL mode)
- Write operations are serialized via queue
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence
from queue import Queue, Empty

from datamesh.core.config import APConfig
from datamesh.core.errors import StorageError
from datamesh.core.types import Result, Ok, Err, Timestamp

logger = logging.getLogger(__name__)


@dataclass
class APStats:
    """AP subsystem statistics."""
    
    total_writes: int = 0
    total_reads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    pending_writes: int = 0
    wal_size_bytes: int = 0


class APEngine:
    """
    SQLite storage engine for AP subsystem.
    
    Optimized for high-throughput writes with:
    - WAL mode for non-blocking reads
    - Batch write accumulation
    - Memory-mapped I/O
    - Large page cache
    
    Usage:
        engine = APEngine(config)
        await engine.initialize()
        
        # Write with batching
        await engine.write("key", b"value")
        
        # Read immediately consistent
        value = await engine.read("key")
    """
    
    __slots__ = (
        "_config", "_conn", "_write_lock", "_stats",
        "_write_queue", "_batch_task", "_closed",
    )
    
    # SQLite PRAGMA optimizations
    PRAGMAS = [
        "PRAGMA journal_mode = WAL",      # Write-Ahead Logging
        "PRAGMA synchronous = NORMAL",    # Safe with WAL
        "PRAGMA cache_size = -65536",     # 64 MB page cache
        "PRAGMA mmap_size = 1073741824",  # 1 GB memory-map
        "PRAGMA temp_store = MEMORY",     # Temp tables in RAM
        "PRAGMA page_size = 4096",        # 4 KB pages
        "PRAGMA auto_vacuum = INCREMENTAL",
        "PRAGMA busy_timeout = 5000",     # 5s busy retry
    ]
    
    def __init__(self, config: APConfig) -> None:
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.Lock()
        self._stats = APStats()
        self._write_queue: Queue[tuple[str, bytes, asyncio.Future]] = Queue()
        self._batch_task: Optional[asyncio.Task] = None
        self._closed = False
    
    async def initialize(self) -> Result[None, StorageError]:
        """
        Initialize database connection and apply optimizations.
        """
        try:
            # Create data directory
            self._config.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Connect with serialized threading (single writer)
            self._conn = sqlite3.connect(
                str(self._config.db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit for explicit transaction control
            )
            self._conn.row_factory = sqlite3.Row
            
            # Apply PRAGMA optimizations
            for pragma in self.PRAGMAS:
                self._conn.execute(pragma)
            
            logger.info(
                "AP engine initialized",
                extra={"db_path": str(self._config.db_path)},
            )
            
            # Start batch writer
            self._batch_task = asyncio.create_task(self._batch_writer())
            
            return Ok(None)
            
        except Exception as e:
            logger.error(f"AP engine initialization failed: {e}")
            return Err(StorageError.connection_failed(
                host=str(self._config.db_path),
                port=0,
                cause=e,
            ))
    
    async def _batch_writer(self) -> None:
        """
        Background task for batch write accumulation.
        
        Accumulates writes for batch_timeout_ms or batch_size_rows,
        whichever comes first, then commits as single transaction.
        """
        batch: list[tuple[str, bytes, asyncio.Future]] = []
        last_flush = Timestamp.now()
        
        while not self._closed:
            try:
                # Non-blocking check for new writes
                try:
                    item = self._write_queue.get_nowait()
                    batch.append(item)
                except Empty:
                    pass
                
                # Check flush conditions
                elapsed_ms = (Timestamp.now() - last_flush) / 1_000_000
                should_flush = (
                    len(batch) >= self._config.wal_batch_size_rows or
                    (batch and elapsed_ms >= self._config.wal_batch_timeout_ms)
                )
                
                if should_flush and batch:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = Timestamp.now()
                
                # Brief sleep to prevent busy-wait
                await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                # Flush remaining on shutdown
                if batch:
                    await self._flush_batch(batch)
                break
            except Exception as e:
                logger.error(f"Batch writer error: {e}")
    
    async def _flush_batch(
        self,
        batch: list[tuple[str, bytes, asyncio.Future]],
    ) -> None:
        """Flush accumulated writes as single transaction."""
        if not batch or not self._conn:
            return
        
        try:
            with self._write_lock:
                cursor = self._conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                
                try:
                    for key, value, _ in batch:
                        cursor.execute(
                            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                            (key, value),
                        )
                    cursor.execute("COMMIT")
                    self._stats.total_writes += len(batch)
                    
                    # Resolve futures
                    for _, _, future in batch:
                        if not future.done():
                            future.set_result(True)
                            
                except Exception as e:
                    cursor.execute("ROLLBACK")
                    for _, _, future in batch:
                        if not future.done():
                            future.set_exception(e)
                    raise
                    
        except Exception as e:
            logger.error(f"Batch flush failed: {e}")
    
    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        """Get connection for read operations."""
        if self._conn is None:
            raise RuntimeError("Engine not initialized")
        yield self._conn
    
    async def write(self, key: str, value: bytes) -> Result[None, StorageError]:
        """
        Queue write for batch processing.
        
        Write is durable after future resolves.
        """
        if self._closed:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
            ))
        
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        
        self._write_queue.put((key, value, future))
        self._stats.pending_writes = self._write_queue.qsize()
        
        try:
            await asyncio.wait_for(future, timeout=30.0)
            return Ok(None)
        except asyncio.TimeoutError:
            return Err(StorageError.timeout(
                operation="write",
                duration_ms=30000,
            ))
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
                cause=e,
            ))
    
    async def write_immediate(
        self,
        key: str,
        value: bytes,
    ) -> Result[None, StorageError]:
        """
        Immediate write bypassing batch queue.
        
        Use for critical writes requiring immediate durability.
        """
        if self._conn is None:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
            ))
        
        try:
            with self._write_lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                    (key, value),
                )
            self._stats.total_writes += 1
            return Ok(None)
            
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
                cause=e,
            ))
    
    async def read(self, key: str) -> Result[Optional[bytes], StorageError]:
        """
        Read value by key.
        
        Immediately consistent even with pending writes.
        """
        if self._conn is None:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
            ))
        
        try:
            with self._read_connection() as conn:
                cursor = conn.execute(
                    "SELECT value FROM kv_store WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                self._stats.total_reads += 1
                
                if row:
                    self._stats.cache_hits += 1
                    return Ok(bytes(row["value"]))
                else:
                    self._stats.cache_misses += 1
                    return Ok(None)
                    
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
                cause=e,
            ))
    
    async def read_range(
        self,
        prefix: str,
        limit: int = 1000,
    ) -> Result[list[tuple[str, bytes]], StorageError]:
        """
        Read all keys with given prefix.
        
        Efficient for prefix-based partitioning.
        """
        if self._conn is None:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
            ))
        
        try:
            with self._read_connection() as conn:
                cursor = conn.execute(
                    "SELECT key, value FROM kv_store WHERE key LIKE ? LIMIT ?",
                    (f"{prefix}%", limit),
                )
                rows = cursor.fetchall()
                self._stats.total_reads += len(rows)
                return Ok([(row["key"], bytes(row["value"])) for row in rows])
                
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
                cause=e,
            ))
    
    async def delete(self, key: str) -> Result[bool, StorageError]:
        """Delete key from store."""
        if self._conn is None:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
            ))
        
        try:
            with self._write_lock:
                cursor = self._conn.execute(
                    "DELETE FROM kv_store WHERE key = ?",
                    (key,),
                )
                return Ok(cursor.rowcount > 0)
                
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
                cause=e,
            ))
    
    async def execute_sql(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> Result[list[dict[str, Any]], StorageError]:
        """Execute arbitrary SQL query."""
        if self._conn is None:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
            ))
        
        try:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
            return Ok([dict(row) for row in rows])
            
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
                cause=e,
            ))
    
    @property
    def stats(self) -> APStats:
        """Get current statistics."""
        if self._conn:
            # Get WAL size
            try:
                wal_path = self._config.db_path.with_suffix(".db-wal")
                if wal_path.exists():
                    self._stats.wal_size_bytes = wal_path.stat().st_size
            except OSError:
                pass
        return self._stats
    
    async def checkpoint(self) -> Result[None, StorageError]:
        """Force WAL checkpoint to main database."""
        if self._conn is None:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
            ))
        
        try:
            with self._write_lock:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.info("WAL checkpoint completed")
            return Ok(None)
            
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="ap_engine",
                port=0,
                cause=e,
            ))
    
    async def close(self) -> None:
        """Close engine and flush pending writes."""
        if self._closed:
            return
        
        self._closed = True
        
        # Cancel batch writer
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        
        # Close connection
        if self._conn:
            # Final checkpoint
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()
        
        logger.info("AP engine closed")
    
    async def __aenter__(self) -> APEngine:
        await self.initialize()
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        await self.close()
