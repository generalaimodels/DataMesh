"""
AP Schema: DDL for SQLite Content Tables

Tables:
- kv_store: Generic key-value storage
- response_dataframes: Structured response content
- embeddings: Vector storage for semantic search
"""

from __future__ import annotations

import logging

from datamesh.storage.ap.engine import APEngine
from datamesh.core.types import Result, Ok, Err
from datamesh.core.errors import StorageError

logger = logging.getLogger(__name__)


class APSchema:
    """Schema manager for AP subsystem."""
    
    KV_STORE_DDL = """
    CREATE TABLE IF NOT EXISTS kv_store (
        key TEXT PRIMARY KEY NOT NULL,
        value BLOB NOT NULL,
        created_at INTEGER DEFAULT (strftime('%s', 'now')),
        updated_at INTEGER DEFAULT (strftime('%s', 'now'))
    );
    """
    
    RESPONSE_DATAFRAMES_DDL = """
    CREATE TABLE IF NOT EXISTS response_dataframes (
        entity_id TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        sequence_id INTEGER NOT NULL,
        payload_ref TEXT,
        embedding BLOB,
        token_length INTEGER,
        quality_score REAL,
        metadata TEXT,
        created_at INTEGER DEFAULT (strftime('%s', 'now')),
        PRIMARY KEY (entity_id, conversation_id, sequence_id)
    );
    """
    
    RESPONSE_DATAFRAMES_INDEXES = [
        """CREATE INDEX IF NOT EXISTS idx_resp_entity 
           ON response_dataframes(entity_id);""",
        """CREATE INDEX IF NOT EXISTS idx_resp_created 
           ON response_dataframes(created_at DESC);""",
        """CREATE INDEX IF NOT EXISTS idx_resp_quality 
           ON response_dataframes(quality_score DESC);""",
    ]
    
    EMBEDDINGS_DDL = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id TEXT PRIMARY KEY NOT NULL,
        vector BLOB NOT NULL,
        dimensions INTEGER NOT NULL,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
    );
    """
    
    EMBEDDINGS_INDEXES = [
        """CREATE INDEX IF NOT EXISTS idx_emb_source 
           ON embeddings(source_type, source_id);""",
    ]
    
    def __init__(self, engine: APEngine) -> None:
        self._engine = engine
    
    async def create_all(self) -> Result[None, StorageError]:
        """Create all tables and indexes."""
        try:
            # Create tables
            result = await self._engine.execute_sql(self.KV_STORE_DDL)
            if result.is_err():
                return result
            
            result = await self._engine.execute_sql(self.RESPONSE_DATAFRAMES_DDL)
            if result.is_err():
                return result
            
            result = await self._engine.execute_sql(self.EMBEDDINGS_DDL)
            if result.is_err():
                return result
            
            # Create indexes
            for idx in self.RESPONSE_DATAFRAMES_INDEXES:
                result = await self._engine.execute_sql(idx)
            
            for idx in self.EMBEDDINGS_INDEXES:
                result = await self._engine.execute_sql(idx)
            
            logger.info("AP schema created successfully")
            return Ok(None)
            
        except Exception as e:
            logger.error(f"AP schema creation failed: {e}")
            return Err(StorageError.connection_failed(
                host="sqlite",
                port=0,
                cause=e,
            ))
    
    async def drop_all(self) -> Result[None, StorageError]:
        """Drop all tables (DANGER)."""
        try:
            await self._engine.execute_sql("DROP TABLE IF EXISTS embeddings")
            await self._engine.execute_sql("DROP TABLE IF EXISTS response_dataframes")
            await self._engine.execute_sql("DROP TABLE IF EXISTS kv_store")
            
            logger.warning("AP schema dropped")
            return Ok(None)
            
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="sqlite",
                port=0,
                cause=e,
            ))
