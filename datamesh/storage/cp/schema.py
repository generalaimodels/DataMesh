"""
CP Schema: DDL for PostgreSQL Metadata Tables

Tables:
- conversation_roots: Primary conversation metadata (sharded by entity_id)
- instruction_sets: Immutable versioned instruction templates
- idempotency_keys: Deduplication for write operations

Design:
- UUID primary keys for global uniqueness
- BYTEA for content hashes (SHA-256)
- JSONB for flexible metadata
- Composite indexes for query patterns
"""

from __future__ import annotations

import logging
from typing import Any

from datamesh.storage.cp.engine import CPEngine
from datamesh.core.types import Result, Ok, Err
from datamesh.core.errors import StorageError

logger = logging.getLogger(__name__)


class CPSchema:
    """
    Schema manager for CP subsystem DDL operations.
    
    Provides idempotent schema creation and migration support.
    All DDL runs within transactions for atomicity.
    """
    
    # ==========================================================================
    # TABLE DEFINITIONS
    # ==========================================================================
    
    CONVERSATION_ROOTS_DDL = """
    CREATE TABLE IF NOT EXISTS conversation_roots (
        -- Primary key: sharded by entity_id
        entity_id UUID NOT NULL,
        conversation_id UUID NOT NULL,
        
        -- Content-addressable reference to prompt
        prompt_hash BYTEA NOT NULL,
        
        -- Immutable instruction schema version
        instruction_version INT NOT NULL DEFAULT 1,
        
        -- Temporal ordering
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        
        -- Geographic and compliance metadata
        geo_region VARCHAR(32) NOT NULL DEFAULT 'us-east-1',
        compliance_tier VARCHAR(16) NOT NULL DEFAULT 'STANDARD',
        
        -- Soft delete support
        deleted_at TIMESTAMPTZ,
        
        -- Composite primary key for sharding
        PRIMARY KEY (entity_id, conversation_id)
    );
    """
    
    CONVERSATION_ROOTS_INDEXES = [
        # Time-range queries for ingestion monitoring
        """CREATE INDEX IF NOT EXISTS idx_conv_created_at 
           ON conversation_roots(created_at DESC);""",
        
        # Geographic queries for compliance
        """CREATE INDEX IF NOT EXISTS idx_conv_geo_region 
           ON conversation_roots(geo_region);""",
        
        # Instruction version queries
        """CREATE INDEX IF NOT EXISTS idx_conv_instruction_version 
           ON conversation_roots(instruction_version);""",
        
        # Soft delete filtering
        """CREATE INDEX IF NOT EXISTS idx_conv_not_deleted 
           ON conversation_roots(entity_id, conversation_id) 
           WHERE deleted_at IS NULL;""",
    ]
    
    INSTRUCTION_SETS_DDL = """
    CREATE TABLE IF NOT EXISTS instruction_sets (
        -- Primary key with versioning
        instruction_id UUID NOT NULL,
        version INT NOT NULL,
        
        -- Content hash for idempotency
        instruction_hash BYTEA NOT NULL,
        
        -- Validated JSON Schema template
        template_structure JSONB NOT NULL,
        
        -- Metadata
        name VARCHAR(256) NOT NULL,
        description TEXT,
        
        -- Temporal
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        
        -- Immutable rows (no updates allowed)
        PRIMARY KEY (instruction_id, version),
        
        -- Unique hash constraint for deduplication
        UNIQUE (instruction_hash)
    );
    """
    
    INSTRUCTION_SETS_INDEXES = [
        # Name search
        """CREATE INDEX IF NOT EXISTS idx_instr_name 
           ON instruction_sets(name);""",
        
        # Latest version lookup
        """CREATE INDEX IF NOT EXISTS idx_instr_latest 
           ON instruction_sets(instruction_id, version DESC);""",
    ]
    
    IDEMPOTENCY_KEYS_DDL = """
    CREATE TABLE IF NOT EXISTS idempotency_keys (
        -- Idempotency key from request header
        idempotency_key VARCHAR(64) NOT NULL PRIMARY KEY,
        
        -- Result of original operation
        result_status VARCHAR(16) NOT NULL,
        result_payload JSONB,
        
        -- TTL management
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL,
        
        -- Request metadata for debugging
        request_hash BYTEA NOT NULL,
        entity_id UUID
    );
    """
    
    IDEMPOTENCY_KEYS_INDEXES = [
        # TTL cleanup
        """CREATE INDEX IF NOT EXISTS idx_idem_expires 
           ON idempotency_keys(expires_at);""",
        
        # Entity scoped lookups
        """CREATE INDEX IF NOT EXISTS idx_idem_entity 
           ON idempotency_keys(entity_id) WHERE entity_id IS NOT NULL;""",
    ]
    
    # ==========================================================================
    # SCHEMA OPERATIONS
    # ==========================================================================
    
    def __init__(self, engine: CPEngine) -> None:
        self._engine = engine
    
    async def create_all(self) -> Result[None, StorageError]:
        """
        Create all tables and indexes idempotently.
        
        Safe to call multiple times - uses IF NOT EXISTS.
        """
        try:
            async with self._engine.transaction() as txn:
                # Create tables
                await txn.execute(self.CONVERSATION_ROOTS_DDL)
                await txn.execute(self.INSTRUCTION_SETS_DDL)
                await txn.execute(self.IDEMPOTENCY_KEYS_DDL)
                
                # Create indexes
                for idx in self.CONVERSATION_ROOTS_INDEXES:
                    await txn.execute(idx)
                for idx in self.INSTRUCTION_SETS_INDEXES:
                    await txn.execute(idx)
                for idx in self.IDEMPOTENCY_KEYS_INDEXES:
                    await txn.execute(idx)
            
            logger.info("CP schema created successfully")
            return Ok(None)
            
        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            return Err(StorageError.connection_failed(
                host="localhost",
                port=5432,
                cause=e,
            ))
    
    async def drop_all(self) -> Result[None, StorageError]:
        """
        Drop all tables (DANGER: irreversible).
        
        Only use in testing or development.
        """
        try:
            async with self._engine.transaction() as txn:
                await txn.execute("DROP TABLE IF EXISTS idempotency_keys CASCADE")
                await txn.execute("DROP TABLE IF EXISTS instruction_sets CASCADE")
                await txn.execute("DROP TABLE IF EXISTS conversation_roots CASCADE")
            
            logger.warning("CP schema dropped")
            return Ok(None)
            
        except Exception as e:
            return Err(StorageError.connection_failed(
                host="localhost",
                port=5432,
                cause=e,
            ))
    
    async def table_exists(self, table_name: str) -> Result[bool, StorageError]:
        """Check if table exists in database."""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = $1
            );
        """
        result = await self._engine.execute(query, table_name)
        if result.is_err():
            return result
        
        rows = result.unwrap().rows
        return Ok(rows[0]["exists"] if rows else False)
    
    async def get_version(self) -> Result[int, StorageError]:
        """Get current schema version from migration table."""
        # For now, return static version
        return Ok(1)
