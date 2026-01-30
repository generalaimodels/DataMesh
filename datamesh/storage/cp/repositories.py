"""
CP Repositories: Data Access Layer for PostgreSQL

Repository pattern implementation for:
- ConversationRepository: CRUD for conversation_roots
- InstructionRepository: CRUD for instruction_sets
- IdempotencyRepository: Deduplication key management

Design:
- All methods return Result types (no exceptions)
- Prepared statements for hot queries
- Batch operations for bulk inserts
- Cursor-based pagination
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence
from uuid import UUID

from datamesh.core.types import (
    Result, Ok, Err,
    EntityId, ConversationId, InstructionId,
    ContentHash, Timestamp, ComplianceTier, GeoRegion,
    PaginationCursor,
)
from datamesh.core.errors import StorageError, IngestionError, QueryError
from datamesh.storage.cp.engine import CPEngine


@dataclass(frozen=True, slots=True)
class ConversationRoot:
    """Domain model for conversation_roots table."""
    
    entity_id: EntityId
    conversation_id: ConversationId
    prompt_hash: ContentHash
    instruction_version: int
    created_at: datetime
    geo_region: GeoRegion
    compliance_tier: ComplianceTier
    deleted_at: Optional[datetime] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": str(self.entity_id),
            "conversation_id": str(self.conversation_id),
            "prompt_hash": self.prompt_hash.to_hex(),
            "instruction_version": self.instruction_version,
            "created_at": self.created_at.isoformat(),
            "geo_region": self.geo_region.value,
            "compliance_tier": self.compliance_tier.name,
        }


@dataclass(frozen=True, slots=True)
class InstructionSet:
    """Domain model for instruction_sets table."""
    
    instruction_id: InstructionId
    version: int
    instruction_hash: ContentHash
    template_structure: dict[str, Any]
    name: str
    description: Optional[str]
    created_at: datetime


class ConversationRepository:
    """
    Repository for conversation_roots table.
    
    Provides CRUD operations with:
    - Idempotent inserts via prompt_hash
    - Time-range queries
    - Geo-region filtering
    - Soft delete support
    """
    
    __slots__ = ("_engine",)
    
    def __init__(self, engine: CPEngine) -> None:
        self._engine = engine
    
    async def insert(
        self,
        root: ConversationRoot,
    ) -> Result[ConversationRoot, StorageError]:
        """
        Insert new conversation root.
        
        Returns existing record if prompt_hash already exists.
        """
        query = """
            INSERT INTO conversation_roots (
                entity_id, conversation_id, prompt_hash,
                instruction_version, geo_region, compliance_tier
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (entity_id, conversation_id) DO NOTHING
            RETURNING *;
        """
        
        result = await self._engine.execute(
            query,
            root.entity_id.value,
            root.conversation_id.value,
            root.prompt_hash.digest,
            root.instruction_version,
            root.geo_region.value,
            root.compliance_tier.name,
        )
        
        if result.is_err():
            return result
        
        return Ok(root)
    
    async def insert_batch(
        self,
        roots: Sequence[ConversationRoot],
    ) -> Result[int, StorageError]:
        """
        Batch insert conversation roots.
        
        Optimized for high-throughput ingestion.
        """
        if not roots:
            return Ok(0)
        
        query = """
            INSERT INTO conversation_roots (
                entity_id, conversation_id, prompt_hash,
                instruction_version, geo_region, compliance_tier
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (entity_id, conversation_id) DO NOTHING;
        """
        
        args_list = [
            (
                r.entity_id.value,
                r.conversation_id.value,
                r.prompt_hash.digest,
                r.instruction_version,
                r.geo_region.value,
                r.compliance_tier.name,
            )
            for r in roots
        ]
        
        return await self._engine.execute_many(query, args_list)
    
    async def get_by_id(
        self,
        entity_id: EntityId,
        conversation_id: ConversationId,
    ) -> Result[Optional[ConversationRoot], StorageError]:
        """Get conversation root by composite key."""
        query = """
            SELECT * FROM conversation_roots
            WHERE entity_id = $1 AND conversation_id = $2
            AND deleted_at IS NULL;
        """
        
        result = await self._engine.execute(
            query,
            entity_id.value,
            conversation_id.value,
        )
        
        if result.is_err():
            return result
        
        rows = result.unwrap().rows
        if not rows:
            return Ok(None)
        
        return Ok(self._row_to_model(rows[0]))
    
    async def list_by_entity(
        self,
        entity_id: EntityId,
        cursor: Optional[PaginationCursor] = None,
        limit: int = 100,
    ) -> Result[tuple[list[ConversationRoot], Optional[PaginationCursor]], StorageError]:
        """
        List conversations for entity with cursor pagination.
        
        Returns (results, next_cursor) tuple.
        """
        if cursor:
            query = """
                SELECT * FROM conversation_roots
                WHERE entity_id = $1
                AND deleted_at IS NULL
                AND (conversation_id, created_at) > ($2, $3)
                ORDER BY conversation_id, created_at
                LIMIT $4;
            """
            result = await self._engine.execute(
                query,
                entity_id.value,
                cursor.conversation_id,
                cursor.sequence_id,
                limit + 1,
            )
        else:
            query = """
                SELECT * FROM conversation_roots
                WHERE entity_id = $1
                AND deleted_at IS NULL
                ORDER BY conversation_id, created_at
                LIMIT $2;
            """
            result = await self._engine.execute(
                query,
                entity_id.value,
                limit + 1,
            )
        
        if result.is_err():
            return result
        
        rows = result.unwrap().rows
        has_more = len(rows) > limit
        items = [self._row_to_model(r) for r in rows[:limit]]
        
        next_cursor = None
        if has_more and items:
            last = items[-1]
            next_cursor = PaginationCursor(
                conversation_id=last.conversation_id.value,
                sequence_id=0,
            )
        
        return Ok((items, next_cursor))
    
    async def list_by_time_range(
        self,
        start: datetime,
        end: datetime,
        geo_region: Optional[GeoRegion] = None,
        limit: int = 1000,
    ) -> Result[list[ConversationRoot], StorageError]:
        """Query conversations by time range."""
        if geo_region:
            query = """
                SELECT * FROM conversation_roots
                WHERE created_at >= $1 AND created_at < $2
                AND geo_region = $3
                AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT $4;
            """
            result = await self._engine.execute(
                query, start, end, geo_region.value, limit
            )
        else:
            query = """
                SELECT * FROM conversation_roots
                WHERE created_at >= $1 AND created_at < $2
                AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT $3;
            """
            result = await self._engine.execute(query, start, end, limit)
        
        if result.is_err():
            return result
        
        return Ok([self._row_to_model(r) for r in result.unwrap().rows])
    
    async def soft_delete(
        self,
        entity_id: EntityId,
        conversation_id: ConversationId,
    ) -> Result[bool, StorageError]:
        """Soft delete conversation root."""
        query = """
            UPDATE conversation_roots
            SET deleted_at = NOW()
            WHERE entity_id = $1 AND conversation_id = $2
            AND deleted_at IS NULL;
        """
        
        result = await self._engine.execute(
            query,
            entity_id.value,
            conversation_id.value,
        )
        
        return Ok(True) if result.is_ok() else result
    
    def _row_to_model(self, row: dict[str, Any]) -> ConversationRoot:
        """Convert database row to domain model."""
        return ConversationRoot(
            entity_id=EntityId(value=row["entity_id"]),
            conversation_id=ConversationId(value=row["conversation_id"]),
            prompt_hash=ContentHash(digest=bytes(row["prompt_hash"])),
            instruction_version=row["instruction_version"],
            created_at=row["created_at"],
            geo_region=GeoRegion(row["geo_region"]),
            compliance_tier=ComplianceTier[row["compliance_tier"]],
            deleted_at=row.get("deleted_at"),
        )


class InstructionRepository:
    """Repository for instruction_sets table."""
    
    __slots__ = ("_engine",)
    
    def __init__(self, engine: CPEngine) -> None:
        self._engine = engine
    
    async def insert(
        self,
        instruction: InstructionSet,
    ) -> Result[InstructionSet, StorageError]:
        """Insert new instruction version."""
        query = """
            INSERT INTO instruction_sets (
                instruction_id, version, instruction_hash,
                template_structure, name, description
            ) VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *;
        """
        
        result = await self._engine.execute(
            query,
            instruction.instruction_id.value,
            instruction.version,
            instruction.instruction_hash.digest,
            json.dumps(instruction.template_structure),
            instruction.name,
            instruction.description,
        )
        
        if result.is_err():
            return result
        
        return Ok(instruction)
    
    async def get_latest(
        self,
        instruction_id: InstructionId,
    ) -> Result[Optional[InstructionSet], StorageError]:
        """Get latest version of instruction set."""
        query = """
            SELECT * FROM instruction_sets
            WHERE instruction_id = $1
            ORDER BY version DESC
            LIMIT 1;
        """
        
        result = await self._engine.execute(query, instruction_id.value)
        
        if result.is_err():
            return result
        
        rows = result.unwrap().rows
        if not rows:
            return Ok(None)
        
        return Ok(self._row_to_model(rows[0]))
    
    async def get_by_version(
        self,
        instruction_id: InstructionId,
        version: int,
    ) -> Result[Optional[InstructionSet], StorageError]:
        """Get specific version of instruction set."""
        query = """
            SELECT * FROM instruction_sets
            WHERE instruction_id = $1 AND version = $2;
        """
        
        result = await self._engine.execute(
            query,
            instruction_id.value,
            version,
        )
        
        if result.is_err():
            return result
        
        rows = result.unwrap().rows
        if not rows:
            return Ok(None)
        
        return Ok(self._row_to_model(rows[0]))
    
    async def list_versions(
        self,
        instruction_id: InstructionId,
    ) -> Result[list[InstructionSet], StorageError]:
        """List all versions of instruction set."""
        query = """
            SELECT * FROM instruction_sets
            WHERE instruction_id = $1
            ORDER BY version DESC;
        """
        
        result = await self._engine.execute(query, instruction_id.value)
        
        if result.is_err():
            return result
        
        return Ok([self._row_to_model(r) for r in result.unwrap().rows])
    
    def _row_to_model(self, row: dict[str, Any]) -> InstructionSet:
        """Convert database row to domain model."""
        template = row["template_structure"]
        if isinstance(template, str):
            template = json.loads(template)
        
        return InstructionSet(
            instruction_id=InstructionId(value=row["instruction_id"]),
            version=row["version"],
            instruction_hash=ContentHash(digest=bytes(row["instruction_hash"])),
            template_structure=template,
            name=row["name"],
            description=row.get("description"),
            created_at=row["created_at"],
        )


class IdempotencyRepository:
    """Repository for idempotency key management."""
    
    __slots__ = ("_engine", "_ttl_hours")
    
    def __init__(self, engine: CPEngine, ttl_hours: int = 24) -> None:
        self._engine = engine
        self._ttl_hours = ttl_hours
    
    async def check_or_set(
        self,
        key: str,
        request_hash: ContentHash,
        entity_id: Optional[EntityId] = None,
    ) -> Result[Optional[dict[str, Any]], IngestionError]:
        """
        Check if key exists; if not, insert it.
        
        Returns:
            None if key was inserted (new request)
            Existing result if key exists (duplicate)
        """
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=self._ttl_hours)
        
        # Try insert first (optimistic)
        insert_query = """
            INSERT INTO idempotency_keys (
                idempotency_key, result_status, request_hash,
                expires_at, entity_id
            ) VALUES ($1, 'PENDING', $2, $3, $4)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING idempotency_key;
        """
        
        result = await self._engine.execute(
            insert_query,
            key,
            request_hash.digest,
            expires,
            entity_id.value if entity_id else None,
        )
        
        if result.is_err():
            return Err(IngestionError.validation_failed(
                field="idempotency_key",
                value=key,
                reason="Database error",
            ))
        
        # If insert succeeded, this is a new request
        if result.unwrap().rows:
            return Ok(None)
        
        # Key exists, fetch existing result
        select_query = """
            SELECT result_status, result_payload, created_at
            FROM idempotency_keys
            WHERE idempotency_key = $1;
        """
        
        result = await self._engine.execute(select_query, key)
        
        if result.is_err():
            return Err(IngestionError.validation_failed(
                field="idempotency_key",
                value=key,
                reason="Database error",
            ))
        
        rows = result.unwrap().rows
        if rows:
            return Ok(rows[0])
        
        return Ok(None)
    
    async def complete(
        self,
        key: str,
        status: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> Result[None, StorageError]:
        """Mark idempotency key as completed."""
        query = """
            UPDATE idempotency_keys
            SET result_status = $1, result_payload = $2
            WHERE idempotency_key = $3;
        """
        
        result = await self._engine.execute(
            query,
            status,
            json.dumps(payload) if payload else None,
            key,
        )
        
        return Ok(None) if result.is_ok() else result
    
    async def cleanup_expired(self) -> Result[int, StorageError]:
        """Delete expired idempotency keys."""
        query = """
            DELETE FROM idempotency_keys
            WHERE expires_at < NOW();
        """
        
        result = await self._engine.execute(query)
        return Ok(0) if result.is_ok() else result
