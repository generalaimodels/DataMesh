"""
AP Repositories: Data Access Layer for SQLite Content Storage

Provides:
- ResponseRepository: CRUD for response_dataframes
- EmbeddingRepository: Vector storage and search
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from uuid import UUID

from datamesh.core.types import (
    Result, Ok, Err,
    EntityId, ConversationId,
    EmbeddingVector, PaginationCursor,
)
from datamesh.core.errors import StorageError
from datamesh.storage.ap.engine import APEngine


@dataclass(frozen=True, slots=True)
class ResponseDataframe:
    """Domain model for response_dataframes table."""
    
    entity_id: EntityId
    conversation_id: ConversationId
    sequence_id: int
    payload_ref: Optional[str]
    embedding: Optional[EmbeddingVector]
    token_length: int
    quality_score: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """Domain model for embeddings table."""
    
    id: str
    vector: EmbeddingVector
    source_type: str
    source_id: str


class ResponseRepository:
    """Repository for response_dataframes table."""
    
    __slots__ = ("_engine",)
    
    def __init__(self, engine: APEngine) -> None:
        self._engine = engine
    
    async def insert(
        self,
        response: ResponseDataframe,
    ) -> Result[ResponseDataframe, StorageError]:
        """Insert new response dataframe."""
        embedding_blob = None
        if response.embedding:
            embedding_blob = response.embedding.data
        
        sql = """
            INSERT OR REPLACE INTO response_dataframes (
                entity_id, conversation_id, sequence_id,
                payload_ref, embedding, token_length,
                quality_score, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        result = await self._engine.execute_sql(
            sql,
            (
                str(response.entity_id),
                str(response.conversation_id),
                response.sequence_id,
                response.payload_ref,
                embedding_blob,
                response.token_length,
                response.quality_score,
                json.dumps(response.metadata),
            ),
        )
        
        if result.is_err():
            return result
        
        return Ok(response)
    
    async def insert_batch(
        self,
        responses: Sequence[ResponseDataframe],
    ) -> Result[int, StorageError]:
        """Batch insert responses."""
        if not responses:
            return Ok(0)
        
        for resp in responses:
            result = await self.insert(resp)
            if result.is_err():
                return result
        
        return Ok(len(responses))
    
    async def get_by_id(
        self,
        entity_id: EntityId,
        conversation_id: ConversationId,
        sequence_id: int,
    ) -> Result[Optional[ResponseDataframe], StorageError]:
        """Get response by composite key."""
        sql = """
            SELECT * FROM response_dataframes
            WHERE entity_id = ? AND conversation_id = ? AND sequence_id = ?
        """
        
        result = await self._engine.execute_sql(
            sql,
            (str(entity_id), str(conversation_id), sequence_id),
        )
        
        if result.is_err():
            return result
        
        rows = result.unwrap()
        if not rows:
            return Ok(None)
        
        return Ok(self._row_to_model(rows[0]))
    
    async def list_by_conversation(
        self,
        entity_id: EntityId,
        conversation_id: ConversationId,
        cursor: Optional[PaginationCursor] = None,
        limit: int = 100,
    ) -> Result[tuple[list[ResponseDataframe], Optional[PaginationCursor]], StorageError]:
        """List responses for conversation with pagination."""
        if cursor:
            sql = """
                SELECT * FROM response_dataframes
                WHERE entity_id = ? AND conversation_id = ?
                AND sequence_id > ?
                ORDER BY sequence_id
                LIMIT ?
            """
            params = (
                str(entity_id),
                str(conversation_id),
                cursor.sequence_id,
                limit + 1,
            )
        else:
            sql = """
                SELECT * FROM response_dataframes
                WHERE entity_id = ? AND conversation_id = ?
                ORDER BY sequence_id
                LIMIT ?
            """
            params = (str(entity_id), str(conversation_id), limit + 1)
        
        result = await self._engine.execute_sql(sql, params)
        
        if result.is_err():
            return result
        
        rows = result.unwrap()
        has_more = len(rows) > limit
        items = [self._row_to_model(r) for r in rows[:limit]]
        
        next_cursor = None
        if has_more and items:
            last = items[-1]
            next_cursor = PaginationCursor(
                conversation_id=last.conversation_id.value,
                sequence_id=last.sequence_id,
            )
        
        return Ok((items, next_cursor))
    
    async def list_by_quality(
        self,
        min_score: float = 0.0,
        limit: int = 100,
    ) -> Result[list[ResponseDataframe], StorageError]:
        """List responses ordered by quality score."""
        sql = """
            SELECT * FROM response_dataframes
            WHERE quality_score >= ?
            ORDER BY quality_score DESC
            LIMIT ?
        """
        
        result = await self._engine.execute_sql(sql, (min_score, limit))
        
        if result.is_err():
            return result
        
        return Ok([self._row_to_model(r) for r in result.unwrap()])
    
    async def delete(
        self,
        entity_id: EntityId,
        conversation_id: ConversationId,
        sequence_id: int,
    ) -> Result[bool, StorageError]:
        """Delete response."""
        sql = """
            DELETE FROM response_dataframes
            WHERE entity_id = ? AND conversation_id = ? AND sequence_id = ?
        """
        
        result = await self._engine.execute_sql(
            sql,
            (str(entity_id), str(conversation_id), sequence_id),
        )
        
        return Ok(True) if result.is_ok() else result
    
    def _row_to_model(self, row: dict[str, Any]) -> ResponseDataframe:
        """Convert row to domain model."""
        embedding = None
        if row.get("embedding"):
            embedding = EmbeddingVector(
                dimensions=1536,
                data=bytes(row["embedding"]),
            )
        
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        return ResponseDataframe(
            entity_id=EntityId(value=UUID(row["entity_id"])),
            conversation_id=ConversationId(value=UUID(row["conversation_id"])),
            sequence_id=row["sequence_id"],
            payload_ref=row.get("payload_ref"),
            embedding=embedding,
            token_length=row.get("token_length", 0),
            quality_score=row.get("quality_score", 0.0),
            metadata=metadata,
        )


class EmbeddingRepository:
    """Repository for vector embeddings with similarity search."""
    
    __slots__ = ("_engine",)
    
    def __init__(self, engine: APEngine) -> None:
        self._engine = engine
    
    async def insert(
        self,
        record: EmbeddingRecord,
    ) -> Result[EmbeddingRecord, StorageError]:
        """Insert embedding vector."""
        sql = """
            INSERT OR REPLACE INTO embeddings (
                id, vector, dimensions, source_type, source_id
            ) VALUES (?, ?, ?, ?, ?)
        """
        
        result = await self._engine.execute_sql(
            sql,
            (
                record.id,
                record.vector.data,
                record.vector.dimensions,
                record.source_type,
                record.source_id,
            ),
        )
        
        if result.is_err():
            return result
        
        return Ok(record)
    
    async def get(self, id: str) -> Result[Optional[EmbeddingRecord], StorageError]:
        """Get embedding by ID."""
        sql = "SELECT * FROM embeddings WHERE id = ?"
        
        result = await self._engine.execute_sql(sql, (id,))
        
        if result.is_err():
            return result
        
        rows = result.unwrap()
        if not rows:
            return Ok(None)
        
        return Ok(self._row_to_model(rows[0]))
    
    async def similarity_search(
        self,
        query_vector: EmbeddingVector,
        limit: int = 10,
    ) -> Result[list[tuple[EmbeddingRecord, float]], StorageError]:
        """
        Find similar vectors using cosine similarity.
        
        Note: For production, use pgvector or dedicated vector DB.
        This is a naive O(n) scan for demonstration.
        """
        sql = "SELECT * FROM embeddings"
        
        result = await self._engine.execute_sql(sql)
        
        if result.is_err():
            return result
        
        # Compute similarities (naive O(n) - for demo only)
        scored: list[tuple[EmbeddingRecord, float]] = []
        
        for row in result.unwrap():
            record = self._row_to_model(row)
            similarity = query_vector.cosine_similarity(record.vector)
            scored.append((record, similarity))
        
        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return Ok(scored[:limit])
    
    def _row_to_model(self, row: dict[str, Any]) -> EmbeddingRecord:
        """Convert row to domain model."""
        return EmbeddingRecord(
            id=row["id"],
            vector=EmbeddingVector(
                dimensions=row["dimensions"],
                data=bytes(row["vector"]),
            ),
            source_type=row["source_type"],
            source_id=row["source_id"],
        )
