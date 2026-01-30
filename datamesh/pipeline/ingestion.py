"""
High-Velocity Ingestion Pipeline

Orchestrates the complete write path:
1. Rate limiting and backpressure
2. Idempotency check
3. Validation
4. Saga execution
5. Response

Optimized for million-row-per-second throughput.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence
from uuid import UUID

from datamesh.core.types import (
    Result, Ok, Err,
    EntityId, ConversationId, InstructionId,
    ContentHash, Timestamp, ComplianceTier, GeoRegion,
    EmbeddingVector,
)
from datamesh.core.errors import IngestionError, StorageError
from datamesh.core.config import DataMeshConfig
from datamesh.storage.cp.engine import CPEngine
from datamesh.storage.cp.repositories import (
    ConversationRepository, ConversationRoot,
    IdempotencyRepository,
)
from datamesh.storage.ap.engine import APEngine
from datamesh.storage.ap.repositories import ResponseRepository, ResponseDataframe
from datamesh.storage.object_store.cas import ContentAddressableStorage
from datamesh.pipeline.saga import (
    SagaCoordinator, InsertMetadataStep, WritePayloadStep, InsertVectorsStep,
)
from datamesh.pipeline.backpressure import BackpressureController
from datamesh.pipeline.deduplication import DeduplicationService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionRequest:
    """Incoming ingestion request."""
    entity_id: EntityId
    prompt: bytes
    response: bytes
    instruction_version: int = 1
    geo_region: GeoRegion = GeoRegion.US_EAST
    compliance_tier: ComplianceTier = ComplianceTier.STANDARD
    embedding: Optional[EmbeddingVector] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class IngestionResponse:
    """Ingestion result."""
    conversation_id: ConversationId
    prompt_hash: ContentHash
    response_hash: ContentHash
    sequence_id: int
    latency_ms: float
    deduplicated: bool = False


@dataclass
class PipelineStats:
    """Pipeline statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    deduplicated_requests: int = 0
    total_bytes_ingested: int = 0
    avg_latency_ms: float = 0.0


class IngestionPipeline:
    """
    High-velocity ingestion pipeline.
    
    Components:
    - BackpressureController: Rate limiting
    - DeduplicationService: Idempotency
    - SagaCoordinator: Distributed transactions
    
    Usage:
        pipeline = await IngestionPipeline.create(config)
        
        result = await pipeline.ingest(request)
    """
    
    __slots__ = (
        "_config", "_cp_engine", "_ap_engine", "_cas",
        "_conv_repo", "_resp_repo", "_idem_repo",
        "_backpressure", "_dedup", "_stats", "_closed",
    )
    
    def __init__(
        self,
        config: DataMeshConfig,
        cp_engine: CPEngine,
        ap_engine: APEngine,
        cas: ContentAddressableStorage,
    ) -> None:
        self._config = config
        self._cp_engine = cp_engine
        self._ap_engine = ap_engine
        self._cas = cas
        
        self._conv_repo = ConversationRepository(cp_engine)
        self._resp_repo = ResponseRepository(ap_engine)
        self._idem_repo = IdempotencyRepository(cp_engine)
        
        self._backpressure = BackpressureController()
        self._dedup = DeduplicationService(self._idem_repo)
        
        self._stats = PipelineStats()
        self._closed = False
    
    @classmethod
    async def create(cls, config: DataMeshConfig) -> Result[IngestionPipeline, StorageError]:
        """Create and initialize pipeline."""
        # Initialize CP engine
        cp_result = await CPEngine.create(config.cp)
        if cp_result.is_err():
            return cp_result
        cp_engine = cp_result.unwrap()
        
        # Initialize AP engine
        ap_engine = APEngine(config.ap)
        ap_result = await ap_engine.initialize()
        if ap_result.is_err():
            await cp_engine.close()
            return ap_result
        
        # Initialize CAS
        cas = ContentAddressableStorage(config.object_store)
        
        return Ok(cls(config, cp_engine, ap_engine, cas))
    
    async def ingest(
        self,
        request: IngestionRequest,
    ) -> Result[IngestionResponse, IngestionError]:
        """
        Ingest single request through pipeline.
        
        Steps:
        1. Acquire backpressure permit
        2. Check idempotency
        3. Validate request
        4. Execute saga
        5. Return response
        """
        start = Timestamp.now()
        self._stats.total_requests += 1
        
        try:
            # 1. Backpressure check
            bp_result = await self._backpressure.acquire()
            if bp_result.is_err():
                self._stats.failed_requests += 1
                return bp_result
            
            try:
                # 2. Idempotency check
                if request.idempotency_key:
                    request_bytes = self._serialize_request(request)
                    dedup_result = await self._dedup.check(
                        request.idempotency_key,
                        request_bytes,
                        request.entity_id,
                    )
                    
                    if dedup_result.is_ok() and dedup_result.unwrap().is_duplicate:
                        self._stats.deduplicated_requests += 1
                        original = dedup_result.unwrap().original_response
                        
                        latency = (Timestamp.now() - start) / 1_000_000
                        
                        return Ok(IngestionResponse(
                            conversation_id=ConversationId.from_string(
                                original.get("conversation_id", "")
                            ).unwrap_or(ConversationId.generate()),
                            prompt_hash=ContentHash.compute(request.prompt),
                            response_hash=ContentHash.compute(request.response),
                            sequence_id=original.get("sequence_id", 0),
                            latency_ms=latency,
                            deduplicated=True,
                        ))
                
                # 3. Validate
                validation = self._validate(request)
                if validation.is_err():
                    self._stats.failed_requests += 1
                    return validation
                
                # 4. Prepare data models
                conversation_id = ConversationId.generate()
                prompt_hash = ContentHash.compute(request.prompt)
                response_hash = ContentHash.compute(request.response)
                
                conv_root = ConversationRoot(
                    entity_id=request.entity_id,
                    conversation_id=conversation_id,
                    prompt_hash=prompt_hash,
                    instruction_version=request.instruction_version,
                    created_at=datetime.now(timezone.utc),
                    geo_region=request.geo_region,
                    compliance_tier=request.compliance_tier,
                )
                
                response_df = ResponseDataframe(
                    entity_id=request.entity_id,
                    conversation_id=conversation_id,
                    sequence_id=0,
                    payload_ref=None,  # Set after CAS put
                    embedding=request.embedding,
                    token_length=len(request.response),
                    quality_score=0.0,
                    metadata=request.metadata,
                )
                
                # 5. Execute saga
                saga = SagaCoordinator()
                saga.add_step(InsertMetadataStep(self._conv_repo))
                saga.add_step(WritePayloadStep(self._cas))
                saga.add_step(InsertVectorsStep(self._resp_repo))
                
                context = {
                    "conversation_root": conv_root,
                    "response_dataframe": response_df,
                    "payload": request.response,
                }
                
                saga_result = await saga.execute(context)
                
                if saga_result.is_err():
                    self._stats.failed_requests += 1
                    return saga_result
                
                # 6. Record completion
                if request.idempotency_key:
                    await self._dedup.complete(
                        request.idempotency_key,
                        {
                            "conversation_id": str(conversation_id),
                            "sequence_id": 0,
                        },
                    )
                
                # 7. Update stats
                latency = (Timestamp.now() - start) / 1_000_000
                self._stats.successful_requests += 1
                self._stats.total_bytes_ingested += len(request.prompt) + len(request.response)
                self._update_latency(latency)
                
                return Ok(IngestionResponse(
                    conversation_id=conversation_id,
                    prompt_hash=prompt_hash,
                    response_hash=response_hash,
                    sequence_id=0,
                    latency_ms=latency,
                ))
                
            finally:
                await self._backpressure.release()
                
        except Exception as e:
            self._stats.failed_requests += 1
            logger.error(f"Ingestion failed: {e}")
            return Err(IngestionError.validation_failed(
                field="request",
                value="",
                reason=str(e),
            ))
    
    async def ingest_batch(
        self,
        requests: Sequence[IngestionRequest],
    ) -> list[Result[IngestionResponse, IngestionError]]:
        """Ingest multiple requests in parallel."""
        tasks = [self.ingest(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    def _validate(self, request: IngestionRequest) -> Result[None, IngestionError]:
        """Validate ingestion request."""
        if not request.prompt:
            return Err(IngestionError.validation_failed(
                field="prompt",
                value="",
                reason="Prompt cannot be empty",
            ))
        
        if len(request.prompt) > 100 * 1024 * 1024:  # 100 MB
            return Err(IngestionError.validation_failed(
                field="prompt",
                value=f"{len(request.prompt)} bytes",
                reason="Prompt exceeds maximum size",
            ))
        
        return Ok(None)
    
    def _serialize_request(self, request: IngestionRequest) -> bytes:
        """Serialize request for hashing."""
        return json.dumps({
            "entity_id": str(request.entity_id),
            "prompt_hash": ContentHash.compute(request.prompt).to_hex(),
            "instruction_version": request.instruction_version,
        }, sort_keys=True).encode()
    
    def _update_latency(self, latency_ms: float) -> None:
        """Update exponential moving average latency."""
        alpha = 0.1
        self._stats.avg_latency_ms = (
            alpha * latency_ms +
            (1 - alpha) * self._stats.avg_latency_ms
        )
    
    @property
    def stats(self) -> PipelineStats:
        return self._stats
    
    @property
    def backpressure_metrics(self):
        return self._backpressure.metrics
    
    async def close(self) -> None:
        """Close pipeline and release resources."""
        if self._closed:
            return
        
        self._closed = True
        await self._cp_engine.close()
        await self._ap_engine.close()
        
        logger.info("Ingestion pipeline closed")
    
    async def __aenter__(self) -> IngestionPipeline:
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        await self.close()
