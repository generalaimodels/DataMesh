"""
Saga Pattern: Distributed Transaction Coordination

Implements Saga pattern for multi-step operations:
T1: Insert metadata → CP tier (PostgreSQL)
T2: Write payload → Object Store (S3/FS)
T3: Insert vectors → AP tier (SQLite)
T4: Publish indexing event (optional)

Each step has execute() and compensate() for atomic rollback.
Avoids 2PC blocking while maintaining eventual consistency.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, Optional, TypeVar
from uuid import uuid4

from datamesh.core.types import Result, Ok, Err, Timestamp
from datamesh.core.errors import IngestionError

logger = logging.getLogger(__name__)


class SagaState(Enum):
    """Saga execution state."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    COMPENSATING = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


T = TypeVar("T")  # Step result type


@dataclass
class StepResult(Generic[T]):
    """Result of a saga step execution."""
    step_name: str
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    duration_ns: int = 0


class SagaStep(ABC, Generic[T]):
    """
    Abstract saga step with execute and compensate.
    
    Subclass this to implement specific operations.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Step identifier for logging."""
        pass
    
    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> Result[T, Exception]:
        """
        Execute forward operation.
        
        Args:
            context: Shared saga context (mutable)
            
        Returns:
            Result with step output or error
        """
        pass
    
    @abstractmethod
    async def compensate(self, context: dict[str, Any]) -> Result[None, Exception]:
        """
        Compensating transaction for rollback.
        
        Called when a later step fails.
        Must be idempotent (safe to call multiple times).
        """
        pass


@dataclass
class SagaExecution:
    """Saga execution record."""
    saga_id: str
    state: SagaState
    current_step: int
    context: dict[str, Any]
    step_results: list[StepResult]
    started_at: Timestamp
    completed_at: Optional[Timestamp] = None
    error: Optional[str] = None


class SagaCoordinator:
    """
    Coordinates saga execution with automatic compensation.
    
    Usage:
        coordinator = SagaCoordinator()
        
        coordinator.add_step(InsertMetadataStep())
        coordinator.add_step(WritePayloadStep())
        coordinator.add_step(InsertVectorsStep())
        
        result = await coordinator.execute(initial_context)
    """
    
    __slots__ = ("_steps", "_max_retries", "_retry_delay_ms")
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_ms: int = 100,
    ) -> None:
        self._steps: list[SagaStep] = []
        self._max_retries = max_retries
        self._retry_delay_ms = retry_delay_ms
    
    def add_step(self, step: SagaStep) -> SagaCoordinator:
        """Add step to saga (fluent API)."""
        self._steps.append(step)
        return self
    
    async def execute(
        self,
        context: Optional[dict[str, Any]] = None,
    ) -> Result[SagaExecution, IngestionError]:
        """
        Execute saga with automatic compensation on failure.
        
        Args:
            context: Initial context passed to all steps
            
        Returns:
            SagaExecution with final state and results
        """
        saga_id = str(uuid4())
        execution = SagaExecution(
            saga_id=saga_id,
            state=SagaState.RUNNING,
            current_step=0,
            context=context or {},
            step_results=[],
            started_at=Timestamp.now(),
        )
        
        logger.info(f"Starting saga {saga_id} with {len(self._steps)} steps")
        
        completed_steps: list[SagaStep] = []
        
        for i, step in enumerate(self._steps):
            execution.current_step = i
            
            # Execute with retry
            result = await self._execute_step_with_retry(step, execution.context)
            
            step_result = StepResult(
                step_name=step.name,
                success=result.is_ok(),
                result=result.unwrap() if result.is_ok() else None,
                error=result.error if result.is_err() else None,
            )
            execution.step_results.append(step_result)
            
            if result.is_ok():
                completed_steps.append(step)
                logger.debug(f"Saga {saga_id}: step '{step.name}' completed")
            else:
                # Step failed - compensate
                logger.warning(f"Saga {saga_id}: step '{step.name}' failed: {result.error}")
                
                execution.state = SagaState.COMPENSATING
                await self._compensate(completed_steps, execution.context)
                
                execution.state = SagaState.ROLLED_BACK
                execution.completed_at = Timestamp.now()
                execution.error = str(result.error)
                
                return Err(IngestionError.saga_failed(
                    saga_id=saga_id,
                    failed_step=step.name,
                    compensated_steps=[s.name for s in completed_steps],
                    cause=result.error,
                ))
        
        execution.state = SagaState.COMPLETED
        execution.completed_at = Timestamp.now()
        
        logger.info(f"Saga {saga_id} completed successfully")
        return Ok(execution)
    
    async def _execute_step_with_retry(
        self,
        step: SagaStep,
        context: dict[str, Any],
    ) -> Result[Any, Exception]:
        """Execute step with exponential backoff retry."""
        last_error: Optional[Exception] = None
        
        for attempt in range(self._max_retries + 1):
            try:
                result = await step.execute(context)
                if result.is_ok():
                    return result
                last_error = result.error
            except Exception as e:
                last_error = e
            
            if attempt < self._max_retries:
                delay = self._retry_delay_ms * (2 ** attempt) / 1000
                await asyncio.sleep(delay)
        
        return Err(last_error or Exception("Unknown error"))
    
    async def _compensate(
        self,
        completed_steps: list[SagaStep],
        context: dict[str, Any],
    ) -> None:
        """Run compensating transactions in reverse order."""
        for step in reversed(completed_steps):
            try:
                await step.compensate(context)
                logger.debug(f"Compensated step '{step.name}'")
            except Exception as e:
                # Log but continue - compensation must complete
                logger.error(f"Compensation failed for '{step.name}': {e}")


# =============================================================================
# CONCRETE SAGA STEPS FOR DATA MESH
# =============================================================================
class InsertMetadataStep(SagaStep[str]):
    """Step 1: Insert metadata to CP tier."""
    
    def __init__(self, cp_repo) -> None:
        self._repo = cp_repo
    
    @property
    def name(self) -> str:
        return "insert_metadata"
    
    async def execute(self, context: dict[str, Any]) -> Result[str, Exception]:
        try:
            root = context["conversation_root"]
            result = await self._repo.insert(root)
            
            if result.is_ok():
                context["metadata_inserted"] = True
                return Ok(str(root.conversation_id))
            return Err(Exception("Insert failed"))
        except Exception as e:
            return Err(e)
    
    async def compensate(self, context: dict[str, Any]) -> Result[None, Exception]:
        if context.get("metadata_inserted"):
            root = context["conversation_root"]
            await self._repo.soft_delete(root.entity_id, root.conversation_id)
        return Ok(None)


class WritePayloadStep(SagaStep[str]):
    """Step 2: Write payload to object storage."""
    
    def __init__(self, cas) -> None:
        self._cas = cas
    
    @property
    def name(self) -> str:
        return "write_payload"
    
    async def execute(self, context: dict[str, Any]) -> Result[str, Exception]:
        try:
            payload = context.get("payload", b"")
            result = await self._cas.put(payload)
            
            if result.is_ok():
                cas_obj = result.unwrap()
                context["payload_hash"] = cas_obj.hash
                return Ok(cas_obj.hash.to_hex())
            return Err(Exception("CAS put failed"))
        except Exception as e:
            return Err(e)
    
    async def compensate(self, context: dict[str, Any]) -> Result[None, Exception]:
        payload_hash = context.get("payload_hash")
        if payload_hash:
            await self._cas.delete(payload_hash)
        return Ok(None)


class InsertVectorsStep(SagaStep[int]):
    """Step 3: Insert vectors to AP tier."""
    
    def __init__(self, ap_repo) -> None:
        self._repo = ap_repo
    
    @property
    def name(self) -> str:
        return "insert_vectors"
    
    async def execute(self, context: dict[str, Any]) -> Result[int, Exception]:
        try:
            response = context.get("response_dataframe")
            if response:
                result = await self._repo.insert(response)
                if result.is_ok():
                    context["vectors_inserted"] = True
                    return Ok(1)
            return Ok(0)
        except Exception as e:
            return Err(e)
    
    async def compensate(self, context: dict[str, Any]) -> Result[None, Exception]:
        if context.get("vectors_inserted"):
            response = context["response_dataframe"]
            await self._repo.delete(
                response.entity_id,
                response.conversation_id,
                response.sequence_id,
            )
        return Ok(None)
