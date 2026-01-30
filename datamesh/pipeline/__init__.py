"""
Pipeline module: Ingestion, Saga pattern, and backpressure management.
"""

from datamesh.pipeline.ingestion import IngestionPipeline, IngestionRequest
from datamesh.pipeline.saga import SagaCoordinator, SagaStep, SagaState
from datamesh.pipeline.backpressure import BackpressureController
from datamesh.pipeline.deduplication import DeduplicationService

__all__ = [
    "IngestionPipeline",
    "IngestionRequest",
    "SagaCoordinator",
    "SagaStep",
    "SagaState",
    "BackpressureController",
    "DeduplicationService",
]
