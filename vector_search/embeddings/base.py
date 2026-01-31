"""
Base Embedder: Protocol and Abstract Base Class

Provides:
    - EmbedderProtocol: Structural interface for all embedders
    - BaseEmbedder: Abstract base with common functionality
    - MockEmbedder: Testing implementation with random vectors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from vector_search.core.errors import Ok, Result
from vector_search.core.types import EmbeddingVector


# =============================================================================
# EMBEDDER PROTOCOL
# =============================================================================
@runtime_checkable
class EmbedderProtocol(Protocol):
    """
    Protocol for embedding providers.
    
    Implementations must support:
        - Single text embedding (sync + async)
        - Batch text embedding (sync + async)
        - Dimension and model info
    """
    
    @property
    def dimension(self) -> int:
        """Output embedding dimension."""
        ...
    
    @property
    def model_name(self) -> str:
        """Model identifier."""
        ...
    
    def embed(self, text: str) -> Result[EmbeddingVector, str]:
        """Embed single text."""
        ...
    
    def embed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """Embed batch of texts."""
        ...
    
    async def aembed(self, text: str) -> Result[EmbeddingVector, str]:
        """Async embed single text."""
        ...
    
    async def aembed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """Async embed batch of texts."""
        ...


# =============================================================================
# BASE EMBEDDER
# =============================================================================
class BaseEmbedder(ABC):
    """
    Abstract base class for embedders.
    
    Provides:
        - Default batch implementation via single embed
        - Default async implementation via sync
        - Common validation and error handling
    """
    
    def __init__(self, model_name: str, dimension: int) -> None:
        self._model_name = model_name
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @abstractmethod
    def embed(self, text: str) -> Result[EmbeddingVector, str]:
        """Embed single text - must be implemented."""
        ...
    
    def embed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """Default batch implementation (override for efficiency)."""
        results: list[EmbeddingVector] = []
        for text in texts:
            result = self.embed(text)
            if result.is_err():
                return result  # type: ignore
            results.append(result.unwrap())
        return Ok(results)
    
    async def aembed(self, text: str) -> Result[EmbeddingVector, str]:
        """Default async implementation - runs sync in executor."""
        return self.embed(text)
    
    async def aembed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """Default async batch - runs sync in executor."""
        return self.embed_batch(texts)


# =============================================================================
# MOCK EMBEDDER (Testing)
# =============================================================================
class MockEmbedder(BaseEmbedder):
    """
    Mock embedder for testing.
    
    Generates deterministic random vectors based on text hash.
    Useful for unit tests without API dependencies.
    """
    
    def __init__(self, dimension: int = 768) -> None:
        super().__init__(model_name="mock", dimension=dimension)
        self._seed = 42
    
    def embed(self, text: str) -> Result[EmbeddingVector, str]:
        """Generate deterministic embedding from text hash."""
        # Deterministic RNG based on text
        rng = np.random.default_rng(hash(text) & 0xFFFFFFFF)
        vec = rng.standard_normal(self._dimension).astype(np.float32)
        
        # Normalize
        vec = vec / np.linalg.norm(vec)
        
        return Ok(EmbeddingVector.from_numpy(vec))
