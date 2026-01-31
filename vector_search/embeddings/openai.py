"""
OpenAI Embeddings: text-embedding-3-* Integration

Supports:
    - text-embedding-3-small (1536 dims, fast)
    - text-embedding-3-large (3072 dims, highest quality)
    - text-embedding-ada-002 (1536 dims, legacy)

Features:
    - Automatic rate limit handling with retry
    - Batch optimization (up to 2048 texts)
    - Async support via openai.AsyncOpenAI
"""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence

from vector_search.core.errors import Err, Ok, Result
from vector_search.core.types import EmbeddingVector
from vector_search.embeddings.base import BaseEmbedder


# Model dimensions mapping
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embeddings integration.
    
    Example:
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        result = embedder.embed("Hello world")
        if result.is_ok():
            vector = result.unwrap()
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize OpenAI embedder.
        
        Args:
            model: Model name
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API endpoint
            dimensions: Override output dimensions (for v3 models)
            max_retries: Max retries on rate limit
        """
        # Import at init to make dependency optional
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI SDK not installed. Run: pip install vectorsearch[openai]"
            )
        
        # Determine dimension
        dim = dimensions or MODEL_DIMENSIONS.get(model, 1536)
        super().__init__(model_name=model, dimension=dim)
        
        # Create client
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )
        self._async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )
        self._dimensions = dimensions
    
    def embed(self, text: str) -> Result[EmbeddingVector, str]:
        """Embed single text."""
        result = self.embed_batch([text])
        if result.is_err():
            return Err(result.error)  # type: ignore
        return Ok(result.unwrap()[0])
    
    def embed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """
        Embed batch of texts.
        
        OpenAI supports up to 2048 texts per request.
        """
        try:
            # Build request params
            params = {
                "model": self._model_name,
                "input": list(texts),
            }
            if self._dimensions:
                params["dimensions"] = self._dimensions
            
            # Call API
            response = self._client.embeddings.create(**params)
            
            # Extract embeddings (sorted by index)
            embeddings = sorted(response.data, key=lambda x: x.index)
            vectors = [
                EmbeddingVector.from_list(e.embedding)
                for e in embeddings
            ]
            
            return Ok(vectors)
            
        except Exception as e:
            return Err(f"OpenAI embedding failed: {str(e)}")
    
    async def aembed(self, text: str) -> Result[EmbeddingVector, str]:
        """Async embed single text."""
        result = await self.aembed_batch([text])
        if result.is_err():
            return Err(result.error)  # type: ignore
        return Ok(result.unwrap()[0])
    
    async def aembed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """Async embed batch of texts."""
        try:
            params = {
                "model": self._model_name,
                "input": list(texts),
            }
            if self._dimensions:
                params["dimensions"] = self._dimensions
            
            response = await self._async_client.embeddings.create(**params)
            
            embeddings = sorted(response.data, key=lambda x: x.index)
            vectors = [
                EmbeddingVector.from_list(e.embedding)
                for e in embeddings
            ]
            
            return Ok(vectors)
            
        except Exception as e:
            return Err(f"OpenAI embedding failed: {str(e)}")
