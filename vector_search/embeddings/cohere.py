"""
Cohere Embeddings: embed-v3 Integration

Supports:
    - embed-english-v3.0 (1024 dims)
    - embed-multilingual-v3.0 (1024 dims)
    - embed-english-light-v3.0 (384 dims, faster)

Features:
    - Input type optimization (search_document, search_query)
    - Automatic rate limit handling
    - Batch support (up to 96 texts)
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from vector_search.core.errors import Err, Ok, Result
from vector_search.core.types import EmbeddingVector
from vector_search.embeddings.base import BaseEmbedder


MODEL_DIMENSIONS = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}


class CohereEmbedder(BaseEmbedder):
    """
    Cohere embeddings integration.
    
    Example:
        embedder = CohereEmbedder(model="embed-english-v3.0")
        result = embedder.embed("Hello world")
    """
    
    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        input_type: Literal["search_document", "search_query"] = "search_document",
    ) -> None:
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere SDK not installed. Run: pip install vectorsearch[cohere]"
            )
        
        dim = MODEL_DIMENSIONS.get(model, 1024)
        super().__init__(model_name=model, dimension=dim)
        
        self._client = cohere.Client(api_key=api_key)
        self._async_client = cohere.AsyncClient(api_key=api_key)
        self._input_type = input_type
    
    def embed(self, text: str) -> Result[EmbeddingVector, str]:
        result = self.embed_batch([text])
        if result.is_err():
            return Err(result.error)  # type: ignore
        return Ok(result.unwrap()[0])
    
    def embed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        try:
            response = self._client.embed(
                model=self._model_name,
                texts=list(texts),
                input_type=self._input_type,
            )
            
            vectors = [
                EmbeddingVector.from_list(emb)
                for emb in response.embeddings
            ]
            return Ok(vectors)
            
        except Exception as e:
            return Err(f"Cohere embedding failed: {str(e)}")
    
    async def aembed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        try:
            response = await self._async_client.embed(
                model=self._model_name,
                texts=list(texts),
                input_type=self._input_type,
            )
            
            vectors = [
                EmbeddingVector.from_list(emb)
                for emb in response.embeddings
            ]
            return Ok(vectors)
            
        except Exception as e:
            return Err(f"Cohere embedding failed: {str(e)}")
