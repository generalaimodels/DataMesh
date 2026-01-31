"""
LangChain Embeddings: Wrapper for LangChain Embeddings Interface

Provides bidirectional integration:
    1. Use any LangChain embedding in VectorSearch
    2. Use VectorSearch embedders in LangChain

This allows seamless integration with LangChain ecosystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from vector_search.core.errors import Err, Ok, Result
from vector_search.core.types import EmbeddingVector
from vector_search.embeddings.base import BaseEmbedder

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class LangChainEmbedder(BaseEmbedder):
    """
    Wrapper to use LangChain Embeddings in VectorSearch.
    
    Example:
        from langchain_openai import OpenAIEmbeddings
        
        lc_embeddings = OpenAIEmbeddings()
        embedder = LangChainEmbedder(lc_embeddings, dimension=1536)
        
        result = embedder.embed("Hello world")
    """
    
    def __init__(
        self,
        langchain_embeddings: "Embeddings",
        dimension: int,
        model_name: str = "langchain",
    ) -> None:
        """
        Initialize LangChain wrapper.
        
        Args:
            langchain_embeddings: LangChain Embeddings instance
            dimension: Output embedding dimension
            model_name: Model identifier for logging
        """
        super().__init__(model_name=model_name, dimension=dimension)
        self._lc_embeddings = langchain_embeddings
    
    def embed(self, text: str) -> Result[EmbeddingVector, str]:
        """Embed single text using LangChain."""
        try:
            embedding = self._lc_embeddings.embed_query(text)
            return Ok(EmbeddingVector.from_list(embedding))
        except Exception as e:
            return Err(f"LangChain embedding failed: {str(e)}")
    
    def embed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """Embed batch using LangChain."""
        try:
            embeddings = self._lc_embeddings.embed_documents(list(texts))
            vectors = [EmbeddingVector.from_list(emb) for emb in embeddings]
            return Ok(vectors)
        except Exception as e:
            return Err(f"LangChain embedding failed: {str(e)}")
    
    async def aembed(self, text: str) -> Result[EmbeddingVector, str]:
        """Async embed single text."""
        try:
            embedding = await self._lc_embeddings.aembed_query(text)
            return Ok(EmbeddingVector.from_list(embedding))
        except Exception as e:
            return Err(f"LangChain embedding failed: {str(e)}")
    
    async def aembed_batch(self, texts: Sequence[str]) -> Result[list[EmbeddingVector], str]:
        """Async embed batch."""
        try:
            embeddings = await self._lc_embeddings.aembed_documents(list(texts))
            vectors = [EmbeddingVector.from_list(emb) for emb in embeddings]
            return Ok(vectors)
        except Exception as e:
            return Err(f"LangChain embedding failed: {str(e)}")


class VectorSearchEmbeddings:
    """
    Adapter to use VectorSearch embedders in LangChain.
    
    Implements LangChain's Embeddings interface.
    
    Example:
        from vector_search.embeddings import OpenAIEmbedder
        
        vs_embedder = OpenAIEmbedder()
        lc_embeddings = VectorSearchEmbeddings(vs_embedder)
        
        # Use in LangChain
        from langchain.vectorstores import FAISS
        vectorstore = FAISS.from_texts(texts, lc_embeddings)
    """
    
    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed list of documents."""
        result = self._embedder.embed_batch(texts)
        if result.is_err():
            raise RuntimeError(result.error)
        return [v.to_list() for v in result.unwrap()]
    
    def embed_query(self, text: str) -> list[float]:
        """Embed single query."""
        result = self._embedder.embed(text)
        if result.is_err():
            raise RuntimeError(result.error)
        return result.unwrap().to_list()
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async embed documents."""
        result = await self._embedder.aembed_batch(texts)
        if result.is_err():
            raise RuntimeError(result.error)
        return [v.to_list() for v in result.unwrap()]
    
    async def aembed_query(self, text: str) -> list[float]:
        """Async embed query."""
        result = await self._embedder.aembed(text)
        if result.is_err():
            raise RuntimeError(result.error)
        return result.unwrap().to_list()
