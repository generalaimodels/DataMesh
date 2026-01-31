"""
Embeddings Module: Multi-Provider SDK Integrations

Provides unified interface for embedding providers:
    - OpenAI text-embedding-3-*
    - HuggingFace Transformers/SentenceTransformers
    - Cohere embed-v3
    - Anthropic (via Voyage)
    - LangChain wrapper

All providers implement EmbedderProtocol for drop-in replacement.
"""

from vector_search.embeddings.base import (
    BaseEmbedder,
    EmbedderProtocol,
    MockEmbedder,
)

__all__ = [
    "BaseEmbedder",
    "EmbedderProtocol",
    "MockEmbedder",
]

# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "OpenAIEmbedder":
        from vector_search.embeddings.openai import OpenAIEmbedder
        return OpenAIEmbedder
    if name == "HuggingFaceEmbedder":
        from vector_search.embeddings.huggingface import HuggingFaceEmbedder
        return HuggingFaceEmbedder
    if name == "CohereEmbedder":
        from vector_search.embeddings.cohere import CohereEmbedder
        return CohereEmbedder
    if name == "LangChainEmbedder":
        from vector_search.embeddings.langchain import LangChainEmbedder
        return LangChainEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
