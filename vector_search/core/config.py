"""
Configuration Classes: Type-Safe Index and Server Configuration

Provides structured configuration with validation for:
    - HNSW index parameters
    - Quantization settings  
    - Server deployment options
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from vector_search.core.types import MetricType, QuantizationType


# =============================================================================
# HNSW INDEX CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class HNSWConfig:
    """
    HNSW index configuration.
    
    Parameters:
        M: Bi-directional links per node (16=balanced, 32=high recall)
        ef_construction: Build beam width (100-400)
        ef_search: Search beam width (adjustable per-query)
    """
    M: int = 16
    M_max: int = 32
    ef_construction: int = 200
    ef_search: int = 100
    dimension: int = 768
    metric: MetricType = MetricType.COSINE
    max_elements: int = 1_000_000
    enable_simd: bool = True
    
    def validate(self) -> Optional[str]:
        if self.M < 2 or self.M > 100:
            return f"M must be in [2, 100], got {self.M}"
        if self.dimension < 1 or self.dimension > 65536:
            return f"dimension must be in [1, 65536], got {self.dimension}"
        return None
    
    @property
    def ml(self) -> float:
        import math
        return 1.0 / math.log(self.M)


# =============================================================================
# QUANTIZATION CONFIGURATION  
# =============================================================================
@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Vector quantization configuration for compression."""
    type: QuantizationType = QuantizationType.NONE
    sq_symmetric: bool = False
    pq_num_subvectors: int = 8
    pq_num_centroids: int = 256
    opq_rotation: bool = True
    
    def compression_ratio(self, dimension: int) -> float:
        if self.type == QuantizationType.NONE:
            return 1.0
        if self.type in (QuantizationType.FLOAT16, QuantizationType.BFLOAT16):
            return 2.0
        if self.type in (QuantizationType.INT8, QuantizationType.UINT8):
            return 4.0
        if self.type in (QuantizationType.PQ, QuantizationType.OPQ):
            return dimension * 4 / self.pq_num_subvectors
        return 1.0


# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
@dataclass(slots=True)
class ServerConfig:
    """Vector search server configuration."""
    grpc_port: int = 50051
    rest_port: int = 8080
    host: str = "0.0.0.0"
    mode: Literal["development", "production", "test"] = "development"
    data_dir: str = "./data"
    wal_enabled: bool = True
    max_concurrent_queries: int = 1000
    query_timeout_ms: float = 10000.0
    metrics_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            grpc_port=int(os.getenv("VECTORSEARCH_GRPC_PORT", "50051")),
            rest_port=int(os.getenv("VECTORSEARCH_REST_PORT", "8080")),
            mode=os.getenv("VECTORSEARCH_MODE", "development"),  # type: ignore
            data_dir=os.getenv("VECTORSEARCH_DATA_DIR", "./data"),
        )
