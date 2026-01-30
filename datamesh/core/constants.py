"""
System-Wide Constants for Planetary-Scale Data Mesh

All magic numbers and configuration defaults centralized here.

Memory Alignment:
- Cache line: 64 bytes (Intel/AMD x86-64)
- SIMD width: 32 bytes (AVX2)
- Page size: 4096 bytes
"""

from typing import Final

# =============================================================================
# SIZE AND TIME UNITS
# =============================================================================
KB: Final[int] = 1024
MB: Final[int] = 1024 * KB
GB: Final[int] = 1024 * MB

MS: Final[int] = 1
SECOND_MS: Final[int] = 1000
MINUTE_MS: Final[int] = 60 * SECOND_MS
HOUR_MS: Final[int] = 60 * MINUTE_MS

NS_PER_MS: Final[int] = 1_000_000
NS_PER_S: Final[int] = 1_000_000_000

# =============================================================================
# MEMORY CONSTANTS
# =============================================================================
CACHE_LINE_BYTES: Final[int] = 64
PAGE_SIZE_BYTES: Final[int] = 4096

# =============================================================================
# CP TIER (PostgreSQL)
# =============================================================================
CP_POOL_MIN: Final[int] = 10
CP_POOL_MAX: Final[int] = 100
CP_CONN_TIMEOUT_MS: Final[int] = 5 * SECOND_MS
CP_QUERY_TIMEOUT_MS: Final[int] = 30 * SECOND_MS

# =============================================================================
# AP TIER (SQLite + LSM)
# =============================================================================
AP_CACHE_SIZE_KB: Final[int] = 64 * 1024
MEMTABLE_SIZE_BYTES: Final[int] = 64 * MB
WAL_BATCH_TIMEOUT_MS: Final[int] = 100
WAL_BATCH_SIZE_ROWS: Final[int] = 1000

# =============================================================================
# SHARDING
# =============================================================================
VIRTUAL_NODES: Final[int] = 4096
VIRTUAL_NODES_PER_PHYSICAL: Final[int] = 256  # Virtual nodes per physical node
HOT_KEY_SALT_BITS: Final[int] = 4

# =============================================================================
# RELIABILITY
# =============================================================================
CIRCUIT_BREAKER_ERROR_THRESHOLD: Final[float] = 0.5
CIRCUIT_BREAKER_RESET_MS: Final[int] = 30 * SECOND_MS
RETRY_BASE_MS: Final[int] = 100
RETRY_MAX_ATTEMPTS: Final[int] = 3

# =============================================================================
# BACKPRESSURE & RATE LIMITING
# =============================================================================
INGESTION_RATE_LIMIT_PER_TENANT: Final[float] = 10000.0  # requests/second
BACKPRESSURE_QUEUE_DEPTH_WARNING: Final[int] = 5000
BACKPRESSURE_QUEUE_DEPTH_CRITICAL: Final[int] = 10000

# =============================================================================
# VECTOR SEARCH
# =============================================================================
EMBEDDING_DIMS: Final[int] = 1536
HNSW_M: Final[int] = 16
HNSW_EF_SEARCH: Final[int] = 50

# =============================================================================
# PAGINATION
# =============================================================================
DEFAULT_PAGE_SIZE: Final[int] = 100
MAX_PAGE_SIZE: Final[int] = 1000

# =============================================================================
# SLO TARGETS
# =============================================================================
SLO_INGESTION_P99_MS: Final[int] = 200
SLO_QUERY_CACHED_P99_MS: Final[int] = 50
SLO_QUERY_UNCACHED_P99_MS: Final[int] = 500
