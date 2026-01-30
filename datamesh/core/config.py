"""
Configuration Management for Planetary-Scale Data Mesh

Provides validated configuration with sensible defaults.
Supports environment variable overrides and hot-reload.

Design:
- Immutable after validation
- Fail-fast on invalid configuration
- Type-safe with dataclasses
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datamesh.core.types import Result, Ok, Err, GeoRegion
from datamesh.core import constants as C


@dataclass(frozen=True)
class CPConfig:
    """CP Subsystem (PostgreSQL) configuration."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "datamesh_cp"
    user: str = "datamesh"
    password: str = ""
    pool_min: int = C.CP_POOL_MIN
    pool_max: int = C.CP_POOL_MAX
    conn_timeout_ms: int = C.CP_CONN_TIMEOUT_MS
    query_timeout_ms: int = C.CP_QUERY_TIMEOUT_MS
    ssl_mode: str = "prefer"
    
    @property
    def dsn(self) -> str:
        """PostgreSQL connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


@dataclass(frozen=True)
class APConfig:
    """AP Subsystem (SQLite + LSM) configuration."""
    
    data_dir: Path = field(default_factory=lambda: Path("./data/ap"))
    cache_size_kb: int = C.AP_CACHE_SIZE_KB
    memtable_size_bytes: int = C.MEMTABLE_SIZE_BYTES
    wal_batch_timeout_ms: int = C.WAL_BATCH_TIMEOUT_MS
    wal_batch_size_rows: int = C.WAL_BATCH_SIZE_ROWS
    mmap_enabled: bool = True
    compression: str = "lz4"
    
    @property
    def db_path(self) -> Path:
        """Primary database file path."""
        return self.data_dir / "content.db"
    
    @property
    def wal_path(self) -> Path:
        """Write-ahead log directory."""
        return self.data_dir / "wal"


@dataclass(frozen=True)
class ObjectStoreConfig:
    """Object Storage configuration."""
    
    backend: str = "filesystem"  # "filesystem" or "s3"
    data_dir: Path = field(default_factory=lambda: Path("./data/objects"))
    s3_endpoint: Optional[str] = None
    s3_bucket: str = "datamesh-objects"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region: str = "us-east-1"
    multipart_threshold_bytes: int = 8 * C.MB
    compression_enabled: bool = True


@dataclass(frozen=True)
class ShardingConfig:
    """Sharding and partitioning configuration."""
    
    virtual_nodes: int = C.VIRTUAL_NODES
    hot_key_salt_bits: int = C.HOT_KEY_SALT_BITS
    replication_factor: int = 3
    geo_region: GeoRegion = GeoRegion.US_EAST


@dataclass(frozen=True)
class ReliabilityConfig:
    """Reliability and fault tolerance configuration."""
    
    circuit_breaker_threshold: float = C.CIRCUIT_BREAKER_ERROR_THRESHOLD
    circuit_breaker_reset_ms: int = C.CIRCUIT_BREAKER_RESET_MS
    retry_base_ms: int = C.RETRY_BASE_MS
    retry_max_attempts: int = C.RETRY_MAX_ATTEMPTS
    lock_ttl_ms: int = 10000


@dataclass(frozen=True)
class ObservabilityConfig:
    """Observability and telemetry configuration."""
    
    metrics_enabled: bool = True
    metrics_port: int = 9090
    tracing_enabled: bool = True
    tracing_sample_rate: float = 0.01
    log_level: str = "INFO"
    log_json: bool = True


@dataclass(frozen=True)
class SecurityConfig:
    """Security configuration."""
    
    encryption_enabled: bool = True
    encryption_key_path: Optional[Path] = None
    tls_enabled: bool = True
    tls_cert_path: Optional[Path] = None
    tls_key_path: Optional[Path] = None
    jwt_issuer: str = "datamesh"
    jwt_audience: str = "datamesh-api"


@dataclass(frozen=True)
class DataMeshConfig:
    """Root configuration for the data mesh."""
    
    cp: CPConfig = field(default_factory=CPConfig)
    ap: APConfig = field(default_factory=APConfig)
    object_store: ObjectStoreConfig = field(default_factory=ObjectStoreConfig)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    @classmethod
    def from_env(cls) -> Result[DataMeshConfig, str]:
        """
        Load configuration from environment variables.
        
        Environment variables are prefixed with DATAMESH_.
        Example: DATAMESH_CP_HOST, DATAMESH_AP_DATA_DIR
        """
        try:
            cp = CPConfig(
                host=os.getenv("DATAMESH_CP_HOST", "localhost"),
                port=int(os.getenv("DATAMESH_CP_PORT", "5432")),
                database=os.getenv("DATAMESH_CP_DATABASE", "datamesh_cp"),
                user=os.getenv("DATAMESH_CP_USER", "datamesh"),
                password=os.getenv("DATAMESH_CP_PASSWORD", ""),
            )
            
            ap = APConfig(
                data_dir=Path(os.getenv("DATAMESH_AP_DATA_DIR", "./data/ap")),
            )
            
            obj = ObjectStoreConfig(
                backend=os.getenv("DATAMESH_OBJECT_BACKEND", "filesystem"),
                data_dir=Path(os.getenv("DATAMESH_OBJECT_DIR", "./data/objects")),
            )
            
            return Ok(cls(cp=cp, ap=ap, object_store=obj))
        except (ValueError, TypeError) as e:
            return Err(f"Configuration error: {e}")
    
    def validate(self) -> Result[None, str]:
        """Validate configuration invariants."""
        if self.cp.pool_min > self.cp.pool_max:
            return Err("CP pool_min cannot exceed pool_max")
        if self.sharding.replication_factor < 1:
            return Err("Replication factor must be >= 1")
        return Ok(None)
