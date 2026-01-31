"""
Production Backend Configuration Module
========================================

Type-safe, immutable configuration dataclasses for production storage backends.
All configurations use frozen dataclasses for thread-safety and hash-ability.

Design Principles:
------------------
1. **Immutability**: All configs are frozen to prevent runtime mutation
2. **Validation**: Pre-conditions checked at construction time
3. **Defaults**: Sensible defaults for development; explicit for production
4. **Environment**: Supports loading from environment variables
5. **Documentation**: Every field documented with units and valid ranges

Complexity Analysis:
--------------------
- Construction: O(1) - field assignment only
- Validation: O(k) where k = number of fields (bounded, constant)
- Hash: O(k) - all fields are hashable primitives
- Memory: Minimal padding via size-ordered field layout

Author: Planetary AI Systems
License: MIT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum, auto


# =============================================================================
# ENUMERATIONS
# =============================================================================

class BackendType(Enum):
    """
    Storage backend type enumeration.
    
    Used for factory pattern dispatch and configuration validation.
    """
    IN_MEMORY = auto()  # Development/testing only
    REDIS = auto()      # AP tier - high availability
    VALKEY = auto()     # Redis-compatible OSS alternative
    SPANNER = auto()    # CP tier - strong consistency (GCP)
    COCKROACH = auto()  # CP tier - strong consistency (multi-cloud)
    S3 = auto()         # Object store - AWS
    GCS = auto()         # Object store - GCP
    MINIO = auto()      # Object store - self-hosted S3-compatible


class RedisMode(Enum):
    """
    Redis deployment topology.
    
    Determines connection pooling and failover strategy.
    """
    STANDALONE = auto()  # Single node - development
    SENTINEL = auto()    # HA via Redis Sentinel
    CLUSTER = auto()     # Sharded cluster mode


class ConsistencyPreference(Enum):
    """
    Read consistency preference for distributed stores.
    
    Trade-off between latency and consistency.
    """
    STRONG = auto()     # Linearizable reads - highest latency
    BOUNDED = auto()    # Bounded staleness - configurable lag
    EVENTUAL = auto()   # Best effort - lowest latency


# =============================================================================
# REDIS CONFIGURATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class RedisConfig:
    """
    Redis/Valkey connection configuration.
    
    Memory Layout (64-bit):
    -----------------------
    Ordered by descending size for minimal padding:
    - sentinel_hosts: 8 bytes (pointer to list)
    - password: 8 bytes (pointer to str or None)
    - host: 8 bytes (pointer to str)
    - socket_timeout_ms: 4 bytes (int32 sufficient)
    - connect_timeout_ms: 4 bytes
    - max_connections: 4 bytes
    - port: 4 bytes
    - db: 4 bytes
    - mode: 4 bytes (enum)
    - ssl: 1 byte (bool)
    - decode_responses: 1 byte
    Total: ~50 bytes + string allocations
    
    Thread Safety:
    -------------
    Frozen dataclass - immutable after construction.
    Safe for concurrent access without synchronization.
    
    Attributes:
        host: Redis server hostname or IP address.
        port: Redis server port (1-65535).
        password: Optional authentication password.
        db: Logical database index (0-15 for single node).
        mode: Deployment topology (standalone/sentinel/cluster).
        sentinel_hosts: List of (host, port) tuples for Sentinel mode.
        max_connections: Connection pool size. Must be > 0.
        connect_timeout_ms: TCP connection timeout in milliseconds.
        socket_timeout_ms: Socket read/write timeout in milliseconds.
        ssl: Enable TLS encryption for connections.
        decode_responses: Decode byte responses to UTF-8 strings.
    
    Example:
        >>> config = RedisConfig.from_env()
        >>> config = RedisConfig(host="redis.example.com", password="secret")
    """
    # 8-byte aligned fields first
    sentinel_hosts: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)
    password: Optional[str] = None
    host: str = "localhost"
    
    # 4-byte fields
    socket_timeout_ms: int = 5000
    connect_timeout_ms: int = 2000
    max_connections: int = 50
    port: int = 6379
    db: int = 0
    mode: RedisMode = RedisMode.STANDALONE
    
    # 1-byte fields
    ssl: bool = False
    decode_responses: bool = True
    
    def __post_init__(self) -> None:
        """
        Validate configuration invariants.
        
        Pre-conditions (enforced):
        - port ∈ [1, 65535]
        - db ∈ [0, 15] for standalone mode
        - max_connections > 0
        - connect_timeout_ms > 0
        - socket_timeout_ms > 0
        - sentinel_hosts non-empty if mode == SENTINEL
        
        Raises:
            ValueError: If any invariant is violated.
        
        Complexity: O(1) - bounded field count.
        """
        # Port validation: 16-bit unsigned range
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be in [1, 65535], got {self.port}")
        
        # Database index validation (Redis default max is 16)
        if self.mode == RedisMode.STANDALONE and not (0 <= self.db <= 15):
            raise ValueError(f"db must be in [0, 15] for standalone, got {self.db}")
        
        # Connection pool validation
        if self.max_connections <= 0:
            raise ValueError(f"max_connections must be > 0, got {self.max_connections}")
        
        # Timeout validation (must be positive)
        if self.connect_timeout_ms <= 0:
            raise ValueError(f"connect_timeout_ms must be > 0, got {self.connect_timeout_ms}")
        if self.socket_timeout_ms <= 0:
            raise ValueError(f"socket_timeout_ms must be > 0, got {self.socket_timeout_ms}")
        
        # Sentinel mode requires hosts
        if self.mode == RedisMode.SENTINEL and len(self.sentinel_hosts) == 0:
            raise ValueError("sentinel_hosts required when mode == SENTINEL")
    
    @classmethod
    def from_env(cls, prefix: str = "REDIS") -> "RedisConfig":
        """
        Construct configuration from environment variables.
        
        Environment Variables:
        - {prefix}_HOST: Server hostname (default: localhost)
        - {prefix}_PORT: Server port (default: 6379)
        - {prefix}_PASSWORD: Authentication password
        - {prefix}_DB: Database index (default: 0)
        - {prefix}_SSL: Enable TLS (default: false)
        - {prefix}_MAX_CONNECTIONS: Pool size (default: 50)
        - {prefix}_MODE: standalone|sentinel|cluster
        - {prefix}_SENTINEL_HOSTS: Comma-separated host:port pairs
        
        Args:
            prefix: Environment variable prefix (default: REDIS).
        
        Returns:
            RedisConfig: Configuration populated from environment.
        
        Complexity: O(k) where k = number of env vars checked.
        """
        def _get(key: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}_{key}", default)
        
        def _get_int(key: str, default: int) -> int:
            val = _get(key)
            return int(val) if val else default
        
        def _get_bool(key: str, default: bool) -> bool:
            val = _get(key).lower()
            if val in ("true", "1", "yes"):
                return True
            if val in ("false", "0", "no"):
                return False
            return default
        
        # Parse mode
        mode_str = _get("MODE", "standalone").lower()
        mode_map = {
            "standalone": RedisMode.STANDALONE,
            "sentinel": RedisMode.SENTINEL,
            "cluster": RedisMode.CLUSTER,
        }
        mode = mode_map.get(mode_str, RedisMode.STANDALONE)
        
        # Parse sentinel hosts
        sentinel_hosts: Tuple[Tuple[str, int], ...] = tuple()
        sentinel_str = _get("SENTINEL_HOSTS")
        if sentinel_str:
            parsed: List[Tuple[str, int]] = []
            for entry in sentinel_str.split(","):
                host_port = entry.strip().split(":")
                if len(host_port) == 2:
                    parsed.append((host_port[0], int(host_port[1])))
            sentinel_hosts = tuple(parsed)
        
        return cls(
            host=_get("HOST", "localhost"),
            port=_get_int("PORT", 6379),
            password=_get("PASSWORD") or None,
            db=_get_int("DB", 0),
            mode=mode,
            sentinel_hosts=sentinel_hosts,
            max_connections=_get_int("MAX_CONNECTIONS", 50),
            connect_timeout_ms=_get_int("CONNECT_TIMEOUT_MS", 2000),
            socket_timeout_ms=_get_int("SOCKET_TIMEOUT_MS", 5000),
            ssl=_get_bool("SSL", False),
            decode_responses=_get_bool("DECODE_RESPONSES", True),
        )
    
    def get_connection_kwargs(self) -> Dict[str, Any]:
        """
        Generate kwargs for redis-py connection.
        
        Returns:
            Dict suitable for redis.Redis() or redis.RedisCluster().
        
        Complexity: O(1) - fixed number of fields.
        """
        kwargs: Dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "max_connections": self.max_connections,
            "socket_connect_timeout": self.connect_timeout_ms / 1000.0,
            "socket_timeout": self.socket_timeout_ms / 1000.0,
            "decode_responses": self.decode_responses,
            "ssl": self.ssl,
        }
        if self.password:
            kwargs["password"] = self.password
        return kwargs


# =============================================================================
# S3 CONFIGURATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class S3Config:
    """
    S3-compatible object store configuration.
    
    Supports AWS S3, MinIO, Cloudflare R2, and other S3-compatible stores.
    
    Memory Layout (64-bit):
    -----------------------
    - bucket_name: 8 bytes (pointer)
    - region: 8 bytes (pointer)
    - endpoint_url: 8 bytes (pointer or None)
    - access_key_id: 8 bytes (pointer or None)
    - secret_access_key: 8 bytes (pointer or None)
    - session_token: 8 bytes (pointer or None)
    - multipart_threshold_bytes: 8 bytes (int64)
    - multipart_chunksize_bytes: 8 bytes (int64)
    - max_concurrency: 4 bytes
    - connect_timeout_seconds: 4 bytes
    - read_timeout_seconds: 4 bytes
    - max_retries: 4 bytes
    - use_ssl: 1 byte
    - verify_ssl: 1 byte
    Total: ~86 bytes + string allocations
    
    Thread Safety:
    -------------
    Frozen dataclass - immutable after construction.
    
    Attributes:
        bucket_name: S3 bucket name (required).
        region: AWS region or 'auto' for discovery.
        endpoint_url: Custom endpoint for MinIO/R2 (None for AWS).
        access_key_id: AWS access key (None for IAM role auth).
        secret_access_key: AWS secret key (None for IAM role auth).
        session_token: Temporary session token for STS.
        multipart_threshold_bytes: Size threshold for multipart upload.
        multipart_chunksize_bytes: Chunk size for multipart parts.
        max_concurrency: Max parallel upload/download threads.
        connect_timeout_seconds: TCP connect timeout.
        read_timeout_seconds: Read operation timeout.
        max_retries: Max retry attempts for transient failures.
        use_ssl: Use HTTPS for connections.
        verify_ssl: Verify SSL certificates (disable for self-signed).
    """
    # String fields (8-byte pointers)
    bucket_name: str
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    
    # 8-byte integer fields
    multipart_threshold_bytes: int = 8 * 1024 * 1024  # 8 MiB
    multipart_chunksize_bytes: int = 8 * 1024 * 1024  # 8 MiB
    
    # 4-byte integer fields
    max_concurrency: int = 10
    connect_timeout_seconds: int = 5
    read_timeout_seconds: int = 60
    max_retries: int = 3
    
    # 1-byte boolean fields
    use_ssl: bool = True
    verify_ssl: bool = True
    
    def __post_init__(self) -> None:
        """
        Validate configuration invariants.
        
        Pre-conditions (enforced):
        - bucket_name is non-empty
        - multipart_threshold >= 5MiB (S3 minimum)
        - multipart_chunksize >= 5MiB (S3 minimum)
        - max_concurrency > 0
        - timeouts > 0
        
        Raises:
            ValueError: If any invariant is violated.
        """
        # Bucket name validation
        if not self.bucket_name or len(self.bucket_name) < 3:
            raise ValueError("bucket_name must be at least 3 characters")
        
        # S3 requires minimum 5MiB for multipart
        min_multipart = 5 * 1024 * 1024
        if self.multipart_threshold_bytes < min_multipart:
            raise ValueError(
                f"multipart_threshold_bytes must be >= {min_multipart}, "
                f"got {self.multipart_threshold_bytes}"
            )
        if self.multipart_chunksize_bytes < min_multipart:
            raise ValueError(
                f"multipart_chunksize_bytes must be >= {min_multipart}, "
                f"got {self.multipart_chunksize_bytes}"
            )
        
        # Concurrency validation
        if self.max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be > 0, got {self.max_concurrency}")
        
        # Timeout validation
        if self.connect_timeout_seconds <= 0:
            raise ValueError(f"connect_timeout_seconds must be > 0")
        if self.read_timeout_seconds <= 0:
            raise ValueError(f"read_timeout_seconds must be > 0")
    
    @classmethod
    def from_env(cls, prefix: str = "S3") -> "S3Config":
        """
        Construct configuration from environment variables.
        
        Environment Variables:
        - {prefix}_BUCKET: Bucket name (required)
        - {prefix}_REGION: AWS region (default: us-east-1)
        - {prefix}_ENDPOINT_URL: Custom endpoint URL
        - {prefix}_ACCESS_KEY_ID: AWS access key ID
        - {prefix}_SECRET_ACCESS_KEY: AWS secret access key
        - AWS_SESSION_TOKEN: STS session token
        - {prefix}_MAX_CONCURRENCY: Parallel operations (default: 10)
        - {prefix}_USE_SSL: Use HTTPS (default: true)
        - {prefix}_VERIFY_SSL: Verify certs (default: true)
        
        Args:
            prefix: Environment variable prefix.
        
        Returns:
            S3Config populated from environment.
        
        Raises:
            ValueError: If required bucket_name is missing.
        """
        def _get(key: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}_{key}", default)
        
        def _get_int(key: str, default: int) -> int:
            val = _get(key)
            return int(val) if val else default
        
        def _get_bool(key: str, default: bool) -> bool:
            val = _get(key).lower()
            if val in ("true", "1", "yes"):
                return True
            if val in ("false", "0", "no"):
                return False
            return default
        
        bucket = _get("BUCKET")
        if not bucket:
            raise ValueError(f"Environment variable {prefix}_BUCKET is required")
        
        return cls(
            bucket_name=bucket,
            region=_get("REGION", "us-east-1"),
            endpoint_url=_get("ENDPOINT_URL") or None,
            access_key_id=_get("ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_access_key=_get("SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            session_token=os.environ.get("AWS_SESSION_TOKEN"),
            max_concurrency=_get_int("MAX_CONCURRENCY", 10),
            connect_timeout_seconds=_get_int("CONNECT_TIMEOUT", 5),
            read_timeout_seconds=_get_int("READ_TIMEOUT", 60),
            max_retries=_get_int("MAX_RETRIES", 3),
            use_ssl=_get_bool("USE_SSL", True),
            verify_ssl=_get_bool("VERIFY_SSL", True),
        )
    
    def get_boto_config(self) -> Dict[str, Any]:
        """
        Generate configuration dict for boto3/aioboto3.
        
        Returns:
            Dict suitable for boto3.client('s3', **config).
        """
        config: Dict[str, Any] = {
            "region_name": self.region,
            "use_ssl": self.use_ssl,
        }
        
        if self.endpoint_url:
            config["endpoint_url"] = self.endpoint_url
        
        if self.access_key_id and self.secret_access_key:
            config["aws_access_key_id"] = self.access_key_id
            config["aws_secret_access_key"] = self.secret_access_key
        
        if self.session_token:
            config["aws_session_token"] = self.session_token
        
        if not self.verify_ssl:
            config["verify"] = False
        
        return config


# =============================================================================
# SPANNER CONFIGURATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class SpannerConfig:
    """
    Google Cloud Spanner configuration.
    
    Spanner provides globally distributed, strongly consistent transactions.
    Used for the Control Plane tier where consistency is paramount.
    
    Memory Layout (64-bit):
    -----------------------
    - project_id: 8 bytes (pointer)
    - instance_id: 8 bytes (pointer)
    - database_id: 8 bytes (pointer)
    - credentials_path: 8 bytes (pointer or None)
    - pool_size: 4 bytes
    - timeout_seconds: 4 bytes
    - max_commit_delay_ms: 4 bytes
    - stale_read_seconds: 4 bytes
    - enable_leader_routing: 1 byte
    Total: ~57 bytes + string allocations
    
    Thread Safety:
    -------------
    Frozen dataclass - immutable after construction.
    
    Attributes:
        project_id: GCP project ID containing the Spanner instance.
        instance_id: Spanner instance ID.
        database_id: Database name within the instance.
        credentials_path: Path to service account JSON key file.
            If None, uses Application Default Credentials.
        pool_size: Session pool size for concurrent operations.
        timeout_seconds: Default operation timeout.
        max_commit_delay_ms: Max commit delay for batching.
        stale_read_seconds: Staleness bound for bounded consistency reads.
        enable_leader_routing: Route reads to leader for freshest data.
    """
    # Required string fields
    project_id: str
    instance_id: str
    database_id: str
    
    # Optional string fields
    credentials_path: Optional[str] = None
    
    # 4-byte integer fields
    pool_size: int = 100
    timeout_seconds: int = 30
    max_commit_delay_ms: int = 100
    stale_read_seconds: int = 15
    
    # 1-byte boolean
    enable_leader_routing: bool = True
    
    def __post_init__(self) -> None:
        """
        Validate configuration invariants.
        
        Pre-conditions:
        - project_id is non-empty
        - instance_id is non-empty
        - database_id is non-empty
        - pool_size > 0
        - timeout_seconds > 0
        """
        if not self.project_id:
            raise ValueError("project_id is required")
        if not self.instance_id:
            raise ValueError("instance_id is required")
        if not self.database_id:
            raise ValueError("database_id is required")
        if self.pool_size <= 0:
            raise ValueError(f"pool_size must be > 0, got {self.pool_size}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
    
    @classmethod
    def from_env(cls, prefix: str = "SPANNER") -> "SpannerConfig":
        """
        Construct configuration from environment variables.
        
        Environment Variables:
        - {prefix}_PROJECT_ID: GCP project ID (required)
        - {prefix}_INSTANCE_ID: Spanner instance ID (required)
        - {prefix}_DATABASE_ID: Database ID (required)
        - {prefix}_CREDENTIALS_PATH: Service account key path
        - {prefix}_POOL_SIZE: Session pool size (default: 100)
        - {prefix}_TIMEOUT_SECONDS: Operation timeout (default: 30)
        - GOOGLE_APPLICATION_CREDENTIALS: Default credential path
        
        Returns:
            SpannerConfig populated from environment.
        
        Raises:
            ValueError: If required fields are missing.
        """
        def _get(key: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}_{key}", default)
        
        def _get_int(key: str, default: int) -> int:
            val = _get(key)
            return int(val) if val else default
        
        project_id = _get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project_id:
            raise ValueError(f"{prefix}_PROJECT_ID or GOOGLE_CLOUD_PROJECT required")
        
        instance_id = _get("INSTANCE_ID")
        if not instance_id:
            raise ValueError(f"{prefix}_INSTANCE_ID is required")
        
        database_id = _get("DATABASE_ID")
        if not database_id:
            raise ValueError(f"{prefix}_DATABASE_ID is required")
        
        credentials_path = (
            _get("CREDENTIALS_PATH") or 
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        
        return cls(
            project_id=project_id,
            instance_id=instance_id,
            database_id=database_id,
            credentials_path=credentials_path or None,
            pool_size=_get_int("POOL_SIZE", 100),
            timeout_seconds=_get_int("TIMEOUT_SECONDS", 30),
            max_commit_delay_ms=_get_int("MAX_COMMIT_DELAY_MS", 100),
            stale_read_seconds=_get_int("STALE_READ_SECONDS", 15),
        )
    
    def get_database_path(self) -> str:
        """
        Get fully qualified Spanner database path.
        
        Returns:
            Path in format: projects/{project}/instances/{instance}/databases/{db}
        """
        return (
            f"projects/{self.project_id}/"
            f"instances/{self.instance_id}/"
            f"databases/{self.database_id}"
        )


# =============================================================================
# UNIFIED STORAGE CONFIGURATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class StorageConfig:
    """
    Unified configuration for all storage backends.
    
    Provides single point of configuration for the entire storage layer.
    Use `StorageConfig.from_env()` for production deployments.
    
    Attributes:
        cp_backend: Control Plane backend type.
        ap_backend: Availability Plane backend type.
        object_backend: Object store backend type.
        redis_config: Redis configuration (if ap_backend == REDIS/VALKEY).
        s3_config: S3 configuration (if object_backend == S3/MINIO).
        spanner_config: Spanner configuration (if cp_backend == SPANNER).
        enable_metrics: Enable latency/throughput metrics collection.
        enable_tracing: Enable distributed tracing spans.
    """
    # Backend type selection
    cp_backend: BackendType = BackendType.IN_MEMORY
    ap_backend: BackendType = BackendType.IN_MEMORY
    object_backend: BackendType = BackendType.IN_MEMORY
    
    # Backend-specific configs (optional based on backend type)
    redis_config: Optional[RedisConfig] = None
    s3_config: Optional[S3Config] = None
    spanner_config: Optional[SpannerConfig] = None
    
    # Observability
    enable_metrics: bool = True
    enable_tracing: bool = False
    
    def __post_init__(self) -> None:
        """
        Validate backend configuration consistency.
        
        Pre-conditions:
        - If ap_backend is REDIS/VALKEY, redis_config must be provided
        - If object_backend is S3/MINIO/GCS, s3_config must be provided
        - If cp_backend is SPANNER, spanner_config must be provided
        """
        if self.ap_backend in (BackendType.REDIS, BackendType.VALKEY):
            if self.redis_config is None:
                raise ValueError(
                    f"redis_config required when ap_backend={self.ap_backend.name}"
                )
        
        if self.object_backend in (BackendType.S3, BackendType.MINIO, BackendType.GCS):
            if self.s3_config is None:
                raise ValueError(
                    f"s3_config required when object_backend={self.object_backend.name}"
                )
        
        if self.cp_backend == BackendType.SPANNER:
            if self.spanner_config is None:
                raise ValueError(
                    "spanner_config required when cp_backend=SPANNER"
                )
    
    @classmethod
    def for_development(cls) -> "StorageConfig":
        """
        Get default configuration for local development.
        
        Uses all in-memory backends for zero external dependencies.
        """
        return cls(
            cp_backend=BackendType.IN_MEMORY,
            ap_backend=BackendType.IN_MEMORY,
            object_backend=BackendType.IN_MEMORY,
            enable_metrics=True,
            enable_tracing=False,
        )
    
    @classmethod
    def for_testing(cls, redis_config: Optional[RedisConfig] = None) -> "StorageConfig":
        """
        Get configuration for integration testing.
        
        Uses in-memory CP but allows real Redis for AP testing.
        """
        ap_backend = BackendType.REDIS if redis_config else BackendType.IN_MEMORY
        return cls(
            cp_backend=BackendType.IN_MEMORY,
            ap_backend=ap_backend,
            object_backend=BackendType.IN_MEMORY,
            redis_config=redis_config,
            enable_metrics=True,
            enable_tracing=False,
        )
    
    @classmethod
    def from_env(cls) -> "StorageConfig":
        """
        Construct full configuration from environment.
        
        Environment Variables:
        - STORAGE_CP_BACKEND: in_memory|spanner|cockroach
        - STORAGE_AP_BACKEND: in_memory|redis|valkey
        - STORAGE_OBJECT_BACKEND: in_memory|s3|gcs|minio
        - STORAGE_ENABLE_METRICS: true|false
        - STORAGE_ENABLE_TRACING: true|false
        
        Plus backend-specific variables (REDIS_*, S3_*, SPANNER_*).
        """
        def _get(key: str, default: str = "") -> str:
            return os.environ.get(f"STORAGE_{key}", default).lower()
        
        def _get_bool(key: str, default: bool) -> bool:
            val = _get(key)
            return val in ("true", "1", "yes") if val else default
        
        # Parse backend types
        backend_map = {
            "in_memory": BackendType.IN_MEMORY,
            "redis": BackendType.REDIS,
            "valkey": BackendType.VALKEY,
            "spanner": BackendType.SPANNER,
            "cockroach": BackendType.COCKROACH,
            "s3": BackendType.S3,
            "gcs": BackendType.GCS,
            "minio": BackendType.MINIO,
        }
        
        cp_str = _get("CP_BACKEND", "in_memory")
        ap_str = _get("AP_BACKEND", "in_memory")
        obj_str = _get("OBJECT_BACKEND", "in_memory")
        
        cp_backend = backend_map.get(cp_str, BackendType.IN_MEMORY)
        ap_backend = backend_map.get(ap_str, BackendType.IN_MEMORY)
        object_backend = backend_map.get(obj_str, BackendType.IN_MEMORY)
        
        # Load backend configs as needed
        redis_config = None
        if ap_backend in (BackendType.REDIS, BackendType.VALKEY):
            redis_config = RedisConfig.from_env()
        
        s3_config = None
        if object_backend in (BackendType.S3, BackendType.MINIO, BackendType.GCS):
            s3_config = S3Config.from_env()
        
        spanner_config = None
        if cp_backend == BackendType.SPANNER:
            spanner_config = SpannerConfig.from_env()
        
        return cls(
            cp_backend=cp_backend,
            ap_backend=ap_backend,
            object_backend=object_backend,
            redis_config=redis_config,
            s3_config=s3_config,
            spanner_config=spanner_config,
            enable_metrics=_get_bool("ENABLE_METRICS", True),
            enable_tracing=_get_bool("ENABLE_TRACING", False),
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BackendType",
    "RedisMode",
    "ConsistencyPreference",
    # Configs
    "RedisConfig",
    "S3Config",
    "SpannerConfig",
    "StorageConfig",
]
