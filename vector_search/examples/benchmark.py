"""
Real-Time Benchmark: VectorSearch Performance Testing

Tests:
    1. Insert throughput (vectors/sec)
    2. Search latency (P50, P95, P99)
    3. Recall@k accuracy
    4. Memory usage per vector

Usage:
    python -m vector_search.examples.benchmark --vectors 100000 --dimension 768
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Try to import C++ core
try:
    from vector_search import _vectorsearch_core as core
    HAS_CPP_CORE = True
    print("✅ C++ SIMD core loaded")
    print(f"   AVX2: {core.simd_support['avx2']}")
    print(f"   AVX-512: {core.simd_support['avx512']}")
    print(f"   FMA: {core.simd_support['fma']}")
except ImportError:
    HAS_CPP_CORE = False
    print("⚠️ C++ core not available, using Python fallback")

# Python fallback
from vector_search.core.types import MetricType
from vector_search.core.config import HNSWConfig
from vector_search.index import HNSWIndex


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    n_vectors: int = 100_000
    dimension: int = 768
    n_queries: int = 1_000
    k: int = 10
    ef_construction: int = 200
    ef_search: int = 50
    M: int = 16
    metric: str = "cosine"
    seed: int = 42
    use_cpp: bool = True
    ground_truth: bool = True  # Compute ground truth for recall


@dataclass
class BenchmarkResults:
    """Benchmark results."""
    # Insert metrics
    insert_time_sec: float = 0.0
    insert_throughput: float = 0.0
    
    # Search metrics
    search_latencies_ms: list[float] = None
    search_p50_ms: float = 0.0
    search_p95_ms: float = 0.0
    search_p99_ms: float = 0.0
    search_mean_ms: float = 0.0
    search_qps: float = 0.0
    
    # Accuracy
    recall_at_k: float = 0.0
    
    # Memory
    memory_mb: float = 0.0
    bytes_per_vector: float = 0.0
    
    def __post_init__(self):
        if self.search_latencies_ms is None:
            self.search_latencies_ms = []


def generate_random_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random normalized vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    return vectors


def compute_ground_truth(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str = "cosine"
) -> np.ndarray:
    """Compute exact k-NN for recall measurement."""
    n_queries = queries.shape[0]
    gt = np.zeros((n_queries, k), dtype=np.int64)
    
    for i, query in enumerate(queries):
        if metric == "cosine":
            # For normalized vectors, cosine = dot product
            scores = data @ query
        elif metric == "l2":
            scores = -np.linalg.norm(data - query, axis=1)
        else:  # inner_product
            scores = data @ query
        
        gt[i] = np.argsort(scores)[::-1][:k]
    
    return gt


def compute_recall(results: list[list], ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    total_recall = 0.0
    
    for i, (query_results, gt) in enumerate(zip(results, ground_truth)):
        retrieved = set(r[0] for r in query_results[:k])
        relevant = set(gt[:k])
        total_recall += len(retrieved & relevant) / k
    
    return total_recall / len(results)


def benchmark_cpp_core(config: BenchmarkConfig) -> Optional[BenchmarkResults]:
    """Benchmark C++ core."""
    if not HAS_CPP_CORE:
        return None
    
    results = BenchmarkResults()
    
    print(f"\n{'='*60}")
    print("C++ SIMD Core Benchmark")
    print(f"{'='*60}")
    
    # Generate data
    print(f"\n📊 Generating {config.n_vectors:,} vectors ({config.dimension}D)...")
    data = generate_random_vectors(config.n_vectors, config.dimension, config.seed)
    queries = generate_random_vectors(config.n_queries, config.dimension, config.seed + 1)
    ids = np.arange(config.n_vectors, dtype=np.uint64)
    
    # Ground truth
    gt = None
    if config.ground_truth:
        print("🔍 Computing ground truth...")
        gt = compute_ground_truth(data, queries, config.k, config.metric)
    
    # Create index
    print(f"\n📈 Creating HNSW index (M={config.M}, ef={config.ef_construction})...")
    index = core.HNSWIndex(
        dimension=config.dimension,
        max_elements=config.n_vectors + 1000,
        M=config.M,
        ef_construction=config.ef_construction,
        metric=config.metric,
    )
    
    # Insert benchmark
    print(f"\n⬆️ Inserting {config.n_vectors:,} vectors...")
    gc.collect()
    start = time.perf_counter()
    
    # Batch insert
    index.insert_batch(ids, data)
    
    results.insert_time_sec = time.perf_counter() - start
    results.insert_throughput = config.n_vectors / results.insert_time_sec
    
    print(f"   Time: {results.insert_time_sec:.2f}s")
    print(f"   Throughput: {results.insert_throughput:,.0f} vectors/sec")
    
    # Search benchmark
    print(f"\n🔍 Running {config.n_queries:,} searches (k={config.k}, ef={config.ef_search})...")
    latencies = []
    all_results = []
    
    gc.collect()
    
    for query in queries:
        start = time.perf_counter()
        result = index.search(query, k=config.k, ef=config.ef_search)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
        all_results.append(result)
    
    results.search_latencies_ms = latencies
    results.search_p50_ms = np.percentile(latencies, 50)
    results.search_p95_ms = np.percentile(latencies, 95)
    results.search_p99_ms = np.percentile(latencies, 99)
    results.search_mean_ms = np.mean(latencies)
    results.search_qps = 1000 / results.search_mean_ms
    
    print(f"   P50: {results.search_p50_ms:.3f}ms")
    print(f"   P95: {results.search_p95_ms:.3f}ms")
    print(f"   P99: {results.search_p99_ms:.3f}ms")
    print(f"   QPS: {results.search_qps:,.0f}")
    
    # Recall
    if gt is not None:
        results.recall_at_k = compute_recall(all_results, gt, config.k)
        print(f"\n📊 Recall@{config.k}: {results.recall_at_k:.4f}")
    
    # Memory
    stats = index.get_stats()
    results.memory_mb = stats["memory_bytes"] / 1024 / 1024
    results.bytes_per_vector = stats["memory_bytes"] / config.n_vectors
    
    print(f"\n💾 Memory: {results.memory_mb:.2f}MB ({results.bytes_per_vector:.1f} bytes/vector)")
    
    return results


def benchmark_python_fallback(config: BenchmarkConfig) -> BenchmarkResults:
    """Benchmark Python fallback."""
    results = BenchmarkResults()
    
    print(f"\n{'='*60}")
    print("Python HNSW Benchmark (Fallback)")
    print(f"{'='*60}")
    
    # Generate data
    print(f"\n📊 Generating {config.n_vectors:,} vectors ({config.dimension}D)...")
    data = generate_random_vectors(config.n_vectors, config.dimension, config.seed)
    queries = generate_random_vectors(config.n_queries, config.dimension, config.seed + 1)
    
    # Ground truth
    gt = None
    if config.ground_truth:
        print("🔍 Computing ground truth...")
        gt = compute_ground_truth(data, queries, config.k, config.metric)
    
    # Create index
    print(f"\n📈 Creating HNSW index...")
    
    metric_map = {
        "cosine": MetricType.COSINE,
        "l2": MetricType.L2,
        "ip": MetricType.INNER_PRODUCT,
    }
    
    hnsw_config = HNSWConfig(
        dimension=config.dimension,
        M=config.M,
        ef_construction=config.ef_construction,
        ef_search=config.ef_search,
        metric=metric_map.get(config.metric, MetricType.COSINE),
        max_elements=config.n_vectors + 1000,
    )
    
    index = HNSWIndex(config=hnsw_config)
    
    # Insert benchmark
    print(f"\n⬆️ Inserting {config.n_vectors:,} vectors...")
    gc.collect()
    start = time.perf_counter()
    
    from vector_search.core.types import VectorId, EmbeddingVector
    
    vectors = [
        (VectorId(str(i)), EmbeddingVector.from_numpy(data[i]))
        for i in range(config.n_vectors)
    ]
    
    for vid, vec in vectors:
        index.insert(vid, vec)
    
    results.insert_time_sec = time.perf_counter() - start
    results.insert_throughput = config.n_vectors / results.insert_time_sec
    
    print(f"   Time: {results.insert_time_sec:.2f}s")
    print(f"   Throughput: {results.insert_throughput:,.0f} vectors/sec")
    
    # Search benchmark
    print(f"\n🔍 Running {config.n_queries:,} searches...")
    latencies = []
    all_results = []
    
    gc.collect()
    
    from vector_search.core.types import SearchQuery
    
    for query in queries:
        q = SearchQuery(
            vector=EmbeddingVector.from_numpy(query),
            k=config.k,
        )
        
        start = time.perf_counter()
        result = index.search(q)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        
        if result.is_ok():
            all_results.append([
                (int(sr.id.value), sr.score) 
                for sr in result.unwrap().matches
            ])
        else:
            all_results.append([])
    
    results.search_latencies_ms = latencies
    results.search_p50_ms = np.percentile(latencies, 50)
    results.search_p95_ms = np.percentile(latencies, 95)
    results.search_p99_ms = np.percentile(latencies, 99)
    results.search_mean_ms = np.mean(latencies)
    results.search_qps = 1000 / results.search_mean_ms
    
    print(f"   P50: {results.search_p50_ms:.3f}ms")
    print(f"   P95: {results.search_p95_ms:.3f}ms")
    print(f"   P99: {results.search_p99_ms:.3f}ms")
    print(f"   QPS: {results.search_qps:,.0f}")
    
    # Recall
    if gt is not None:
        results.recall_at_k = compute_recall(all_results, gt, config.k)
        print(f"\n📊 Recall@{config.k}: {results.recall_at_k:.4f}")
    
    # Memory estimate
    stats = index.stats()
    results.memory_mb = stats.index_size_bytes / 1024 / 1024
    results.bytes_per_vector = stats.bytes_per_vector
    
    print(f"\n💾 Memory: {results.memory_mb:.2f}MB ({results.bytes_per_vector:.1f} bytes/vector)")
    
    return results


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(description="VectorSearch Benchmark")
    parser.add_argument("--vectors", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--dimension", type=int, default=768, help="Vector dimension")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="k for k-NN")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="ef_construction")
    parser.add_argument("--ef-search", type=int, default=50, help="ef_search")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2", "ip"])
    parser.add_argument("--no-cpp", action="store_true", help="Skip C++ benchmark")
    parser.add_argument("--no-python", action="store_true", help="Skip Python benchmark")
    parser.add_argument("--no-gt", action="store_true", help="Skip ground truth computation")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        n_vectors=args.vectors,
        dimension=args.dimension,
        n_queries=args.queries,
        k=args.k,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        metric=args.metric,
        ground_truth=not args.no_gt,
    )
    
    print("=" * 60)
    print("VectorSearch Performance Benchmark")
    print("=" * 60)
    print(f"Vectors: {config.n_vectors:,}")
    print(f"Dimension: {config.dimension}")
    print(f"Queries: {config.n_queries}")
    print(f"k: {config.k}")
    print(f"Metric: {config.metric}")
    
    cpp_results = None
    py_results = None
    
    # C++ benchmark
    if not args.no_cpp and HAS_CPP_CORE:
        cpp_results = benchmark_cpp_core(config)
    
    # Python benchmark
    if not args.no_python:
        py_results = benchmark_python_fallback(config)
    
    # Comparison
    if cpp_results and py_results:
        print(f"\n{'='*60}")
        print("Performance Comparison: C++ vs Python")
        print(f"{'='*60}")
        
        insert_speedup = py_results.insert_throughput / cpp_results.insert_throughput if cpp_results.insert_throughput > 0 else 0
        search_speedup = py_results.search_mean_ms / cpp_results.search_mean_ms if cpp_results.search_mean_ms > 0 else 0
        
        print(f"\n📊 Insert Speedup: {1/insert_speedup:.1f}x (C++ faster)")
        print(f"🔍 Search Speedup: {search_speedup:.1f}x (C++ faster)")
        
        if cpp_results.recall_at_k > 0 and py_results.recall_at_k > 0:
            print(f"\n📈 Recall@{config.k}:")
            print(f"   C++: {cpp_results.recall_at_k:.4f}")
            print(f"   Python: {py_results.recall_at_k:.4f}")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
