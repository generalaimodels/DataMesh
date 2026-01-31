"""
SOTA Multi-Field Vector Search Integration with DataMesh

Features:
    1. Multi-field embedding (weighted concatenation)
    2. Hybrid search (vector + metadata filtering)
    3. Re-ranking with Cross-Encoder (SOTA)
    4. Streaming dataset processing with efficient batching
    5. REAL DataMesh API integration (Async AP Store)
    6. HIGH PERFORMANCE: Async Producer-Consumer Pipeline + Thread Offloading
    7. OPTIMIZATION: INT8 Dynamic Quantization + HNSW Tuning for CPU

Dataset: nvidia/Nemotron-Personas-Brazil
Model: intfloat/multilingual-e5-base (SOTA) + cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import asyncio
import time
import sys
import os
import hashlib
import warnings
from typing import Any, Iterator, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Robustly suppress PyTorch/HuggingFace warnings
warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao.quantization")
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure workspace root is in path for datamesh import
sys.path.append(os.getcwd())

import numpy as np

# VectorSearch
from vector_search.core.types import (
    VectorId,
    EmbeddingVector,
    SearchQuery,
    SearchResult,
    SearchResults,
    IndexConfig,
    MetricType,
)
from vector_search.core.config import HNSWConfig
from vector_search.core.errors import Result, Ok, Err
from vector_search.embeddings import MockEmbedder
from vector_search.index import HNSWIndex

# DataMesh API
try:
    from datamesh.storage import create_ap_store
    from datamesh.core.types import Result as DMResult
    print("✅ DataMesh API loaded successfully")
except ImportError:
    print("❌ DataMesh API not found. Please run from workspace root.")
    sys.exit(1)


# =============================================================================
# DATASET SCHEMA
# =============================================================================

EMBEDDING_COLUMNS = [
    "professional_persona",
    "sports_persona",
    "arts_persona", 
    "travel_persona",
    "culinary_persona",
    "personality",
    "career_goals_and_ambitions",
    "skills_and_expertise",
    "hobbies_and_interests",
    "cultural_background",
]

FILTER_COLUMNS = [
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    "municipality",
    "state",
    "country",
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SOTAConfig:
    """SOTA VectorSearch configuration."""
    
    # Dataset
    dataset_name: str = "nvidia/Nemotron-Personas-Brazil"
    dataset_split: str = "train"
    max_samples: int = 5000
    
    # Embedding strategy (SOTA Upgrade)
    embedding_columns: list[str] = field(default_factory=lambda: EMBEDDING_COLUMNS)
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dimension: int = 768
    batch_size: int = 64
    
    # HNSW (Tuned for Python CPU)
    # Reduced parameters to alleviate Python loop overhead
    hnsw_M: int = 16              
    hnsw_ef_construction: int = 100 
    hnsw_ef_search: int = 50      
    metric: MetricType = MetricType.COSINE
    
    # Search & Reranking
    enable_reranking: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 50
    final_k: int = 5
    
    # Performance
    num_workers: int = 2
    pipeline_queue_size: int = 10
    
    # Inference Optimization
    use_quantization: bool = True # Enable INT8 Dynamic Quantization
    
    # Field weights
    field_weights: dict[str, float] = field(default_factory=lambda: {
        "professional_persona": 2.0,
        "skills_and_expertise": 1.5,
        "career_goals_and_ambitions": 1.2,
        "personality": 1.0,
        "hobbies_and_interests": 0.8,
        "cultural_background": 0.5,
    })


# =============================================================================
# SOTA EMBEDDER (Bi-Encoder + Cross-Encoder)
# =============================================================================

class SOTAEmbedder:
    """Handles both Embedding (Bi-Encoder) and Reranking (Cross-Encoder)."""
    
    def __init__(self, config: SOTAConfig):
        self.config = config
        self._bi_encoder = None
        self._cross_encoder = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self._device = "cpu"
        
    def _ensure_initialized(self):
        if self._initialized: return
        
        try:
            import torch
            from sentence_transformers import SentenceTransformer, CrossEncoder
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"💻 Inference Device: {self._device.upper()}")
            
            # Load Bi-Encoder
            print(f"🔄 Loading Bi-Encoder: {self.config.embedding_model}")
            self._bi_encoder = SentenceTransformer(
                self.config.embedding_model, 
                device=self._device
            )
            
            # OPTIMIZATION: Dynamic Quantization for CPU
            if self._device == "cpu" and self.config.use_quantization:
                # Silently apply quantization, optimized to suppress all logs
                try:
                    self._bi_encoder = torch.quantization.quantize_dynamic(
                        self._bi_encoder, {torch.nn.Linear}, dtype=torch.qint8
                    )
                except Exception as e:
                    print(f"⚠️ Quantization failed: {e}")
            
            # Load Cross-Encoder
            if self.config.enable_reranking:
                print(f"🔄 Loading Cross-Encoder: {self.config.rerank_model}")
                self._cross_encoder = CrossEncoder(
                    self.config.rerank_model, 
                    device=self._device
                )
                
            print("✅ Models loaded successfully")
        except ImportError:
            print("⚠️ sentence-transformers not found. Using MockEmbedder.")
            pass
            
        self._initialized = True

    def _prepare_text(self, record: dict[str, Any]) -> str:
        """Construct weighted prompt for embedding."""
        parts = []
        for col in self.config.embedding_columns:
            text = record.get(col, "")
            if not text: continue
            
            weight = self.config.field_weights.get(col, 1.0)
            section = f"{col.replace('_', ' ').title()}: {text}"
            
            if weight >= 2.0:
                parts.insert(0, section)
                parts.append(section)
            elif weight >= 1.5:
                parts.append(section)
            else:
                parts.append(text[:200])
                
        return "passage: " + " ".join(parts)

    def _embed_sync(self, texts: list[str]) -> np.array:
        """Run embedding in thread (sync)."""
        if self._bi_encoder:
            return self._bi_encoder.encode(
                texts, 
                batch_size=self.config.batch_size, 
                normalize_embeddings=True, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
        return np.random.randn(len(texts), self.config.embedding_dimension).astype(np.float32)

    async def embed_batch_async(self, records: list[dict[str, Any]]) -> np.array:
        """Async wrapper for batch embedding (offloads to thread)."""
        self._ensure_initialized()
        texts = [self._prepare_text(r) for r in records]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._embed_sync, texts)

    def embed_query(self, query: str) -> np.array:
        """Embed query with correct prefix."""
        self._ensure_initialized()
        if self._bi_encoder:
            return self._bi_encoder.encode(
                f"query: {query}", 
                normalize_embeddings=True
            )
        return np.random.randn(self.config.embedding_dimension).astype(np.float32)

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank top-k results using Cross-Encoder."""
        if not self._cross_encoder or not results:
            return results
        
        # Cross-encoder inference is also CPU bound, careful. 
        # But applied only to 50 items.
        pairs = []
        for res in results:
            record = res['record']
            text = f"{record.get('professional_persona', '')} {record.get('skills_and_expertise', '')}"
            pairs.append([query, text])
            
        scores = self._cross_encoder.predict(pairs)
        
        for res, score in zip(results, scores):
            res['score'] = float(score)
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results


# =============================================================================
# INDEXING & SEARCH ENGINE
# =============================================================================

class SOTASearchEngine:
    def __init__(self, config: SOTAConfig):
        self.config = config
        self.embedder = SOTAEmbedder(config)
        self.ap_store = create_ap_store()
        
        hnsw_config = HNSWConfig(
            dimension=config.embedding_dimension,
            M=config.hnsw_M,
            ef_construction=config.hnsw_ef_construction,
            ef_search=config.hnsw_ef_search,
            metric=config.metric,
            max_elements=config.max_samples + 2000
        )
        self.index = HNSWIndex(config=hnsw_config)
        
        # Async Pipeline
        self.queue = asyncio.Queue(maxsize=config.pipeline_queue_size)
        self.workers = []
        self.stats = {"indexed": 0, "persistence_writes": 0}

    async def start(self):
        """Start async workers."""
        if not self.workers:
            for _ in range(self.config.num_workers):
                task = asyncio.create_task(self._worker_loop())
                self.workers.append(task)
            print(f"🚀 Started {len(self.workers)} ingestion workers")

    async def stop(self):
        """Stop workers and empty queue."""
        await self.queue.join()  # Wait for queue to empty
        for task in self.workers:
            task.cancel()
        self.workers = []

    async def enqueue_batch(self, batch: list[dict]):
        """Producer: push batch to queue."""
        await self.queue.put(batch)

    async def _worker_loop(self):
        """Consumer: Embed -> Index -> Store."""
        while True:
            try:
                batch = await self.queue.get()
                t_start = time.perf_counter()
                
                # 1. Embed (Async Offload)
                t_embed_0 = time.perf_counter()
                embeddings = await self.embedder.embed_batch_async(batch)
                t_embed = time.perf_counter() - t_embed_0
                
                items_to_store = {}
                
                # 2. Index & Prepare
                t_index_0 = time.perf_counter()
                for record, vec in zip(batch, embeddings):
                    rid = record.get("uuid") or hashlib.md5(str(record).encode()).hexdigest()[:16]
                    vid = VectorId(value=rid)
                    meta = {k: record.get(k) for k in FILTER_COLUMNS if record.get(k)}
                    
                    self.index.insert(vid, EmbeddingVector.from_numpy(vec), metadata=meta)
                    items_to_store[rid] = record
                t_index = time.perf_counter() - t_index_0
                    
                # 3. Persist (Async IO)
                t_store_0 = time.perf_counter()
                result = await self.ap_store.multi_put(items_to_store)
                t_store = time.perf_counter() - t_store_0
                
                if result.is_ok():
                    self.stats["persistence_writes"] += len(batch)
                
                self.stats["indexed"] += len(batch)
                t_total = time.perf_counter() - t_start
                self.queue.task_done()
                
                # Profile Log (First batch of each worker or every 10th)
                if self.stats["indexed"] % (self.config.batch_size * 5) == 0:
                    print(f"   ⏱️ [Worker] Batch ({len(batch)}): Embed={t_embed:.2f}s, Index={t_index:.2f}s, Store={t_store:.2f}s | Speed={len(batch)/t_total:.1f} r/s")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Worker Error: {e}")
                self.queue.task_done()

    async def search(self, query: str, filters: dict = None) -> list[dict]:
        k_fetch = self.config.rerank_top_k if self.config.enable_reranking else self.config.final_k
        query_vec = self.embedder.embed_query(query)
        
        sq = SearchQuery(
            vector=EmbeddingVector.from_numpy(query_vec),
            k=k_fetch,
            include_metadata=True
        )
        t0 = time.time()
        idx_result = self.index.search(sq)
        if idx_result.is_err(): return []
        t_search = time.time() - t0
        
        matches = idx_result.unwrap().matches
        
        candidates = []
        ids_to_fetch = []
        match_map = {}
        
        for match in matches:
            rid = match.id.value
            
            # Simple Filter Check
            if filters and match.metadata:
                skip = False
                for k, v in filters.items():
                    if match.metadata.get(k) != v:
                        skip = True; break
                if skip: continue
            
            ids_to_fetch.append(rid)
            match_map[rid] = match.score

        if ids_to_fetch:
            dm_result = await self.ap_store.multi_get(ids_to_fetch)
            if dm_result.is_ok():
                data_map = dm_result.unwrap()
                for rid, record in data_map.items():
                    candidates.append({
                        "id": rid,
                        "score": match_map[rid],
                        "record": record
                    })
        
        t1 = time.time()
        if self.config.enable_reranking and candidates:
            candidates = self.embedder.rerank(query, candidates)
        t_rerank = time.time() - t1
        
        print(f"   (Profile: Search={t_search*1000:.0f}ms, Rerank={t_rerank*1000:.0f}ms)")
        return candidates[:self.config.final_k]


# =============================================================================
# RUNNER
# =============================================================================

async def run_pipeline(config: SOTAConfig):
    print(f"🚀 Starting High-Performance SOTA Pipeline")
    print(f"Dataset: {config.dataset_name}")
    print(f"Config: Batch={config.batch_size}, Workers={config.num_workers}, Quantization={config.use_quantization}")
    print(f"HNSW: M={config.hnsw_M}, ef_c={config.hnsw_ef_construction}")
    
    engine = SOTASearchEngine(config)
    await engine.start()
    
    try:
        from datasets import load_dataset
        print("📂 Streaming dataset...")
        ds = load_dataset(config.dataset_name, split=config.dataset_split, streaming=True)
        
        batch = []
        t0 = time.time()
        
        for i, record in enumerate(ds):
            if i >= config.max_samples: break
            
            batch.append(dict(record))
            
            if len(batch) >= config.batch_size:
                await engine.enqueue_batch(batch)
                batch = []
                if i % 100 == 0:
                     print(f"   Indexed: {engine.stats['indexed']}...", end="\r")
                
        if batch: await engine.enqueue_batch(batch)
        
        await engine.stop()
        
        dt = time.time() - t0
        print(f"\n✅ Ingestion complete: {engine.stats['indexed']} records in {dt:.2f}s ({engine.stats['indexed']/dt:.1f} rec/s)")
        
    except ImportError:
        print("❌ datasets lib not installed.")
        return

    # --- Search Demonstrations ---
    queries = [
        ("Software engineer with Python skills", None),
        ("Healthcare professional in São Paulo", {"state": "São Paulo"}),
        ("Creative writer interested in sci-fi", None)
    ]
    
    print("\n🔍 Running SOTA Search Queries...")
    
    for q, f in queries:
        print(f"\n❓ Query: {q} (Filters: {f})")
        results = await engine.search(q, filters=f)
        
        for i, res in enumerate(results):
            rec = res['record']
            print(f"   {i+1}. [{res['score']:.4f}] {rec.get('occupation')} - {rec.get('professional_persona')[:80]}...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()
    
    cfg = SOTAConfig(max_samples=args.max_samples)
    asyncio.run(run_pipeline(cfg))
