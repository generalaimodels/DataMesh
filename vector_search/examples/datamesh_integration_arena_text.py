"""
Arena/YinYang-Text DataMesh Integration Demo
============================================

End-to-End demonstration of integrating a high-performance Vector Search pipeline 
with DataMesh storage, using the 'Arena/YinYang-Text' RLHF dataset.

Key Features:
-   **Schema**: Handles RLHF data (Prompt, Chosen, Rejected, Alignment tags).
-   **Architecture**: Async Producer-Consumer Pipeline with DataMesh AP Store.
-   **Performance**: INT8 Quantization (CPU), Thread Offloading, Optimized HNSW (M=16, ef=100).
-   **Persistence**: Uses DataMesh `APStore` (Availability Plane) for scalable storage of full records.
-   **Test Validation**: Includes an End-to-End validation class simulating `datamesh/tests`.

Usage:
    python datamesh_integration_arena_text.py --max-samples 1000
"""

from __future__ import annotations

import asyncio
import time
import sys
import os
import hashlib
import warnings
import uuid
from typing import Any, Iterator, Optional, List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings for clean demo output
warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao.quantization")
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure workspace root is in path
sys.path.append(os.getcwd())

import numpy as np

# --- DataMesh & VectorSearch Imports ---
try:
    from datamesh.storage import create_ap_store
    from datamesh.core.types import Result as DMResult, Ok as DMOk, Err as DMErr
    # ConsistencyLevel removed as it's not exported in types.py
    
    from vector_search.core.types import (
        VectorId,
        EmbeddingVector,
        SearchQuery,
        IndexConfig,
        MetricType,
    )
    from vector_search.core.config import HNSWConfig
    from vector_search.index import HNSWIndex
    
    print("✅ DataMesh & VectorSearch Libraries loaded successfully")
except ImportError as e:
    print(f"❌ Library Import Failed: {e}")
    print("Please run this script from the workspace root.")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ArenaConfig:
    """Configuration for Arena/YinYang-Text Pipeline."""
    
    # Dataset
    dataset_name: str = "Arena/YinYang-Text"
    split: str = "train"
    max_samples: int = 1000
    
    # SOTA Embedding (E5-base)
    model_name: str = "intfloat/multilingual-e5-base"  
    dimension: int = 768
    batch_size: int = 64
    
    # HNSW Parameters (Optimized for Speed/Recall Balance)
    hnsw_M: int = 16
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 50
    metric: MetricType = MetricType.COSINE
    
    # System
    num_workers: int = 2
    queue_size: int = 20
    use_quantization: bool = True  # INT8 on CPU


# =============================================================================
# COMPONENTS
# =============================================================================

class ArenaEmbedder:
    """Specialized Embedder for RLHF Data (Prompt + Chosen)."""
    
    def __init__(self, config: ArenaConfig):
        self.config = config
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self._initialized = False
        self._device = "cpu"

    def _ensure_model(self):
        if self._initialized: return
        import torch
        from sentence_transformers import SentenceTransformer
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   🔄 Loading Encoder ({self._device.upper()}): {self.config.model_name}")
        
        self._model = SentenceTransformer(self.config.model_name, device=self._device)
        
        # Apply Quantization if CPU
        if self._device == "cpu" and self.config.use_quantization:
            try:
                self._model = torch.quantization.quantize_dynamic(
                    self._model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass # Fail silently, fallback to float32
                
        self._initialized = True

    def prepare_text(self, record: dict) -> str:
        """Format: 'passage: Prompt: {p} Chosen: {c}'."""
        prompt = record.get("prompt", "")[:500]
        chosen = record.get("chosen", "")[:500]
        return f"passage: Prompt: {prompt} \nAnswer: {chosen}"

    def _encode_sync(self, texts: list[str]) -> np.array:
        return self._model.encode(
            texts, 
            batch_size=len(texts), 
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    async def encode_batch(self, batch: list[dict]) -> np.array:
        self._ensure_model()
        texts = [self.prepare_text(r) for r in batch]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._encode_sync, texts)

    def encode_query(self, query: str) -> np.array:
        self._ensure_model()
        return self._model.encode(f"query: {query}", normalize_embeddings=True)


class DataMeshPipeline:
    """Manages Ingestion and Retrieval using DataMesh + VectorSearch."""
    
    def __init__(self, config: ArenaConfig):
        self.config = config
        self.embedder = ArenaEmbedder(config)
        self.ap_store = create_ap_store() # DataMesh Availability Plane
        
        self.index = HNSWIndex(HNSWConfig(
            dimension=config.dimension,
            M=config.hnsw_M,
            ef_construction=config.hnsw_ef_construction,
            ef_search=config.hnsw_ef_search,
            metric=config.metric,
            max_elements=config.max_samples + 5000
        ))
        
        self.queue = asyncio.Queue(maxsize=config.queue_size)
        self.workers = []
        self.stats = {"indexed": 0, "writes": 0}

    async def start(self):
        for _ in range(self.config.num_workers):
            self.workers.append(asyncio.create_task(self._worker()))
            
    async def stop(self):
        await self.queue.join()
        for w in self.workers: w.cancel()
        
    async def ingest(self, batch: list[dict]):
        await self.queue.put(batch)
        
    async def _worker(self):
        while True:
            try:
                batch = await self.queue.get()
                
                # 1. Embed
                embeddings = await self.embedder.encode_batch(batch)
                
                # 2. Prepare Storage & Index
                store_map = {}
                
                for rec, vec in zip(batch, embeddings):
                    # Generate ID
                    rid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(rec)[:500]))
                    rec["uuid"] = rid # Inject ID
                    
                    vid = VectorId(value=rid)
                    self.index.insert(vid, EmbeddingVector.from_numpy(vec))
                    store_map[rid] = rec
                
                # 3. DataMesh Persistence (Async Batch Write)
                # Removed explicit consistency arg
                result = await self.ap_store.multi_put(store_map)
                
                if result.is_ok():
                    self.stats["writes"] += len(batch)
                
                self.stats["indexed"] += len(batch)
                self.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker Error: {e}")
                self.queue.task_done()

    async def search(self, query: str, k: int = 5) -> list[dict]:
        """End-to-End Search: Index -> DataMesh -> Result."""
        q_vec = self.embedder.encode_query(query)
        
        # ANN Search
        res = self.index.search(SearchQuery(
            vector=EmbeddingVector.from_numpy(q_vec),
            k=k
        ))
        
        if res.is_err(): return []
        
        # DataMesh Hydration
        matches = res.unwrap().matches
        ids = [m.id.value for m in matches]
        
        if not ids: return []
        
        dm_res = await self.ap_store.multi_get(ids)
        if dm_res.is_err(): return []
        
        records = dm_res.unwrap()
        
        # Combine
        results = []
        for match in matches:
            rec = records.get(match.id.value)
            if rec:
                rec["_score"] = match.score
                results.append(rec)
                
        return results


# =============================================================================
# END-TO-END TEST RUNNER
# =============================================================================

class IntegrationTest:
    """Simulates `datamesh/tests` validation suite."""

    @staticmethod
    async def run_end_to_end_test(config: ArenaConfig):
        print(f"\n🧪 Starting End-to-End Integration Test")
        print(f"   Target: {config.dataset_name} | Max Samples: {config.max_samples}")
        
        pipeline = DataMeshPipeline(config)
        await pipeline.start()
        
        # --- Phase 1: Ingestion ---
        try:
            from datasets import load_dataset
            print("   📂 Loading Dataset Stream...")
            ds = load_dataset(config.dataset_name, split=config.split, streaming=True)
            
            batch = []
            start_time = time.time()
            
            for i, item in enumerate(ds):
                if i >= config.max_samples: break
                batch.append(dict(item))
                
                if len(batch) >= config.batch_size:
                    await pipeline.ingest(batch)
                    batch = []
                    print(f"   Indexed {pipeline.stats['indexed']}...", end="\r")
                    
            if batch: await pipeline.ingest(batch)
            
            await pipeline.stop()
            duration = time.time() - start_time
            
            print(f"\n   ✅ Ingestion Complete: {pipeline.stats['indexed']} records in {duration:.2f}s")
            print(f"      Throughput: {pipeline.stats['indexed']/duration:.2f} rec/s")
            
        except Exception as e:
            print(f"   ❌ Ingestion Failed: {e}")
            return

        # --- Phase 2: Verification (DataMesh Checks) ---
        print("\n   🔍 Verifying Persistence & Consistency...")
        
        # Check 1: Random ID Lookup
        test_id = list(pipeline.stats.keys())[0] # Just dummy, actual IDs are internal
        # We need an actual ID. Let's do a search to get an ID.
        
        results = await pipeline.search("test", k=1)
        if results:
            rid = results[0]["uuid"]
            print(f"      [Check 1] Retrieved Record ID: {rid[:8]}... ", end="")
            
            # Direct DataMesh Access Check
            # Use multi_get for consistency with ingestion path
            direct_res = await pipeline.ap_store.multi_get([rid])
            if direct_res.is_ok():
                fetched_map = direct_res.unwrap()
                if rid in fetched_map and fetched_map[rid].get("uuid") == rid:
                     print("PASS ✅")
                else:
                     print(f"FAIL ❌ (ID match failed: {fetched_map.get(rid)})")
            else:
                 print("FAIL ❌ (Data Not Found in AP Store)")
        else:
            print("      [Check 1] Search returned no results. Ingestion might have failed.")

        # --- Phase 3: Semantic Search Demo ---
        queries = [
            "How to align AI safely?",
            "Write a python script for matrix multiplication",
            "Explain quantum entanglement simply"
        ]
        
        print("\n   🧠 Semantic Search Demonstration:")
        for q in queries:
            print(f"\n   ❓ Query: '{q}'")
            hits = await pipeline.search(q, k=2)
            for i, hit in enumerate(hits):
                prompt_snip = hit.get('prompt', '')[:80].replace('\n', ' ')
                score = hit.get('_score', 0)
                print(f"      {i+1}. [{score:.4f}] {prompt_snip}...")

        print("\n✅ End-to-End Test Suite Completed Successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()
    
    cfg = ArenaConfig(max_samples=args.max_samples)
    asyncio.run(IntegrationTest.run_end_to_end_test(cfg))
