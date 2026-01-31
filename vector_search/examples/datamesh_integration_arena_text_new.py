"""
Arena/YinYang-Text DataMesh Advanced Integration (v2)
=====================================================

Demonstrates the "Better API" using the High-Velocity Ingestion Pipeline.
Implements the full DataMesh architecture:
-   **Layer 7 API**: IngestionPipeline with Backpressure & Rate Limiting.
-   **Saga Pattern**: Distributed transactions across CP/AP/Object tiers.
-   **Advanced Types**: EntityId, GeoRegion, ComplianceTier.
-   **CDC**: Automatic Vector Indexing via Pipeline.

Usage:
    python datamesh_integration_arena_text_new.py --max-samples 100
"""

from __future__ import annotations

import asyncio
import time
import sys
import os
import uuid
import warnings
from typing import Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Suppress quantization warnings
warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.getcwd())

import numpy as np

# --- DataMesh Advanced Imports ---
try:
    from datamesh.core.config import DataMeshConfig, CPConfig, APConfig
    from datamesh.core.types import (
        Result, Ok, Err,
        EntityId, GeoRegion, ComplianceTier,
        EmbeddingVector, Timestamp
    )
    from datamesh.pipeline.ingestion import IngestionPipeline, IngestionRequest
    
    print("✅ Advanced DataMesh API loaded successfully")
except ImportError as e:
    print(f"❌ Library Import Failed: {e}")
    sys.exit(1)


# =============================================================================
# EXTENDED CONFIGURATION
# =============================================================================

@dataclass
class AdvancedArenaConfig:
    dataset_name: str = "Arena/YinYang-Text"
    split: str = "train"
    max_samples: int = 100
    
    # Model
    model_name: str = "intfloat/multilingual-e5-base"
    batch_size: int = 32
    num_workers: int = 2


# =============================================================================
# REUSED EMBEDDER (Optimized)
# =============================================================================

class SOTAEmbedder:
    """Optimized Embedder from v1 Integration."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    def _ensure_model(self):
        if self._model: return
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   🔄 Loading Encoder ({device}): {self.model_name}")
        self._model = SentenceTransformer(self.model_name, device=device)
        
        if device == "cpu":
            try:
                self._model = torch.quantization.quantize_dynamic(
                    self._model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except: pass

    def prepare_text(self, record: dict) -> str:
        prompt = record.get("prompt", "")[:500]
        chosen = record.get("chosen", "")[:500]
        return f"passage: Prompt: {prompt} \nAnswer: {chosen}"

    def _encode_sync(self, texts: list[str]) -> np.array:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    async def encode(self, record: dict) -> np.ndarray:
        self._ensure_model()
        text = self.prepare_text(record)
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(self._executor, self._encode_sync, [text])
        return vec[0]

    async def encode_batch(self, records: list[dict]) -> np.ndarray:
        self._ensure_model()
        texts = [self.prepare_text(r) for r in records]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._encode_sync, texts)


# =============================================================================
# MONKEYPATCH & SCHEMA UTILS (Run without Postgres)
# =============================================================================

def _monkeypatch_cp_engine():
    """Force CPEngine to mock mode if config provided."""
    from datamesh.storage.cp.engine import CPEngine
    
    original_init = CPEngine._initialize_pool
    
    async def mock_init_pool(self):
        # Always simulate restricted environment for demo
        # This prevents connection attempts to localhost:5432
        print("   ⚠️  [Demo Mode] Using Mock CP Engine (No Postgres Required)")
        return Ok(None)
        
    CPEngine._initialize_pool = mock_init_pool

async def _initialize_ap_schema(ap_engine):
    """Ensure SQLite tables exist for successful ingestion."""
    # 1. KV Store
    await ap_engine.execute_sql("""
        CREATE TABLE IF NOT EXISTS kv_store (
            key TEXT PRIMARY KEY,
            value BLOB
        )
    """)
    
    # 2. Response Dataframes
    # Clear for Demo Accuracy (Avoid duplicates from previous runs)
    await ap_engine.execute_sql("DELETE FROM response_dataframes") 
    
    await ap_engine.execute_sql("""
        CREATE TABLE IF NOT EXISTS response_dataframes (
            entity_id TEXT,
            conversation_id TEXT,
            sequence_id INTEGER,
            payload_ref TEXT,
            embedding BLOB,
            token_length INTEGER,
            quality_score REAL,
            metadata TEXT,
            PRIMARY KEY (entity_id, conversation_id, sequence_id)
        )
    """)
    
    # 3. Embeddings (if used by repo)
    await ap_engine.execute_sql("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY, 
            vector BLOB, 
            dimensions INTEGER, 
            source_type TEXT, 
            source_id TEXT
        )
    """)

# =============================================================================
# ADVANCED PIPELINE RUNNER
# =============================================================================

async def run_advanced_pipeline(config: AdvancedArenaConfig):
    print(f"\n🚀 Starting Advanced Ingestion Pipeline (IngestionPipeline) [BATCHED MODE]")
    
    # Apply Patch
    _monkeypatch_cp_engine()
    
    # 1. Initialize Configuration
    dm_config = DataMeshConfig.from_env().unwrap_or(DataMeshConfig())
    
    # 2. Create Pipeline
    print("   🔧 Initializing Architecture (CP/AP/Object/Index)...")
    pipeline_res = await IngestionPipeline.create(dm_config)
    
    if pipeline_res.is_err():
        print(f"   ❌ Pipeline Creation Failed: {pipeline_res.error}")
        return
        
    pipeline = pipeline_res.unwrap()
    await _initialize_ap_schema(pipeline._ap_engine)
    
    embedder = SOTAEmbedder(config.model_name)
    stats = {
        "ingested": 0, "failed": 0, "latency_sum": 0.0, 
        "latencies": [], "eval_data": []
    }
    
    try:
        from datasets import load_dataset
        print(f"   📂 Streaming Dataset (Batch Size: {config.batch_size})...")
        ds = load_dataset(config.dataset_name, split=config.split, streaming=True)
        
        start_time = time.time()
        batch_records = []
        sample_prompt = None
        
        async with pipeline:
            for i, item in enumerate(ds):
                if i >= config.max_samples: break
                batch_records.append(item)
                if sample_prompt is None:
                    sample_prompt = item.get("prompt", "")
                
                if len(batch_records) >= config.batch_size:
                    await _process_batch(pipeline, embedder, batch_records, stats)
                    print(f"   Processed {stats['ingested']}... (Throughput: {stats['ingested']/(time.time()-start_time):.1f} r/s)", end="\r")
                    batch_records = []
            
            if batch_records:
                await _process_batch(pipeline, embedder, batch_records, stats)
            
            # Verify Retrieval (while pipeline is open)
            if stats['ingested'] > 0 and len(stats.get('eval_data', [])) > 0:
                await evaluate_harness(pipeline, embedder, stats['eval_data'], stats['latencies'])

            # Interactive Mode
            if getattr(config, 'interactive', False):
                await interactive_cli(pipeline, embedder)
        
        duration = time.time() - start_time
        print(f"\n\n✅ Advanced Ingestion Complete")
        print(f"   Total: {stats['ingested']} | Failed: {stats['failed']}")
        
        # Final Latency Stats are printed in evaluate_harness

    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        import traceback
        traceback.print_exc()

async def _process_batch(pipeline, embedder, records, stats):
    # 1. Embed Batch
    vectors = await embedder.encode_batch(records)
    
    # 2. Prepare Requests
    requests = []
    batch_eval_data = [] # (prompt, entity_id)
    
    for rec, vec in zip(records, vectors):
        emb_obj = EmbeddingVector.from_floats(vec.tolist(), dimensions=768)
        eid = EntityId.generate()
        
        # Capture for Evaluation
        batch_eval_data.append((rec.get("prompt",""), str(eid)))
        
        req = IngestionRequest(
            entity_id=eid,
            prompt=rec.get("prompt", "").encode("utf-8")[:1000],
            response=rec.get("chosen", "").encode("utf-8")[:1000],
            geo_region=GeoRegion.US_EAST,
            compliance_tier=ComplianceTier.STANDARD,
            embedding=emb_obj,
            metadata={"source": "hf_arena", "rejected": rec.get("rejected", "")[:100]}
        )
        requests.append(req)
        
    # 3. Ingest Batch (Parallel Sagas)
    results = await pipeline.ingest_batch(requests)
    
    success_count = 0
    for res in results:
        if res.is_ok():
            success_count += 1
            lat = res.unwrap().latency_ms
            stats["ingested"] += 1
            stats["latency_sum"] += lat
            stats["latencies"].append(lat)
        else:
            stats["failed"] += 1
            
    # Add successful items to eval set
    # (Assuming sequential alignment of results to requests. ingest_batch preserves order)
    for i, res in enumerate(results):
        if res.is_ok():
            stats["eval_data"].append(batch_eval_data[i])

async def evaluate_harness(pipeline, embedder, eval_data, latencies):
    """
    Offline Evaluation Harness
    Calculates: P95/P99 Latency, NDCG@k, Recall@k
    """
    print(f"\n📊 Offline Evaluation Harness")
    print(f"===========================")
    
    # 1. Latency Metrics
    if latencies:
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        print(f"⏱️  Latency (End-to-End Pipeline):")
        print(f"    P50: {p50:.2f} ms")
        print(f"    P95: {p95:.2f} ms")
        print(f"    P99: {p99:.2f} ms")
    
    # 2. Retrieval Metrics (NDCG / Recall)
    # Sample 20 queries from eval_data for speed
    import random
    query_sample = random.sample(eval_data, min(20, len(eval_data)))
    
    print(f"\n🔍 Retrieval Quality (sampled {len(query_sample)} queries):")
    k = 5
    hits_at_k = 0
    ndcg_sum = 0.0
    
    for prompt, true_eid in query_sample:
        q_vec = await embedder.encode({"prompt": prompt, "chosen": ""})
        
        # Scan (In production this would be proper Index Search)
        # For Demo, we scan AP Store.
        rows = (await pipeline._ap_engine.execute_sql("SELECT entity_id, embedding FROM response_dataframes")).unwrap()
        
        # Calculate Scores
        candidates = []
        for r in rows:
            if not r["embedding"]: continue
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
            score = np.dot(q_vec, vec)
            candidates.append((r["entity_id"], score))
            
        # Sort Top K
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = candidates[:k]
        
        # Check Hit
        rank = -1
        for i, (cid, score) in enumerate(top_k):
            if cid == true_eid:
                rank = i
                break
        
        if rank != -1:
            hits_at_k += 1
            ndcg_sum += 1.0 / np.log2(rank + 2) # rank 0 -> log2(2)=1 -> 1.0
            
    recall = hits_at_k / len(query_sample)
    ndcg = ndcg_sum / len(query_sample)
    
    print(f"    Recall@{k}: {recall:.2f}")
    print(f"    NDCG@{k}:   {ndcg:.2f}")
    
    if recall > 0.8:
        print("✅ Quality Gate Passed")
    else:
        print("⚠️ Quality Gate Warning")

async def interactive_cli(pipeline, embedder):
    """Real-time Interactive Testing Mode."""
    print(f"\n🎮 Interactive CLI Mode")
    print("=======================")
    loop = asyncio.get_running_loop()
    
    while True:
        print("\nOptions:")
        print("  [1] 🔎 Real-time Search")
        print("  [2] 📊 Pipeline Stats")
        print("  [3] 🚪 Exit")
        
        try:
            choice = await loop.run_in_executor(None, input, "Select option: ")
            choice = choice.strip()
            
            if choice == "1":
                query = await loop.run_in_executor(None, input, "Enter query: ")
                if not query: continue
                
                print(f"   Running semantic search for: '{query}'...")
                q_vec = await embedder.encode({"prompt": query, "chosen": ""})
                
                rows = (await pipeline._ap_engine.execute_sql("SELECT entity_id, embedding, metadata FROM response_dataframes")).unwrap()
                
                candidates = []
                for r in rows:
                    if not r["embedding"]: continue
                    vec = np.frombuffer(r["embedding"], dtype=np.float32)
                    score = np.dot(q_vec, vec)
                    candidates.append((score, r["metadata"]))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                print(f"   🏆 Top 3 Results:")
                for i, (score, meta) in enumerate(candidates[:3]):
                    print(f"   {i+1}. Score: {score:.4f} | {meta[:100]}...")
                    
            elif choice == "2":
                print(f"   📈 Stats: {pipeline.stats}")
                
            elif choice == "3":
                print("   👋 Exiting.")
                break
                
            else:
                print("   ❌ Invalid option.")
                
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode after ingestion")
    args = parser.parse_args()
    
    cfg = AdvancedArenaConfig(max_samples=args.max_samples, batch_size=args.batch_size)
    
    # Store interactive flag in config shim or pass it
    # Easier to pass implicit config or modify AdvancedArenaConfig
    cfg.interactive = args.interactive
    
    asyncio.run(run_advanced_pipeline(cfg))
