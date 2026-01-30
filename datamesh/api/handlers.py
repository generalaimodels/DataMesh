"""
API Handlers: Request Processing Logic

Implements:
- IngestHandler: Multi-dataframe ingestion
- QueryHandler: Conversation retrieval
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from datamesh.api.router import Request, Response
from datamesh.core.types import EntityId, ConversationId, EmbeddingVector, GeoRegion, ComplianceTier
from datamesh.core.errors import IngestionError, StorageError
from datamesh.pipeline.ingestion import IngestionPipeline, IngestionRequest
from datamesh.storage.cp.repositories import ConversationRepository
from datamesh.storage.ap.repositories import ResponseRepository


class IngestHandler:
    """
    Handler for ingestion requests.
    
    Endpoints:
    - POST /api/v1/ingest: Ingest single conversation
    - POST /api/v1/ingest/batch: Batch ingestion
    """
    
    __slots__ = ("_pipeline",)
    
    def __init__(self, pipeline: IngestionPipeline) -> None:
        self._pipeline = pipeline
    
    async def ingest(self, request: Request) -> Response:
        """
        Single document ingestion.
        
        Request:
            {
                "entity_id": "uuid",
                "prompt": "base64 or string",
                "response": "base64 or string",
                "instruction_version": 1,
                "geo_region": "US_EAST",
                "embedding": [0.1, 0.2, ...],
                "metadata": {}
            }
        """
        try:
            data = request.json()
            if data is None:
                return Response.error("Request body required")
            
            # Parse entity_id
            try:
                entity_id = EntityId(value=UUID(data["entity_id"]))
            except (KeyError, ValueError) as e:
                return Response.error(f"Invalid entity_id: {e}")
            
            # Parse content
            prompt = self._decode_content(data.get("prompt", ""))
            response = self._decode_content(data.get("response", ""))
            
            if not prompt:
                return Response.error("prompt is required")
            if not response:
                return Response.error("response is required")
            
            # Parse optional fields
            embedding = None
            if "embedding" in data:
                embedding = EmbeddingVector.from_list(data["embedding"])
            
            geo_region = GeoRegion.US_EAST
            if "geo_region" in data:
                try:
                    geo_region = GeoRegion[data["geo_region"].upper()]
                except KeyError:
                    pass
            
            compliance_tier = ComplianceTier.STANDARD
            if "compliance_tier" in data:
                try:
                    compliance_tier = ComplianceTier[data["compliance_tier"].upper()]
                except KeyError:
                    pass
            
            # Create ingestion request
            ingest_req = IngestionRequest(
                entity_id=entity_id,
                prompt=prompt,
                response=response,
                instruction_version=data.get("instruction_version", 1),
                geo_region=geo_region,
                compliance_tier=compliance_tier,
                embedding=embedding,
                metadata=data.get("metadata", {}),
                idempotency_key=request.header("idempotency-key"),
            )
            
            # Execute ingestion
            result = await self._pipeline.ingest(ingest_req)
            
            if result.is_err():
                error = result.error
                return Response.json(
                    {"error": str(error), "code": error.code.value},
                    status=429 if "backpressure" in str(error) else 400,
                )
            
            resp = result.unwrap()
            return Response.json({
                "conversation_id": str(resp.conversation_id),
                "prompt_hash": resp.prompt_hash.to_hex(),
                "response_hash": resp.response_hash.to_hex(),
                "sequence_id": resp.sequence_id,
                "latency_ms": resp.latency_ms,
                "deduplicated": resp.deduplicated,
            }, status=201)
            
        except json.JSONDecodeError as e:
            return Response.error(f"Invalid JSON: {e}")
        except Exception as e:
            return Response.error(f"Ingestion failed: {e}", status=500)
    
    async def ingest_batch(self, request: Request) -> Response:
        """
        Batch ingestion of multiple documents.
        
        Request:
            {
                "items": [
                    { ... ingestion request ... },
                    ...
                ]
            }
        """
        try:
            data = request.json()
            items = data.get("items", [])
            
            if not items:
                return Response.error("items array required")
            
            if len(items) > 1000:
                return Response.error("Maximum 1000 items per batch")
            
            results: list[dict[str, Any]] = []
            
            for i, item in enumerate(items):
                # Create sub-request
                sub_request = Request(
                    method="POST",
                    path="/api/v1/ingest",
                    headers=request.headers,
                    body=json.dumps(item).encode(),
                )
                
                response = await self.ingest(sub_request)
                result_data = json.loads(response.body)
                result_data["index"] = i
                result_data["success"] = response.status < 400
                results.append(result_data)
            
            success_count = sum(1 for r in results if r.get("success"))
            
            return Response.json({
                "total": len(items),
                "success": success_count,
                "failed": len(items) - success_count,
                "results": results,
            })
            
        except Exception as e:
            return Response.error(f"Batch ingestion failed: {e}", status=500)
    
    async def stats(self, request: Request) -> Response:
        """Get pipeline statistics."""
        stats = self._pipeline.stats
        bp = self._pipeline.backpressure_metrics
        
        return Response.json({
            "pipeline": {
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "deduplicated_requests": stats.deduplicated_requests,
                "total_bytes_ingested": stats.total_bytes_ingested,
                "avg_latency_ms": round(stats.avg_latency_ms, 2),
            },
            "backpressure": {
                "state": bp.state.name,
                "queue_depth": bp.queue_depth,
                "inflight_requests": bp.inflight_requests,
                "current_batch_size": bp.current_batch_size,
            },
        })
    
    @staticmethod
    def _decode_content(content: Any) -> bytes:
        """Decode content from string or base64."""
        if isinstance(content, bytes):
            return content
        if isinstance(content, str):
            # Try base64 first
            try:
                import base64
                return base64.b64decode(content)
            except Exception:
                return content.encode("utf-8")
        return b""


class QueryHandler:
    """
    Handler for query requests.
    
    Endpoints:
    - GET /api/v1/conversations/{id}: Get conversation
    - GET /api/v1/entities/{id}/conversations: List conversations
    """
    
    __slots__ = ("_conv_repo", "_resp_repo")
    
    def __init__(
        self,
        conv_repo: ConversationRepository,
        resp_repo: ResponseRepository,
    ) -> None:
        self._conv_repo = conv_repo
        self._resp_repo = resp_repo
    
    async def get_conversation(self, request: Request) -> Response:
        """Get conversation by ID."""
        try:
            conv_id_str = request.path_params.get("id", "")
            entity_id_str = request.query("entity_id")
            
            if not entity_id_str:
                return Response.error("entity_id query param required")
            
            try:
                entity_id = EntityId(value=UUID(entity_id_str))
                conv_id = ConversationId(value=UUID(conv_id_str))
            except ValueError as e:
                return Response.error(f"Invalid ID: {e}")
            
            result = await self._conv_repo.get_by_id(entity_id, conv_id)
            
            if result.is_err():
                return Response.error(str(result.error), status=500)
            
            conv = result.unwrap()
            if conv is None:
                return Response.not_found()
            
            return Response.json({
                "entity_id": str(conv.entity_id),
                "conversation_id": str(conv.conversation_id),
                "prompt_hash": conv.prompt_hash.to_hex(),
                "instruction_version": conv.instruction_version,
                "created_at": conv.created_at.isoformat(),
                "geo_region": conv.geo_region.name,
                "compliance_tier": conv.compliance_tier.name,
            })
            
        except Exception as e:
            return Response.error(str(e), status=500)
    
    async def list_conversations(self, request: Request) -> Response:
        """List conversations for entity."""
        try:
            entity_id_str = request.path_params.get("id", "")
            
            try:
                entity_id = EntityId(value=UUID(entity_id_str))
            except ValueError as e:
                return Response.error(f"Invalid entity_id: {e}")
            
            limit = int(request.query("limit", "100"))
            cursor = request.query("cursor")
            
            result = await self._conv_repo.list_by_entity(
                entity_id,
                limit=min(limit, 1000),
            )
            
            if result.is_err():
                return Response.error(str(result.error), status=500)
            
            convs, next_cursor = result.unwrap()
            
            return Response.json({
                "items": [
                    {
                        "conversation_id": str(c.conversation_id),
                        "prompt_hash": c.prompt_hash.to_hex(),
                        "instruction_version": c.instruction_version,
                        "created_at": c.created_at.isoformat(),
                    }
                    for c in convs
                ],
                "next_cursor": next_cursor.to_string() if next_cursor else None,
            })
            
        except Exception as e:
            return Response.error(str(e), status=500)
    
    async def get_responses(self, request: Request) -> Response:
        """Get response dataframes for conversation."""
        try:
            conv_id_str = request.path_params.get("id", "")
            entity_id_str = request.query("entity_id")
            
            if not entity_id_str:
                return Response.error("entity_id query param required")
            
            try:
                entity_id = EntityId(value=UUID(entity_id_str))
                conv_id = ConversationId(value=UUID(conv_id_str))
            except ValueError as e:
                return Response.error(f"Invalid ID: {e}")
            
            result = await self._resp_repo.list_by_conversation(entity_id, conv_id)
            
            if result.is_err():
                return Response.error(str(result.error), status=500)
            
            responses, next_cursor = result.unwrap()
            
            return Response.json({
                "items": [
                    {
                        "sequence_id": r.sequence_id,
                        "payload_ref": r.payload_ref,
                        "token_length": r.token_length,
                        "quality_score": r.quality_score,
                        "metadata": r.metadata,
                    }
                    for r in responses
                ],
                "next_cursor": next_cursor.to_string() if next_cursor else None,
            })
            
        except Exception as e:
            return Response.error(str(e), status=500)
