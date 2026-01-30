"""
Context Window: Token-Aware Context Management

Provides:
- Dynamic window sizing (4K-128K tokens)
- Priority-based retention (system > user > assistant)
- Truncation strategies (head, tail, middle-out)
- Token budget allocation

Design:
    Context window manages the subset of working set
    that fits within the model's context length.
    Implements intelligent truncation to maximize
    information density within token budgets.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Sequence

from datamesh.core.types import Result, Ok, Err, EntityId
from datamesh.memory.stm.working_set import WorkingSet, WorkingSetItem


# =============================================================================
# CONTEXT PRIORITY
# =============================================================================
class ContextPriority(Enum):
    """Priority levels for context retention."""
    SYSTEM = 100     # System prompts, instructions (highest)
    PINNED = 90      # User-pinned context
    RECENT = 80      # Most recent interactions
    TOOL = 70        # Tool results
    USER = 60        # User messages
    ASSISTANT = 50   # Assistant responses
    SUMMARIZED = 40  # Summarized older context
    LOW = 10         # Low priority, evict first
    
    @property
    def weight(self) -> float:
        """Weight for budget allocation."""
        return self.value / 100.0


# =============================================================================
# TRUNCATION STRATEGY
# =============================================================================
class TruncationStrategy(Enum):
    """Strategy for truncating oversized context."""
    HEAD = auto()       # Remove from beginning
    TAIL = auto()       # Remove from end
    MIDDLE_OUT = auto() # Keep beginning and end, remove middle
    SUMMARIZE = auto()  # Summarize truncated content
    PRIORITY = auto()   # Remove by priority (lowest first)
    
    def describe(self) -> str:
        """Human-readable description."""
        return {
            TruncationStrategy.HEAD: "Remove oldest content first",
            TruncationStrategy.TAIL: "Remove newest content first",
            TruncationStrategy.MIDDLE_OUT: "Keep first and last, remove middle",
            TruncationStrategy.SUMMARIZE: "Summarize truncated content",
            TruncationStrategy.PRIORITY: "Remove lowest priority first",
        }[self]


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass(frozen=True, slots=True)
class ContextWindowConfig:
    """Configuration for context window."""
    max_tokens: int = 8192            # Maximum context tokens
    reserve_tokens: int = 512          # Reserve for response
    system_budget_ratio: float = 0.1   # 10% for system prompt
    recent_budget_ratio: float = 0.5   # 50% for recent context
    truncation_strategy: TruncationStrategy = TruncationStrategy.PRIORITY
    include_timestamps: bool = False   # Include timestamps in context
    
    @property
    def available_tokens(self) -> int:
        """Tokens available after reserve."""
        return self.max_tokens - self.reserve_tokens
    
    @property
    def system_budget(self) -> int:
        """Token budget for system context."""
        return int(self.available_tokens * self.system_budget_ratio)
    
    @property
    def recent_budget(self) -> int:
        """Token budget for recent context."""
        return int(self.available_tokens * self.recent_budget_ratio)


# =============================================================================
# CONTEXT ITEM
# =============================================================================
@dataclass(slots=True)
class ContextItem:
    """
    Item prepared for context window.
    
    Includes priority and token allocation.
    """
    item_id: str
    content: str  # Decoded and ready for prompt
    role: str
    token_count: int
    priority: ContextPriority
    sequence_id: int
    
    # Budget tracking
    allocated_tokens: int = 0
    is_truncated: bool = False
    truncation_ratio: float = 0.0


# =============================================================================
# CONTEXT WINDOW
# =============================================================================
class ContextWindow:
    """
    Token-aware context window manager.
    
    Features:
        - Priority-based retention
        - Intelligent truncation
        - Token budget allocation
        - Dynamic window sizing
    
    Usage:
        config = ContextWindowConfig(max_tokens=8192)
        window = ContextWindow(working_set, config)
        
        # Build context for model
        result = await window.build_context()
        if result.is_ok():
            items = result.unwrap()
            context_str = "\\n".join(i.content for i in items)
    """
    
    __slots__ = (
        "_working_set", "_config", "_lock",
        "_system_context", "_token_counter",
    )
    
    def __init__(
        self,
        working_set: WorkingSet,
        config: Optional[ContextWindowConfig] = None,
        token_counter: Optional[Any] = None,
    ) -> None:
        self._working_set = working_set
        self._config = config or ContextWindowConfig()
        self._lock = asyncio.Lock()
        self._system_context: Optional[str] = None
        self._token_counter = token_counter or self._default_token_counter
    
    @staticmethod
    def _default_token_counter(text: str) -> int:
        """Default token counter (approximation)."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def set_system_context(self, system_prompt: str) -> None:
        """Set the system context (highest priority)."""
        self._system_context = system_prompt
    
    async def build_context(
        self,
        target_tokens: Optional[int] = None,
    ) -> Result[list[ContextItem], str]:
        """
        Build context window from working set.
        
        Args:
            target_tokens: Target token count (defaults to config)
        
        Returns list of ContextItems within token budget.
        """
        async with self._lock:
            target = target_tokens or self._config.available_tokens
            items: list[ContextItem] = []
            remaining_tokens = target
            
            # Add system context first (highest priority)
            if self._system_context:
                system_tokens = self._token_counter(self._system_context)
                if system_tokens <= self._config.system_budget:
                    items.append(ContextItem(
                        item_id="system",
                        content=self._system_context,
                        role="system",
                        token_count=system_tokens,
                        priority=ContextPriority.SYSTEM,
                        sequence_id=-1,
                        allocated_tokens=system_tokens,
                    ))
                    remaining_tokens -= system_tokens
            
            # Get working set items
            ws_items = await self._working_set.get_recent(limit=1000)
            
            # Assign priorities
            prioritized = self._assign_priorities(ws_items)
            
            # Apply truncation strategy
            if self._config.truncation_strategy == TruncationStrategy.PRIORITY:
                # Sort by priority descending, then sequence_id descending
                prioritized.sort(
                    key=lambda x: (x[1].value, x[0].sequence_id),
                    reverse=True,
                )
            
            # Allocate tokens
            for ws_item, priority in prioritized:
                content = ws_item.decompress().decode("utf-8", errors="replace")
                token_count = self._token_counter(content)
                
                if token_count <= remaining_tokens:
                    # Fits completely
                    items.append(ContextItem(
                        item_id=ws_item.item_id,
                        content=content,
                        role=ws_item.role,
                        token_count=token_count,
                        priority=priority,
                        sequence_id=ws_item.sequence_id,
                        allocated_tokens=token_count,
                    ))
                    remaining_tokens -= token_count
                elif remaining_tokens > 100:
                    # Partial fit - truncate
                    truncated = self._truncate_content(
                        content, remaining_tokens
                    )
                    trunc_tokens = self._token_counter(truncated)
                    
                    items.append(ContextItem(
                        item_id=ws_item.item_id,
                        content=truncated,
                        role=ws_item.role,
                        token_count=token_count,
                        priority=priority,
                        sequence_id=ws_item.sequence_id,
                        allocated_tokens=trunc_tokens,
                        is_truncated=True,
                        truncation_ratio=1 - (trunc_tokens / token_count),
                    ))
                    remaining_tokens -= trunc_tokens
                    break
            
            # Sort final result by sequence_id ascending (chronological)
            items.sort(key=lambda x: x.sequence_id)
            
            return Ok(items)
    
    def _assign_priorities(
        self,
        items: list[WorkingSetItem],
    ) -> list[tuple[WorkingSetItem, ContextPriority]]:
        """Assign priorities to working set items."""
        result: list[tuple[WorkingSetItem, ContextPriority]] = []
        
        for item in items:
            priority = self._role_to_priority(item.role)
            
            # Boost recent items
            if item == items[0]:  # Most recent
                priority = ContextPriority.RECENT
            
            result.append((item, priority))
        
        return result
    
    def _role_to_priority(self, role: str) -> ContextPriority:
        """Map role to context priority."""
        return {
            "system": ContextPriority.SYSTEM,
            "user": ContextPriority.USER,
            "assistant": ContextPriority.ASSISTANT,
            "tool": ContextPriority.TOOL,
        }.get(role, ContextPriority.LOW)
    
    def _truncate_content(
        self,
        content: str,
        target_tokens: int,
    ) -> str:
        """Truncate content to fit token budget."""
        # Approximate character limit
        char_limit = target_tokens * 4
        
        if self._config.truncation_strategy == TruncationStrategy.HEAD:
            return content[-char_limit:]
        elif self._config.truncation_strategy == TruncationStrategy.TAIL:
            return content[:char_limit]
        elif self._config.truncation_strategy == TruncationStrategy.MIDDLE_OUT:
            half = char_limit // 2
            return content[:half] + "\n...[truncated]...\n" + content[-half:]
        else:
            # Default to tail truncation
            return content[:char_limit]
    
    async def get_token_usage(self) -> dict[str, int]:
        """Get current token usage by category."""
        async with self._lock:
            result = await self.build_context()
            if result.is_err():
                return {}
            
            items = result.unwrap()
            usage: dict[str, int] = {}
            
            for item in items:
                key = item.priority.name.lower()
                usage[key] = usage.get(key, 0) + item.allocated_tokens
            
            usage["total"] = sum(usage.values())
            usage["available"] = self._config.available_tokens
            usage["utilization"] = usage["total"] / usage["available"]
            
            return usage
    
    @property
    def config(self) -> ContextWindowConfig:
        """Current configuration."""
        return self._config
