"""
Short-Term Memory (STM) Module

Provides:
- WorkingSet: Sliding window buffer for recent interactions
- ContextWindow: Token-aware context management
- Eviction: LRU eviction policies with consolidation triggers
"""

from datamesh.memory.stm.working_set import (
    WorkingSet,
    WorkingSetConfig,
    WorkingSetItem,
)
from datamesh.memory.stm.context_window import (
    ContextWindow,
    ContextWindowConfig,
    TruncationStrategy,
    ContextPriority,
)
from datamesh.memory.stm.eviction import (
    EvictionPolicy,
    EvictionEvent,
    EvictionManager,
)

__all__ = [
    "WorkingSet",
    "WorkingSetConfig",
    "WorkingSetItem",
    "ContextWindow",
    "ContextWindowConfig",
    "TruncationStrategy",
    "ContextPriority",
    "EvictionPolicy",
    "EvictionEvent",
    "EvictionManager",
]
