"""
Session State Machine: Lifecycle FSM with Guard Conditions

States:
    INITIATED  → Initial state after creation request
    ACTIVE     → Session actively processing interactions
    PAUSED     → Temporarily suspended (user-initiated)
    SUSPENDED  → System-initiated suspension (resource constraints)
    TERMINATED → Final state, session data archived

Transitions:
    INITIATED  → ACTIVE     : On first interaction
    ACTIVE     → PAUSED     : User pause request
    ACTIVE     → SUSPENDED  : System resource pressure
    ACTIVE     → TERMINATED : Explicit termination or timeout
    PAUSED     → ACTIVE     : User resume request
    PAUSED     → TERMINATED : Timeout expiration
    SUSPENDED  → ACTIVE     : Resource recovery
    SUSPENDED  → TERMINATED : Extended suspension timeout

Design:
    - Immutable state objects prevent race conditions
    - Guard conditions validate transition preconditions
    - Event emission enables reactive architectures
    - Version vectors support optimistic concurrency
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Sequence

from datamesh.core.types import (
    Result, Ok, Err,
    EntityId, Timestamp,
)


# =============================================================================
# SESSION STATE ENUMERATION
# =============================================================================
class SessionState(Enum):
    """
    Session lifecycle states.
    
    Ordered by typical progression; terminal state is TERMINATED.
    States are mutually exclusive (exactly one active at any time).
    """
    INITIATED = auto()   # Created but not yet active
    ACTIVE = auto()      # Processing interactions
    PAUSED = auto()      # User-initiated suspension
    SUSPENDED = auto()   # System-initiated suspension
    TERMINATED = auto()  # Final state (immutable)
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal (final) state."""
        return self == SessionState.TERMINATED
    
    @property
    def is_active(self) -> bool:
        """Check if session can process interactions."""
        return self == SessionState.ACTIVE
    
    @property
    def can_transition(self) -> bool:
        """Check if session can transition to another state."""
        return not self.is_terminal


# =============================================================================
# TRANSITION DEFINITIONS
# =============================================================================
@dataclass(frozen=True, slots=True)
class SessionTransition:
    """
    Represents a valid state transition.
    
    Immutable to prevent accidental modification during validation.
    """
    from_state: SessionState
    to_state: SessionState
    trigger: str  # Event/action that triggers this transition
    
    def __hash__(self) -> int:
        return hash((self.from_state, self.to_state, self.trigger))


# Valid transitions with their triggers
VALID_TRANSITIONS: frozenset[SessionTransition] = frozenset({
    # INITIATED transitions
    SessionTransition(SessionState.INITIATED, SessionState.ACTIVE, "FIRST_INTERACTION"),
    SessionTransition(SessionState.INITIATED, SessionState.TERMINATED, "CREATION_TIMEOUT"),
    
    # ACTIVE transitions
    SessionTransition(SessionState.ACTIVE, SessionState.PAUSED, "USER_PAUSE"),
    SessionTransition(SessionState.ACTIVE, SessionState.SUSPENDED, "SYSTEM_SUSPEND"),
    SessionTransition(SessionState.ACTIVE, SessionState.TERMINATED, "USER_TERMINATE"),
    SessionTransition(SessionState.ACTIVE, SessionState.TERMINATED, "IDLE_TIMEOUT"),
    
    # PAUSED transitions
    SessionTransition(SessionState.PAUSED, SessionState.ACTIVE, "USER_RESUME"),
    SessionTransition(SessionState.PAUSED, SessionState.TERMINATED, "PAUSE_TIMEOUT"),
    
    # SUSPENDED transitions
    SessionTransition(SessionState.SUSPENDED, SessionState.ACTIVE, "RESOURCE_RECOVERED"),
    SessionTransition(SessionState.SUSPENDED, SessionState.TERMINATED, "SUSPEND_TIMEOUT"),
})


# =============================================================================
# GUARD CONDITIONS
# =============================================================================
class TransitionGuard:
    """
    Guard condition for state transitions.
    
    Guards are evaluated before transition execution.
    All guards must pass for transition to proceed.
    """
    
    __slots__ = ("_name", "_predicate", "_error_message")
    
    def __init__(
        self,
        name: str,
        predicate: Callable[[SessionSnapshot], bool],
        error_message: str,
    ) -> None:
        self._name = name
        self._predicate = predicate
        self._error_message = error_message
    
    def evaluate(self, snapshot: SessionSnapshot) -> Result[None, str]:
        """
        Evaluate guard condition.
        
        Returns:
            Ok(None) if guard passes
            Err with message if guard fails
        """
        if self._predicate(snapshot):
            return Ok(None)
        return Err(f"Guard '{self._name}' failed: {self._error_message}")
    
    @property
    def name(self) -> str:
        return self._name


# =============================================================================
# SESSION SNAPSHOT (IMMUTABLE STATE CAPTURE)
# =============================================================================
@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """
    Immutable snapshot of session state.
    
    Used for guard evaluation and event emission.
    Frozen dataclass ensures no mutation during processing.
    """
    entity_id: EntityId
    session_id: str
    state: SessionState
    version: int
    created_at: Timestamp
    last_activity: Timestamp
    interaction_count: int
    accumulated_tokens: int
    metadata: tuple[tuple[str, Any], ...]  # Immutable dict representation
    
    @property
    def age_seconds(self) -> float:
        """Session age in seconds."""
        now = Timestamp.now()
        return (now.nanos - self.created_at.nanos) / 1_000_000_000
    
    @property
    def idle_seconds(self) -> float:
        """Time since last activity in seconds."""
        now = Timestamp.now()
        return (now.nanos - self.last_activity.nanos) / 1_000_000_000
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata value by key."""
        for k, v in self.metadata:
            if k == key:
                return v
        return default


# =============================================================================
# STATE MACHINE IMPLEMENTATION
# =============================================================================
@dataclass
class SessionContext:
    """
    Mutable session context (internal state).
    
    Only modified through state machine transitions.
    """
    entity_id: EntityId
    session_id: str
    state: SessionState = SessionState.INITIATED
    version: int = 0
    created_at: Timestamp = field(default_factory=Timestamp.now)
    last_activity: Timestamp = field(default_factory=Timestamp.now)
    interaction_count: int = 0
    accumulated_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def snapshot(self) -> SessionSnapshot:
        """Create immutable snapshot of current state."""
        return SessionSnapshot(
            entity_id=self.entity_id,
            session_id=self.session_id,
            state=self.state,
            version=self.version,
            created_at=self.created_at,
            last_activity=self.last_activity,
            interaction_count=self.interaction_count,
            accumulated_tokens=self.accumulated_tokens,
            metadata=tuple(self.metadata.items()),
        )


@dataclass(frozen=True, slots=True)
class StateTransitionEvent:
    """Event emitted on state transition."""
    session_id: str
    entity_id: EntityId
    from_state: SessionState
    to_state: SessionState
    trigger: str
    version: int
    timestamp: Timestamp


class SessionStateMachine:
    """
    Finite State Machine for session lifecycle management.
    
    Features:
        - Guard condition validation before transitions
        - Atomic state updates with version increments
        - Event emission for reactive architectures
        - Optimistic concurrency via version vectors
    
    Usage:
        ctx = SessionContext(entity_id=eid, session_id="sess-123")
        fsm = SessionStateMachine(ctx)
        
        result = fsm.transition("FIRST_INTERACTION")
        if result.is_ok():
            event = result.unwrap()
            publish(event)
    
    Thread Safety:
        External synchronization required for concurrent access.
        Use DistributedLease for distributed coordination.
    """
    
    __slots__ = ("_context", "_guards", "_listeners")
    
    def __init__(self, context: SessionContext) -> None:
        self._context = context
        self._guards: dict[SessionTransition, list[TransitionGuard]] = {}
        self._listeners: list[Callable[[StateTransitionEvent], None]] = []
        self._register_default_guards()
    
    def _register_default_guards(self) -> None:
        """Register built-in guard conditions."""
        # Guard: Cannot transition from terminal state
        terminal_guard = TransitionGuard(
            "not_terminal",
            lambda s: not s.state.is_terminal,
            "Session is in terminal state",
        )
        
        # Guard: Activity required for certain transitions
        activity_guard = TransitionGuard(
            "has_activity",
            lambda s: s.interaction_count > 0,
            "Session has no recorded interactions",
        )
        
        # Apply guards to relevant transitions
        for transition in VALID_TRANSITIONS:
            self._guards.setdefault(transition, []).append(terminal_guard)
            
            # Require activity for pause/suspend
            if transition.trigger in ("USER_PAUSE", "SYSTEM_SUSPEND"):
                self._guards[transition].append(activity_guard)
    
    def add_guard(
        self,
        from_state: SessionState,
        to_state: SessionState,
        trigger: str,
        guard: TransitionGuard,
    ) -> None:
        """Add custom guard condition to transition."""
        transition = SessionTransition(from_state, to_state, trigger)
        if transition in VALID_TRANSITIONS:
            self._guards.setdefault(transition, []).append(guard)
    
    def add_listener(
        self,
        listener: Callable[[StateTransitionEvent], None],
    ) -> None:
        """Register listener for state transition events."""
        self._listeners.append(listener)
    
    def transition(
        self,
        trigger: str,
        **metadata: Any,
    ) -> Result[StateTransitionEvent, str]:
        """
        Attempt state transition.
        
        Args:
            trigger: Transition trigger name
            **metadata: Additional context for guards
        
        Returns:
            Ok(event) on successful transition
            Err(message) on guard failure or invalid transition
        """
        current_state = self._context.state
        
        # Find valid transition for this trigger
        valid_transition: Optional[SessionTransition] = None
        for t in VALID_TRANSITIONS:
            if t.from_state == current_state and t.trigger == trigger:
                valid_transition = t
                break
        
        if valid_transition is None:
            return Err(
                f"No valid transition from {current_state.name} "
                f"with trigger '{trigger}'"
            )
        
        # Create snapshot for guard evaluation
        snapshot = self._context.snapshot()
        
        # Evaluate all guards
        guards = self._guards.get(valid_transition, [])
        for guard in guards:
            result = guard.evaluate(snapshot)
            if result.is_err():
                return result
        
        # Execute transition
        old_state = self._context.state
        self._context.state = valid_transition.to_state
        self._context.version += 1
        self._context.last_activity = Timestamp.now()
        
        # Update metadata
        for key, value in metadata.items():
            self._context.metadata[key] = value
        
        # Create event
        event = StateTransitionEvent(
            session_id=self._context.session_id,
            entity_id=self._context.entity_id,
            from_state=old_state,
            to_state=valid_transition.to_state,
            trigger=trigger,
            version=self._context.version,
            timestamp=Timestamp.now(),
        )
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # Swallow listener exceptions (logged externally)
        
        return Ok(event)
    
    def record_interaction(self, token_count: int) -> None:
        """
        Record interaction and update counters.
        
        Non-transitioning state update for activity tracking.
        """
        self._context.interaction_count += 1
        self._context.accumulated_tokens += token_count
        self._context.last_activity = Timestamp.now()
    
    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._context.state
    
    @property
    def version(self) -> int:
        """Current version for optimistic concurrency."""
        return self._context.version
    
    @property
    def snapshot(self) -> SessionSnapshot:
        """Immutable snapshot of current state."""
        return self._context.snapshot()
    
    @property
    def context(self) -> SessionContext:
        """Direct access to mutable context (use with caution)."""
        return self._context
    
    def can_transition(self, trigger: str) -> bool:
        """Check if transition is possible (without executing)."""
        for t in VALID_TRANSITIONS:
            if t.from_state == self._context.state and t.trigger == trigger:
                return True
        return False
    
    def available_triggers(self) -> list[str]:
        """List available triggers from current state."""
        return [
            t.trigger for t in VALID_TRANSITIONS
            if t.from_state == self._context.state
        ]
