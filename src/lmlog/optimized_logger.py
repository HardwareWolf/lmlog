"""
Optimized LLMLogger using modern Python 3.11+ performance techniques.
"""

import msgspec
from collections import deque
from datetime import datetime, UTC
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union, TextIO, List, Deque
from contextlib import contextmanager

from .serializers import FastJSONEncoder
from .backends import FileBackend, StreamBackend


class OptimizedLLMLogger:
    """
    High-performance LLMLogger implementation using modern Python optimizations.

    Key optimizations:
    - Uses msgspec for 10-80x faster serialization
    - Leverages functools.lru_cache for repeated operations
    - Preallocates buffers to avoid repeated appends
    - Minimizes attribute lookups in hot paths
    - Uses generators for memory efficiency
    - Supports batch operations with deque
    """

    __slots__ = (
        "_backend",
        "_enabled",
        "_global_context",
        "_buffer_size",
        "_buffer",
        "_lock",
        "_encoder",
        "_write_method",
        "_get_time",
    )

    def __init__(
        self,
        output: Union[str, Path, TextIO] = "llm_log.jsonl",
        enabled: bool = True,
        global_context: Optional[Dict[str, Any]] = None,
        buffer_size: int = 0,
        encoder: str = "msgspec",
        async_writes: bool = False,
    ):
        """
        Initialize optimized logger with performance-focused defaults.

        Args:
            output: Output destination
            enabled: Whether logging is enabled
            global_context: Context to include in all events
            buffer_size: Buffer size (0 = no buffering)
            encoder: Encoder type ("msgspec" or "orjson")
            async_writes: Enable async I/O for file writes
        """
        self._enabled = enabled
        self._global_context = global_context or {}
        self._buffer_size = buffer_size
        self._lock = Lock()

        # Preallocate buffer as deque for O(1) append/pop
        self._buffer: Deque[Dict[str, Any]] = deque(
            maxlen=buffer_size if buffer_size > 0 else None
        )

        # Initialize backend
        if isinstance(output, (str, Path)):
            self._backend: Union[FileBackend, StreamBackend] = FileBackend(
                output, encoder=encoder, async_writes=async_writes
            )
        else:
            self._backend = StreamBackend(output, encoder=encoder)

        # Cache encoder selection
        if encoder == "msgspec":
            self._encoder: Union[msgspec.json.Encoder, FastJSONEncoder] = (
                msgspec.json.Encoder()
            )
        else:
            self._encoder = FastJSONEncoder()

        # Cache method references to avoid attribute lookups
        self._write_method = self._backend.write
        self._get_time = lambda: datetime.now(UTC)

    @lru_cache(maxsize=128)
    def _get_caller_info_cached(
        self, filename: str, lineno: int, function: str
    ) -> Dict[str, Any]:
        """Cache caller info to avoid repeated path operations."""
        return {
            "source": f"{Path(filename).name}:{lineno}",
            "function": function,
            "line": lineno,
            "file": filename,
        }

    def _get_caller_info(self, skip_frames: int = 2) -> Dict[str, Any]:
        """Extract caller information with caching."""
        import inspect

        frame = inspect.currentframe()
        for _ in range(skip_frames):
            if frame is None:
                break
            frame = frame.f_back

        if frame is None:
            return {"source": "unknown", "function": "unknown", "line": 0}

        return self._get_caller_info_cached(
            frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name
        )

    def _create_event_base(self) -> Dict[str, Any]:
        """Create base event with minimal overhead."""
        # Use cached time getter
        timestamp = self._get_time().isoformat() + "Z"

        # Start with global context
        event = self._global_context.copy()
        event["timestamp"] = timestamp

        return event

    def log_event(
        self,
        event_type: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Log event with optimized performance.

        Uses preallocated structures and cached operations.
        """
        if not self._enabled:
            return

        # Create base event efficiently
        event = self._create_event_base()
        event["event_type"] = event_type
        event["source"] = self._get_caller_info()

        # Add optional fields only if present
        if entity_type or entity_id:
            entity = {}
            if entity_type:
                entity["type"] = entity_type
            if entity_id:
                entity["id"] = entity_id
            event["entity"] = entity

        if operation:
            event["operation"] = operation

        if context:
            event["context"] = context

        # Use generator expression for kwargs filtering
        event.update(
            (k, v) for k, v in kwargs.items() if v is not None and k not in event
        )

        self._write_event(event)

    def _write_event(self, event: Dict[str, Any]) -> None:
        """Write event using optimized path."""
        if self._buffer_size > 0:
            with self._lock:
                self._buffer.append(event)
                if len(self._buffer) >= self._buffer_size:
                    # Batch write all events
                    self._flush_buffer_optimized()
        else:
            # Direct write using cached method
            self._write_method(event)

    def _flush_buffer_optimized(self) -> None:
        """Optimized buffer flush using batch operations."""
        if not self._buffer:
            return

        # Process all events in one go
        events = list(self._buffer)
        self._buffer.clear()

        # Batch write
        for event in events:
            self._write_method(event)

        self._backend.flush()

    @lru_cache(maxsize=64)
    def _create_state_change_dict(
        self, field: str, before_hash: int, after_hash: int, trigger: str
    ) -> Dict[str, Any]:
        """Cache common state change structures."""
        return {"field": field, "trigger": trigger}

    def log_state_change(
        self,
        entity_type: str,
        entity_id: str,
        field: str,
        before: Any,
        after: Any,
        trigger: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log state change with caching for common patterns."""
        state_dict = self._create_state_change_dict(
            field,
            hash(str(before)) if before is not None else 0,
            hash(str(after)) if after is not None else 0,
            trigger,
        )

        # Add actual values
        state_dict["before"] = before
        state_dict["after"] = after

        self.log_event(
            event_type="state_change",
            entity_type=entity_type,
            entity_id=entity_id,
            state_change=state_dict,
            context=context,
        )

    def log_batch_events(self, events: List[Dict[str, Any]]) -> None:
        """
        Log multiple events efficiently in batch.

        Uses generator for memory efficiency and batch writes.
        """
        if not self._enabled:
            return

        # Process events as generator to avoid memory spike
        def event_generator():
            base_event = self._create_event_base()
            caller_info = self._get_caller_info()

            for event_data in events:
                event = base_event.copy()
                event.update(event_data)
                event["source"] = caller_info
                yield event

        # Batch write all events
        with self._lock:
            for event in event_generator():
                self._write_method(event)
            self._backend.flush()

    @contextmanager
    def operation_context(self, operation: str, **context):
        """Optimized context manager using cached operations."""
        start_time = self._get_time()
        operation_id = f"{operation}_{int(start_time.timestamp() * 1000000)}"

        # Cache these values to avoid repeated lookups
        log_method = self.log_event

        log_method(
            event_type="operation_start",
            operation=operation,
            operation_id=operation_id,
            context=context,
        )

        # Store original context efficiently
        operation_context_item = {"operation_id": operation_id}
        old_context = self._global_context.copy()
        self._global_context.update(operation_context_item)

        success = False

        try:
            yield operation_id
            success = True
        finally:
            # Restore context
            self._global_context = old_context

            # Calculate duration
            end_time = self._get_time()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            log_method(
                event_type="operation_end",
                operation=operation,
                operation_id=operation_id,
                success=success,
                duration_ms=duration_ms,
                context=context,
            )

    def flush(self) -> None:
        """Flush any buffered events."""
        with self._lock:
            self._flush_buffer_optimized()

    def enable(self) -> None:
        """Enable logging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable logging."""
        self._enabled = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.flush()
        return False
