"""
LLM logger with high-performance optimizations.
"""

import time
import threading
from datetime import datetime, UTC
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union, TextIO, List, Protocol, runtime_checkable
from contextlib import contextmanager, asynccontextmanager

from .pools import get_event_pool, get_string_pool, ObjectPool
from .sampling import (
    Sampler,
    LogLevel,
    create_smart_sampler,
)
from .async_processing import AsyncEventQueue, CircuitBreaker
from .otel_integration import (
    extract_trace_context,
    get_correlation_context,
    get_metric_generator,
    get_resource_detector,
)
from .serializers import MsgSpecEncoder, FastJSONEncoder
from .backends import FileBackend, StreamBackend


@runtime_checkable
class LogBackend(Protocol):
    """Protocol for logging backends."""

    def write(self, event: Dict[str, Any]) -> None:
        """Write event to backend."""
        ...

    async def awrite(self, event: Dict[str, Any]) -> None:
        """Write event to backend asynchronously."""
        ...

    def flush(self) -> None:
        """Flush any buffered data."""
        ...

    def close(self) -> None:
        """Close the backend."""
        ...


@runtime_checkable
class LogEncoder(Protocol):
    """Protocol for log encoders."""

    def encode(self, event: Any) -> bytes:
        """Encode event to bytes."""
        ...


class LogEventContext:
    """Implementation of SamplingContext protocol for log events."""

    __slots__ = ("level", "event_type", "context")

    def __init__(self, level: LogLevel, event_type: str, context: Dict[str, Any]):
        """
        Initialize log event context.

        Args:
            level: Log level
            event_type: Event type
            context: Event context
        """
        self.level = level
        self.event_type = event_type
        self.context = context

    def get_level(self) -> LogLevel:
        """Get the log level."""
        return self.level

    def get_event_type(self) -> str:
        """Get the event type."""
        return self.event_type

    def get_context(self) -> Dict[str, Any]:
        """Get additional context."""
        return self.context


class EventProcessor:
    """Event processor for the logger."""

    __slots__ = ("_backend", "_encoder", "_circuit_breaker", "_metric_generator")

    def __init__(self, backend, encoder):
        """
        Initialize event processor.

        Args:
            backend: Backend for writing events
            encoder: Encoder for serializing events
        """
        self._backend = backend
        self._encoder = encoder
        self._circuit_breaker = CircuitBreaker()
        self._metric_generator = get_metric_generator()

    async def process_event(self, event: Dict[str, Any]) -> None:
        """Process a single event."""
        await self.process_batch([event])

    async def process_batch(self, events: List[Dict[str, Any]]) -> None:
        """Process a batch of events."""
        await self._circuit_breaker.call(self._write_batch, events)

    async def _write_batch(self, events: List[Dict[str, Any]]) -> None:
        """Write a batch of events."""
        for event in events:
            try:
                await self._backend.awrite(event)
                self._metric_generator.generate_from_event(event)
            except Exception:
                continue


class LLMLogger:
    """
    LLM logger with high-performance optimizations.

    Features:
    - Object pooling for zero-allocation logging
    - Adaptive sampling for intelligent volume control
    - Async processing with circuit breaker resilience
    - OpenTelemetry integration for trace correlation
    - Memory optimization with string interning
    """

    __slots__ = (
        "_backend",
        "_enabled",
        "_global_context",
        "_sampler",
        "_async_queue",
        "_event_pool",
        "_string_pool",
        "_caller_cache",
        "_lock",
        "_encoder",
        "_correlation_context",
        "_resource_info",
        "_stats",
        "_buffer_size",
        "_auto_flush",
        "_buffer",
    )

    def __init__(
        self,
        output: Union[str, Path, TextIO] = "llm_log.jsonl",
        enabled: bool = True,
        global_context: Optional[Dict[str, Any]] = None,
        sampler: Optional[Sampler] = None,
        async_processing: bool = True,
        encoder: str = "msgspec",
        max_events_per_second: int = 1000,
        buffer_size: int = 0,
        auto_flush: bool = True,
    ):
        """
        Initialize LLM logger.

        Args:
            output: Output destination
            enabled: Whether logging is enabled
            global_context: Global context for all events
            sampler: Sampling strategy
            async_processing: Enable async processing
            encoder: Encoder type ("msgspec" or "json")
            max_events_per_second: Target events per second for adaptive sampling
            buffer_size: Maximum number of events to buffer
            auto_flush: Whether to auto-flush the buffer
        """
        self._enabled = enabled
        self._global_context = global_context or {}
        self._buffer_size = buffer_size
        self._auto_flush = auto_flush
        self._buffer: List[Dict[str, Any]] = []

        if isinstance(output, (str, Path)):
            self._backend: LogBackend = FileBackend(output)
        else:
            self._backend = StreamBackend(output)

        if encoder == "msgspec":
            self._encoder: Union[MsgSpecEncoder, FastJSONEncoder] = MsgSpecEncoder()
        else:
            self._encoder = FastJSONEncoder()

        self._sampler = sampler or create_smart_sampler(
            target_rate=max_events_per_second
        )

        self._event_pool = get_event_pool()
        self._string_pool = get_string_pool()
        self._caller_cache = ObjectPool(dict, max_size=1000)
        self._lock = threading.Lock()

        self._correlation_context = get_correlation_context()
        self._resource_info = get_resource_detector().get_resource_info()

        self._stats = {
            "events_logged": 0,
            "events_sampled_out": 0,
            "async_queue_errors": 0,
            "pool_hits": 0,
            "pool_misses": 0,
        }

        if async_processing:
            processor = EventProcessor(self._backend, self._encoder)
            self._async_queue: Optional[AsyncEventQueue] = AsyncEventQueue(
                processor, max_batch_size=100
            )
        else:
            self._async_queue = None

    @lru_cache(maxsize=1000)
    def _get_caller_info_cached(
        self, filename: str, lineno: int, function: str
    ) -> Dict[str, Any]:
        """Get cached caller information."""
        return {
            "file": self._string_pool.intern(filename),
            "line": lineno,
            "function": self._string_pool.intern(function),
        }

    def _get_caller_info(self, skip_frames: int = 3) -> Dict[str, Any]:
        """Extract optimized caller information."""
        import inspect

        frame = inspect.currentframe()
        for _ in range(skip_frames):
            if frame is None:
                break
            frame = frame.f_back

        if frame is None:
            return {"file": "unknown", "line": 0, "function": "unknown"}

        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        function = frame.f_code.co_name

        return self._get_caller_info_cached(filename, lineno, function)

    def _create_base_event(self) -> Dict[str, Any]:
        """Create base event with pooled objects."""
        event = self._event_pool.acquire()
        self._stats["pool_hits"] += 1

        event.update(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "source": self._get_caller_info(),
                "resource": self._resource_info,
            }
        )

        if self._global_context:
            event["global_context"] = self._global_context

        correlation_context = self._correlation_context.get_context()
        if correlation_context:
            event["correlation"] = correlation_context

        trace_context = extract_trace_context()
        if trace_context:
            event["trace"] = trace_context

        return event

    def _should_sample_event(
        self, event_type: str, level: str, context: Dict[str, Any]
    ) -> bool:
        """Determine if event should be sampled."""
        log_level = getattr(LogLevel, level.upper(), LogLevel.INFO)
        sampling_context = LogEventContext(log_level, event_type, context)
        decision = self._sampler.should_sample(sampling_context)

        if not decision.should_sample:
            with self._lock:
                self._stats["events_sampled_out"] += 1

        return decision.should_sample

    def log_event(
        self,
        event_type: str,
        level: str = "info",
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Log a structured event with optimizations.

        Args:
            event_type: Type of event
            level: Log level
            entity_type: Type of entity involved
            entity_id: ID of entity involved
            context: Additional context
            **kwargs: Additional event fields
        """
        if not self._enabled:
            return

        context = context or {}

        if not self._should_sample_event(event_type, level, context):
            return

        event = self._create_base_event()

        event.update(
            {
                "event_type": self._string_pool.intern(event_type),
                "level": self._string_pool.intern(level.lower()),
            }
        )

        if entity_type:
            event["entity_type"] = self._string_pool.intern(entity_type)
        if entity_id:
            event["entity_id"] = str(entity_id)
        if context:
            event["context"] = context

        event.update(kwargs)

        self._write_event(event)
        with self._lock:
            self._stats["events_logged"] += 1

    def _write_event(self, event: Dict[str, Any]) -> None:
        """Write event using buffering, async queue, or direct backend."""
        if self._buffer_size > 0 and not self._auto_flush:
            with self._lock:
                self._buffer.append(event.copy())
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer_to_backend()
        else:
            if self._async_queue:
                self._ensure_async_queue_started()
                success = self._async_queue.put_nowait(event)
                if not success:
                    with self._lock:
                        self._stats["async_queue_errors"] += 1
                    self._write_event_sync(event)
            else:
                self._write_event_sync(event)

    def _ensure_async_queue_started(self) -> None:
        """Start async queue if not already started."""
        if self._async_queue and not self._async_queue.is_running:
            try:
                import asyncio

                asyncio.create_task(self._async_queue.start())
            except RuntimeError:
                pass

    def _write_event_sync(self, event: Dict[str, Any]) -> None:
        """Write event synchronously."""
        try:
            self._backend.write(event)
            get_metric_generator().generate_from_event(event)
        finally:
            self._event_pool.release(event)

    def _flush_buffer_to_backend(self) -> None:
        """Flush buffer contents to backend."""
        with self._lock:
            buffer_copy = self._buffer.copy()
            self._buffer.clear()
        for buffered_event in buffer_copy:
            if self._async_queue:
                self._ensure_async_queue_started()
                success = self._async_queue.put_nowait(buffered_event)
                if not success:
                    with self._lock:
                        self._stats["async_queue_errors"] += 1
                    self._backend.write(buffered_event)
            else:
                self._backend.write(buffered_event)

    async def alog_event(
        self,
        event_type: str,
        level: str = "info",
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Async version of log_event.

        Args:
            event_type: Type of event
            level: Log level
            entity_type: Type of entity involved
            entity_id: ID of entity involved
            context: Additional context
            **kwargs: Additional event fields
        """
        if not self._enabled:
            return

        context = context or {}

        if not self._should_sample_event(event_type, level, context):
            return

        event = self._create_base_event()

        event.update(
            {
                "event_type": self._string_pool.intern(event_type),
                "level": self._string_pool.intern(level.lower()),
            }
        )

        if entity_type:
            event["entity_type"] = self._string_pool.intern(entity_type)
        if entity_id:
            event["entity_id"] = str(entity_id)
        if context:
            event["context"] = context

        event.update(kwargs)

        if self._async_queue:
            if not self._async_queue.is_running:
                await self._async_queue.start()
            await self._async_queue.put(event)
        else:
            await self._write_event_async(event)

        with self._lock:
            self._stats["events_logged"] += 1

    async def _write_event_async(self, event: Dict[str, Any]) -> None:
        """Write event asynchronously."""
        try:
            if hasattr(self._backend, "awrite"):
                await self._backend.awrite(event)
            else:
                self._backend.write(event)
            get_metric_generator().generate_from_event(event)
        finally:
            self._event_pool.release(event)

    def log_state_change(
        self,
        entity_type: str,
        entity_id: str,
        field: str,
        before: Any,
        after: Any,
        trigger: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a state change event."""
        context = {
            "field": field,
            "before": before,
            "after": after,
        }
        if trigger:
            context["trigger"] = trigger

        self.log_event(
            event_type="state_change",
            entity_type=entity_type,
            entity_id=entity_id,
            context=context,
            **kwargs,
        )

    def log_performance_issue(
        self,
        operation: str,
        duration_ms: float,
        threshold_ms: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Log a performance issue."""
        perf_context = {
            "operation": operation,
            "duration_ms": duration_ms,
        }
        if threshold_ms:
            perf_context["threshold_ms"] = threshold_ms

        if "context" in kwargs:
            perf_context.update(kwargs.pop("context"))

        level = "warning" if threshold_ms and duration_ms > threshold_ms else "info"

        self.log_event(
            event_type="performance_issue", level=level, context=perf_context, **kwargs
        )

    def log_exception(
        self,
        exception: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        include_traceback: bool = True,
        **kwargs,
    ) -> None:
        """Log an exception."""
        import traceback

        exc_context = {
            "operation": operation,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
        }

        if include_traceback:
            exc_context["traceback"] = traceback.format_exc()

        if context:
            exc_context.update(context)

        self.log_event(
            event_type="exception", level="error", context=exc_context, **kwargs
        )

    @contextmanager
    def operation_context(self, operation_id: str, **context):
        """Context manager for operation tracking."""
        start_time = time.time()
        operation_context = {"operation_id": operation_id, **context}

        with self._correlation_context.with_context(**operation_context):
            self.log_event(event_type="operation_start", context=operation_context)

            try:
                yield operation_id
            except Exception as e:
                self.log_event(
                    event_type="operation_error",
                    level="error",
                    context={
                        **operation_context,
                        "error": str(e),
                        "duration_ms": (time.time() - start_time) * 1000,
                    },
                )
                raise
            finally:
                self.log_event(
                    event_type="operation_end",
                    context={
                        **operation_context,
                        "duration_ms": (time.time() - start_time) * 1000,
                    },
                )

    @asynccontextmanager
    async def aoperation_context(self, operation_id: str, **context):
        """Async context manager for operation tracking."""
        start_time = time.time()
        operation_context = {"operation_id": operation_id, **context}

        with self._correlation_context.with_context(**operation_context):
            await self.alog_event(
                event_type="operation_start", context=operation_context
            )

            try:
                yield operation_id
            except Exception as e:
                await self.alog_event(
                    event_type="operation_error",
                    level="error",
                    context={
                        **operation_context,
                        "error": str(e),
                        "duration_ms": (time.time() - start_time) * 1000,
                    },
                )
                raise
            finally:
                await self.alog_event(
                    event_type="operation_end",
                    context={
                        **operation_context,
                        "duration_ms": (time.time() - start_time) * 1000,
                    },
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        with self._lock:
            stats: Dict[str, Any] = self._stats.copy()
        stats["event_pool_size"] = self._event_pool.size()
        stats["string_pool_size"] = self._string_pool.size()
        stats["caller_cache_info"] = self._get_caller_info_cached.cache_info()._asdict()

        if self._async_queue:
            stats["async_queue"] = self._async_queue.get_stats()
            stats["async_queue_size"] = self._async_queue.qsize()

        return stats

    def set_sampler(self, sampler: Sampler) -> None:
        """Update the sampling strategy."""
        self._sampler = sampler

    def clear_caches(self) -> None:
        """Clear internal caches."""
        self._get_caller_info_cached.cache_clear()
        self._string_pool.clear()

    async def close(self) -> None:
        """Close the logger and cleanup resources."""
        if self._async_queue:
            await self._async_queue.stop()

        if hasattr(self._backend, "close"):
            self._backend.close()

    @property
    def enabled(self) -> bool:
        """Whether logging is enabled."""
        return self._enabled

    @property
    def global_context(self) -> Dict[str, Any]:
        """Global context for all events."""
        with self._lock:
            return self._global_context.copy()

    @property
    def buffer_size(self) -> int:
        """Maximum number of events to buffer."""
        return self._buffer_size

    @property
    def auto_flush(self) -> bool:
        """Whether to auto-flush the buffer."""
        return self._auto_flush

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)

    def flush_buffer(self) -> None:
        """Flush the buffer."""
        with self._lock:
            if not self._buffer:
                return
        self._flush_buffer_to_backend()

    def clear_buffer(self) -> None:
        """Clear the buffer without writing."""
        with self._lock:
            self._buffer.clear()

    def add_global_context(self, **kwargs) -> None:
        """Add key-value pairs to global context."""
        with self._lock:
            self._global_context.update(kwargs)

    def remove_global_context(self, key: str) -> None:
        """Remove a key from global context."""
        with self._lock:
            self._global_context.pop(key, None)

    def clear_global_context(self) -> None:
        """Clear all global context."""
        with self._lock:
            self._global_context.clear()

    def set_output(self, output: Union[str, Path, TextIO]) -> None:
        """Set new output destination."""
        if isinstance(output, (str, Path)):
            self._backend = FileBackend(output)
        else:
            self._backend = StreamBackend(output)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and flush buffer."""
        self.flush_buffer()
        return False
