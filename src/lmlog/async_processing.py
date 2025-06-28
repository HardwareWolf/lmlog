"""
Asynchronous processing with intelligent batching and queuing.
"""

import asyncio
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Protocol, Callable
from enum import Enum


class ProcessingMode(Enum):
    """Processing modes for async handlers."""

    IMMEDIATE = "immediate"
    BATCHED = "batched"
    ADAPTIVE = "adaptive"


class EventProcessor(Protocol):
    """Protocol for event processors."""

    async def process_event(self, event: Dict[str, Any]) -> None:
        """Process a single event."""
        ...

    async def process_batch(self, events: List[Dict[str, Any]]) -> None:
        """Process a batch of events."""
        ...


class AsyncEventQueue:
    """
    High-performance async event queue with intelligent batching.
    """

    __slots__ = (
        "_queue",
        "_processor",
        "_max_batch_size",
        "_max_wait_time",
        "_running",
        "_worker_task",
        "_stats",
    )

    def __init__(
        self,
        processor: EventProcessor,
        max_batch_size: int = 100,
        max_wait_time: float = 1.0,
        queue_size: int = 10000,
    ):
        """
        Initialize async event queue.

        Args:
            processor: Event processor
            max_batch_size: Maximum events per batch
            max_wait_time: Maximum wait time before flushing batch
            queue_size: Maximum queue size
        """
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self._processor = processor
        self._max_batch_size = max_batch_size
        self._max_wait_time = max_wait_time
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._stats = {
            "events_processed": 0,
            "batches_processed": 0,
            "queue_full_errors": 0,
            "processing_errors": 0,
        }

    async def put(self, event: Dict[str, Any]) -> bool:
        """
        Add event to queue.

        Args:
            event: Event to process

        Returns:
            True if added successfully, False if queue full
        """
        try:
            await self._queue.put(event)
            return True
        except asyncio.QueueFull:
            self._stats["queue_full_errors"] += 1
            return False

    def put_nowait(self, event: Dict[str, Any]) -> bool:
        """
        Add event to queue without waiting.

        Args:
            event: Event to process

        Returns:
            True if added successfully, False if queue full
        """
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._stats["queue_full_errors"] += 1
            return False

    async def start(self) -> None:
        """Start the async processor."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the async processor and wait for completion."""
        if not self._running:
            return

        self._running = False

        if self._worker_task:
            await self._worker_task
            self._worker_task = None

    async def _worker(self) -> None:
        """Main worker loop for processing events."""
        batch = []
        last_flush_time = time.time()

        while self._running or not self._queue.empty():
            try:
                time_since_flush = time.time() - last_flush_time
                timeout = max(0.01, self._max_wait_time - time_since_flush)

                event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                batch.append(event)

                should_flush = (
                    len(batch) >= self._max_batch_size
                    or time.time() - last_flush_time >= self._max_wait_time
                )

                if should_flush:
                    await self._process_batch(batch)
                    batch.clear()
                    last_flush_time = time.time()

            except asyncio.TimeoutError:
                if batch:
                    await self._process_batch(batch)
                    batch.clear()
                    last_flush_time = time.time()

        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of events."""
        if not batch:
            return

        try:
            await self._processor.process_batch(batch.copy())
            self._stats["events_processed"] += len(batch)
            self._stats["batches_processed"] += 1
        except Exception:
            self._stats["processing_errors"] += 1

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return self._stats.copy()

    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


class AdaptiveBatcher:
    """
    Adaptive batching system that adjusts batch size based on throughput.
    """

    __slots__ = (
        "_min_batch_size",
        "_max_batch_size",
        "_current_batch_size",
        "_target_latency",
        "_adjustment_factor",
        "_latency_history",
        "_last_adjustment",
    )

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
        target_latency: float = 0.1,
        adjustment_factor: float = 0.1,
    ):
        """
        Initialize adaptive batcher.

        Args:
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_latency: Target processing latency in seconds
            adjustment_factor: How aggressively to adjust batch size
        """
        self._min_batch_size = min_batch_size
        self._max_batch_size = max_batch_size
        self._current_batch_size = min_batch_size
        self._target_latency = target_latency
        self._adjustment_factor = adjustment_factor
        self._latency_history: deque[float] = deque(maxlen=10)
        self._last_adjustment = time.time()

    def record_latency(self, latency: float) -> None:
        """
        Record processing latency.

        Args:
            latency: Processing latency in seconds
        """
        self._latency_history.append(latency)

        if time.time() - self._last_adjustment > 5.0:
            self._adjust_batch_size()
            self._last_adjustment = time.time()

    def get_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self._current_batch_size

    def _adjust_batch_size(self) -> None:
        """Adjust batch size based on latency history."""
        if len(self._latency_history) < 3:
            return

        avg_latency = sum(self._latency_history) / len(self._latency_history)

        if avg_latency > self._target_latency * 1.2:
            new_size = int(self._current_batch_size * (1.0 - self._adjustment_factor))
            self._current_batch_size = max(self._min_batch_size, new_size)
        elif avg_latency < self._target_latency * 0.8:
            new_size = int(self._current_batch_size * (1.0 + self._adjustment_factor))
            self._current_batch_size = min(self._max_batch_size, new_size)


class CircuitBreaker:
    """
    Circuit breaker pattern for async processing resilience.
    """

    __slots__ = (
        "_failure_threshold",
        "_recovery_timeout",
        "_failure_count",
        "_last_failure_time",
        "_state",
        "_success_count_in_half_open",
    )

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying half-open state
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = "closed"
        self._success_count_in_half_open = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self._state == "open":
            if time.time() - self._last_failure_time > self._recovery_timeout:
                self._state = "half_open"
                self._success_count_in_half_open = 0
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)

            if self._state == "half_open":
                self._success_count_in_half_open += 1
                if self._success_count_in_half_open >= 3:
                    self._state = "closed"
                    self._failure_count = 0

            return result

        except Exception as e:
            self._record_failure()
            raise e

    def _record_failure(self) -> None:
        """Record a failure and update circuit state."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self._failure_threshold:
            self._state = "open"

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    def get_failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count


class BackpressureManager:
    """
    Manages backpressure for async processing systems.
    """

    __slots__ = (
        "_max_queue_size",
        "_high_water_mark",
        "_low_water_mark",
        "_current_pressure",
        "_callbacks",
    )

    def __init__(
        self,
        max_queue_size: int = 10000,
        high_water_mark: float = 0.8,
        low_water_mark: float = 0.5,
    ):
        """
        Initialize backpressure manager.

        Args:
            max_queue_size: Maximum queue size
            high_water_mark: Fraction at which to apply backpressure
            low_water_mark: Fraction at which to relieve backpressure
        """
        self._max_queue_size = max_queue_size
        self._high_water_mark = high_water_mark
        self._low_water_mark = low_water_mark
        self._current_pressure = 0.0
        self._callbacks: List[Callable[[float], None]] = []

    def update_queue_size(self, current_size: int) -> None:
        """
        Update current queue size and calculate pressure.

        Args:
            current_size: Current queue size
        """
        old_pressure = self._current_pressure
        self._current_pressure = current_size / self._max_queue_size

        if (
            old_pressure < self._high_water_mark
            and self._current_pressure >= self._high_water_mark
        ):
            self._notify_callbacks()
        elif (
            old_pressure > self._low_water_mark
            and self._current_pressure <= self._low_water_mark
        ):
            self._notify_callbacks()

    def get_pressure(self) -> float:
        """Get current backpressure (0.0 to 1.0)."""
        return self._current_pressure

    def is_high_pressure(self) -> bool:
        """Check if system is under high pressure."""
        return self._current_pressure >= self._high_water_mark

    def should_drop_events(self) -> bool:
        """Check if events should be dropped due to pressure."""
        return self._current_pressure >= 0.95

    def add_callback(self, callback: Callable[[float], None]) -> None:
        """
        Add callback for pressure changes.

        Args:
            callback: Function to call when pressure changes
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self) -> None:
        """Notify all callbacks of pressure change."""
        for callback in self._callbacks:
            try:
                callback(self._current_pressure)
            except Exception:
                pass


class PriorityQueue:
    """
    Priority queue for event processing with multiple priority levels.
    """

    __slots__ = ("_queues", "_priorities", "_lock")

    def __init__(self, priorities: Optional[List[str]] = None):
        """
        Initialize priority queue.

        Args:
            priorities: List of priority levels (highest to lowest)
        """
        self._priorities = priorities or ["critical", "high", "normal", "low"]
        self._queues = {priority: deque() for priority in self._priorities}
        self._lock = threading.Lock()

    def put(self, event: Dict[str, Any], priority: str = "normal") -> None:
        """
        Add event with specified priority.

        Args:
            event: Event to add
            priority: Priority level
        """
        if priority not in self._priorities:
            priority = "normal"

        with self._lock:
            self._queues[priority].append(event)

    def get(self) -> Optional[Dict[str, Any]]:
        """
        Get highest priority event.

        Returns:
            Event or None if queue is empty
        """
        with self._lock:
            for priority in self._priorities:
                queue = self._queues[priority]
                if queue:
                    return queue.popleft()
        return None

    def size(self) -> int:
        """Get total number of events."""
        with self._lock:
            return sum(len(queue) for queue in self._queues.values())

    def size_by_priority(self) -> Dict[str, int]:
        """Get event count by priority."""
        with self._lock:
            return {priority: len(queue) for priority, queue in self._queues.items()}
