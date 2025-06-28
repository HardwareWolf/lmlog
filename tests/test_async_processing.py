"""
Tests for async processing functionality.
"""

import asyncio
import pytest
import time

from lmlog.async_processing import (
    AsyncEventQueue,
    AdaptiveBatcher,
    CircuitBreaker,
    BackpressureManager,
    PriorityQueue,
)


class MockEventProcessor:
    """Mock event processor for testing."""

    def __init__(self):
        self.processed_events = []
        self.processed_batches = []
        self.should_fail = False

    async def process_event(self, event):
        if self.should_fail:
            raise ValueError("Mock failure")
        self.processed_events.append(event)

    async def process_batch(self, events):
        if self.should_fail:
            raise ValueError("Mock failure")
        self.processed_batches.append(events)


class TestAsyncEventQueue:
    """Test async event queue."""

    @pytest.mark.asyncio
    async def test_basic_processing(self):
        """Test basic event processing."""
        processor = MockEventProcessor()
        queue = AsyncEventQueue(processor, max_batch_size=2, max_wait_time=0.1)

        await queue.start()

        # Add events
        await queue.put({"event": "test1"})
        await queue.put({"event": "test2"})

        # Wait for processing - longer wait
        await asyncio.sleep(0.5)
        await queue.stop()

        assert len(processor.processed_batches) >= 1
        if processor.processed_batches:
            batch = processor.processed_batches[0]
            assert len(batch) > 0

    @pytest.mark.asyncio
    async def test_batch_size_trigger(self):
        """Test batch size trigger."""
        processor = MockEventProcessor()
        queue = AsyncEventQueue(processor, max_batch_size=2, max_wait_time=10.0)

        await queue.start()

        # Add exactly batch_size events
        await queue.put({"event": "test1"})
        await queue.put({"event": "test2"})

        # Wait for processing
        await asyncio.sleep(0.2)
        await queue.stop()

        assert len(processor.processed_batches) >= 1

    @pytest.mark.asyncio
    async def test_time_trigger(self):
        """Test time-based trigger."""
        processor = MockEventProcessor()
        queue = AsyncEventQueue(processor, max_batch_size=10, max_wait_time=0.1)

        await queue.start()

        # Add single event
        await queue.put({"event": "test1"})

        # Wait for time trigger
        await asyncio.sleep(0.2)
        await queue.stop()

        assert len(processor.processed_batches) >= 1
        assert len(processor.processed_batches[0]) == 1

    def test_put_nowait(self):
        """Test non-blocking put."""
        processor = MockEventProcessor()
        queue = AsyncEventQueue(processor, queue_size=1)

        # First should succeed
        result1 = queue.put_nowait({"event": "test1"})
        assert result1 is True

        # Second should fail (queue full)
        result2 = queue.put_nowait({"event": "test2"})
        assert result2 is False

        stats = queue.get_stats()
        assert stats["queue_full_errors"] == 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling."""
        processor = MockEventProcessor()
        processor.should_fail = True

        queue = AsyncEventQueue(processor, max_batch_size=1, max_wait_time=0.1)

        await queue.start()
        await queue.put({"event": "test1"})

        await asyncio.sleep(0.2)
        await queue.stop()

        stats = queue.get_stats()
        assert stats["processing_errors"] > 0


class TestAdaptiveBatcher:
    """Test adaptive batcher."""

    def test_initial_state(self):
        """Test initial state."""
        batcher = AdaptiveBatcher(min_batch_size=1, max_batch_size=100)
        assert batcher.get_batch_size() == 1

    def test_latency_recording(self):
        """Test latency recording and adjustment."""
        batcher = AdaptiveBatcher(
            min_batch_size=1,
            max_batch_size=10,
            target_latency=0.1,
            adjustment_factor=0.5,
        )

        initial_size = batcher.get_batch_size()

        # Record high latencies
        for _ in range(5):
            batcher.record_latency(0.2)  # Higher than target

        # Trigger adjustment
        time.sleep(6)  # Wait for adjustment interval
        batcher.record_latency(0.2)

        # Batch size should decrease
        new_size = batcher.get_batch_size()
        assert new_size <= initial_size

    def test_bounds_checking(self):
        """Test bounds checking."""
        batcher = AdaptiveBatcher(min_batch_size=5, max_batch_size=10)

        # Record very low latencies to trigger increase
        for _ in range(10):
            batcher.record_latency(0.001)

        time.sleep(6)
        batcher.record_latency(0.001)

        # Should not exceed max
        assert batcher.get_batch_size() <= 10


class TestCircuitBreaker:
    """Test circuit breaker."""

    @pytest.mark.asyncio
    async def test_closed_state(self):
        """Test closed state."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == "closed"

    @pytest.mark.asyncio
    async def test_failure_threshold(self):
        """Test failure threshold."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def fail_func():
            raise ValueError("Test failure")

        # First failure
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        assert breaker.get_state() == "closed"

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        assert breaker.get_state() == "open"

        # Next call should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await breaker.call(fail_func)

    @pytest.mark.asyncio
    async def test_half_open_recovery(self):
        """Test half-open recovery."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        async def fail_func():
            raise ValueError("Test failure")

        async def success_func():
            return "success"

        # Trigger open state
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        assert breaker.get_state() == "open"

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should allow test calls (half-open)
        result = await breaker.call(success_func)
        assert result == "success"

        # More successes should close circuit
        await breaker.call(success_func)
        await breaker.call(success_func)
        assert breaker.get_state() == "closed"


class TestBackpressureManager:
    """Test backpressure manager."""

    def test_pressure_calculation(self):
        """Test pressure calculation."""
        manager = BackpressureManager(
            max_queue_size=100, high_water_mark=0.8, low_water_mark=0.5
        )

        manager.update_queue_size(50)
        assert manager.get_pressure() == 0.5
        assert not manager.is_high_pressure()

        manager.update_queue_size(85)
        assert manager.get_pressure() == 0.85
        assert manager.is_high_pressure()

        manager.update_queue_size(95)
        assert manager.should_drop_events()

    def test_callback_notifications(self):
        """Test callback notifications."""
        manager = BackpressureManager(
            max_queue_size=100, high_water_mark=0.8, low_water_mark=0.5
        )

        callback_calls = []

        def callback(pressure):
            callback_calls.append(pressure)

        manager.add_callback(callback)

        # Trigger high pressure
        manager.update_queue_size(85)
        assert len(callback_calls) == 1

        # Trigger low pressure
        manager.update_queue_size(40)
        assert len(callback_calls) == 2


class TestPriorityQueue:
    """Test priority queue."""

    def test_basic_operations(self):
        """Test basic operations."""
        queue = PriorityQueue(["high", "normal", "low"])

        queue.put({"event": "normal"}, "normal")
        queue.put({"event": "high"}, "high")
        queue.put({"event": "low"}, "low")

        # Should get high priority first
        event1 = queue.get()
        assert event1["event"] == "high"

        event2 = queue.get()
        assert event2["event"] == "normal"

        event3 = queue.get()
        assert event3["event"] == "low"

        assert queue.get() is None  # Empty

    def test_size_tracking(self):
        """Test size tracking."""
        queue = PriorityQueue()

        assert queue.size() == 0

        queue.put({"event": "test1"}, "high")
        queue.put({"event": "test2"}, "normal")

        assert queue.size() == 2

        sizes = queue.size_by_priority()
        assert sizes["high"] == 1
        assert sizes["normal"] == 1
        assert sizes["low"] == 0

    def test_invalid_priority(self):
        """Test invalid priority handling."""
        queue = PriorityQueue(["high", "normal", "low"])

        queue.put({"event": "test"}, "invalid")

        # Should default to normal
        sizes = queue.size_by_priority()
        assert sizes["normal"] == 1
