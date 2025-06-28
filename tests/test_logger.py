"""
Tests for the logger implementation.
"""

import asyncio
import tempfile
import pytest

from lmlog import LLMLogger, AlwaysSampler, NeverSampler


class TestLLMLogger:
    """Test suite for LLMLogger."""

    def test_initialization(self):
        """Test logger initialization."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(tmp.name, async_processing=False, encoder="msgspec")
            assert logger._enabled is True
            assert logger._async_queue is None

    def test_sync_logging(self):
        """Test synchronous logging."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name,
                async_processing=False,
                encoder="msgspec",
                sampler=AlwaysSampler(),
            )

            logger.log_event(
                event_type="test_event",
                level="info",
                entity_type="test",
                entity_id="123",
                context={"key": "value"},
            )

            stats = logger.get_stats()
            assert stats["events_logged"] == 1

    @pytest.mark.asyncio
    async def test_async_logging(self):
        """Test asynchronous logging."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name,
                async_processing=True,
                encoder="msgspec",
                sampler=AlwaysSampler(),
            )

            await logger.alog_event(
                event_type="test_event",
                level="info",
                entity_type="test",
                entity_id="123",
                context={"key": "value"},
            )

            # Wait longer for processing and force flush by closing
            await asyncio.sleep(0.3)  # Give more time for processing
            await logger.close()  # This should flush any remaining events

            stats = logger.get_stats()
            assert stats["events_logged"] == 1

            # Also check if the async queue has stats
            if logger._async_queue:
                queue_stats = logger._async_queue.get_stats()
                assert queue_stats["events_processed"] >= 1

    def test_sampling_integration(self):
        """Test sampling integration."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(tmp.name, async_processing=False, sampler=NeverSampler())

            logger.log_event(event_type="test_event", level="info")

            stats = logger.get_stats()
            assert stats["events_sampled_out"] == 1
            assert stats["events_logged"] == 0

    def test_always_sampler(self):
        """Test always sampler."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            logger.log_event(event_type="test_event", level="info")

            stats = logger.get_stats()
            assert stats["events_logged"] == 1

    def test_state_change_logging(self):
        """Test state change logging."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            logger.log_state_change(
                entity_type="user",
                entity_id="123",
                field="status",
                before="inactive",
                after="active",
                trigger="user_login",
            )

            stats = logger.get_stats()
            assert stats["events_logged"] == 1

    def test_performance_issue_logging(self):
        """Test performance issue logging."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            logger.log_performance_issue(
                operation="database_query", duration_ms=5000, threshold_ms=1000
            )

            stats = logger.get_stats()
            assert stats["events_logged"] == 1

    def test_operation_context(self):
        """Test operation context manager."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            with logger.operation_context("test_op", user_id="123"):
                pass

            stats = logger.get_stats()
            assert stats["events_logged"] == 2  # start and end

    @pytest.mark.asyncio
    async def test_async_operation_context(self):
        """Test async operation context manager."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            async with logger.aoperation_context("test_op", user_id="123"):
                pass

            stats = logger.get_stats()
            assert stats["events_logged"] == 2  # start and end

    def test_operation_context_error(self):
        """Test operation context with error."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            try:
                with logger.operation_context("test_op"):
                    raise ValueError("Test error")
            except ValueError:
                pass

            stats = logger.get_stats()
            assert stats["events_logged"] == 3  # start, error, end

    def test_pool_statistics(self):
        """Test pool statistics."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            logger.log_event(event_type="test")
            stats = logger.get_stats()

            assert "event_pool_size" in stats
            assert "string_pool_size" in stats
            assert "caller_cache_info" in stats

    def test_sampler_update(self):
        """Test sampler update."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            logger.log_event(event_type="test")
            assert logger.get_stats()["events_logged"] == 1

            logger.set_sampler(NeverSampler())
            logger.log_event(event_type="test2")
            assert logger.get_stats()["events_sampled_out"] == 1

    def test_cache_clearing(self):
        """Test cache clearing."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            logger.log_event(event_type="test")
            logger.clear_caches()

            stats = logger.get_stats()
            cache_info = stats["caller_cache_info"]
            assert cache_info["currsize"] == 0

    def test_disabled_logger(self):
        """Test disabled logger."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(tmp.name, enabled=False, async_processing=False)

            logger.log_event(event_type="test")
            stats = logger.get_stats()
            assert stats["events_logged"] == 0

    def test_json_encoder(self):
        """Test JSON encoder."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            logger = LLMLogger(
                tmp.name,
                async_processing=False,
                encoder="json",
                sampler=AlwaysSampler(),
            )

            logger.log_event(event_type="test")
            stats = logger.get_stats()
            assert stats["events_logged"] == 1
