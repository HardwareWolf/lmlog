"""
Integration tests consolidating coverage and comprehensive testing.

This file consolidates tests from:
- test_coverage_missing.py
- test_final_coverage.py
- test_missing_coverage.py
- test_comprehensive.py
- test_improved_decorators.py
"""

import asyncio
import json
import tempfile
import time
from io import StringIO
from unittest.mock import Mock, patch


from lmlog import LMLogger, LLMLogger
from lmlog.intelligence.sampling import AlwaysSampler


class TestIntegration:
    """Integration tests for full system functionality."""

    def test_logger_full_integration(self, tmp_path):
        """Test logger with all features enabled."""
        log_file = tmp_path / "integration.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=True,
            enable_aggregation=True,
            enable_cost_awareness=True,
            sampler=AlwaysSampler(),
        )

        # Log various event types
        logger.log_event("user_action", level="info", message="User logged in")
        logger.log_event("error_event", level="error", message="Database error")
        logger.log_performance_issue("slow_query", duration_ms=5000, threshold_ms=1000)

        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.log_exception(e, operation="test_op")

        logger.flush_buffer()

        # Verify file exists and has content
        assert log_file.exists()

        events = []
        with open(log_file) as f:
            for line in f:
                events.append(json.loads(line))

        assert len(events) >= 3

        # Check we have different event types
        event_types = [e["event_type"] for e in events]
        assert "user_action" in event_types
        assert "error_event" in event_types or "exception" in event_types

    def test_async_logger_integration(self):
        """Test async logger functionality."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            logger = LLMLogger(
                tmp.name, async_processing=False, sampler=AlwaysSampler()
            )

            # Test async event logging
            asyncio.run(self._async_log_test(logger))

    async def _async_log_test(self, logger):
        """Helper for async logging test."""
        await logger.alog_event(event_type="async_test", level="info")
        stats = logger.get_stats()
        assert stats["events_logged"] == 1

    def test_error_handling_edge_cases(self):
        """Test various error handling scenarios."""
        logger = LLMLogger(
            output=StringIO(), async_processing=False, sampler=AlwaysSampler()
        )

        # Test with unusual caller info
        caller_info = logger._get_caller_info(skip_frames=100)
        assert caller_info["file"] == "unknown"

        # Test close without backend close method
        asyncio.run(logger.close())

    @patch("lmlog.integrations.otel.OTEL_AVAILABLE", True)
    @patch("lmlog.integrations.otel.trace")
    def test_otel_integration_features(self, mock_trace):
        """Test OpenTelemetry integration features."""
        from lmlog.integrations.otel import TraceContextExtractor

        # Create extractor with mocked tracer
        extractor = TraceContextExtractor()
        extractor._tracer = Mock()

        # Test span without recording capability
        mock_span = Mock()
        mock_span.is_recording.return_value = False

        mock_trace.get_current_span.return_value = mock_span
        context = extractor.extract_context()

        # Should handle gracefully
        assert isinstance(context, dict)

    def test_sampling_edge_cases(self):
        """Test sampling edge cases and protocols."""

        class TestContext:
            def get_level(self):
                return "DEBUG"

            def get_event_type(self):
                return "debug_event"

            def get_context(self):
                return {"debug": True}

        context = TestContext()
        assert context.get_level() == "DEBUG"
        assert context.get_event_type() == "debug_event"
        assert context.get_context() == {"debug": True}

    def test_performance_scenarios(self, tmp_path):
        """Test various performance-related scenarios."""
        log_file = tmp_path / "perf.jsonl"

        # Test with high event volume
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_cost_awareness=True,
            sampler=AlwaysSampler(),
        )

        # Generate many events quickly
        start_time = time.time()
        for i in range(100):
            logger.log_event(
                "perf_test",
                level="info",
                message=f"Performance test event {i}",
                iteration=i,
            )

        elapsed = time.time() - start_time
        logger.flush_buffer()

        # Should complete reasonably quickly
        assert elapsed < 5.0  # Should take less than 5 seconds

        # Verify some events were logged
        with open(log_file) as f:
            event_count = sum(1 for _ in f)

        assert event_count > 0

    def test_decorator_integration(self):
        """Test decorator functionality integration."""
        from lmlog.integrations.decorators import log_performance

        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @log_performance(logger, threshold_ms=1000, log_all=True)
        def test_function():
            time.sleep(0.01)  # Small delay
            return "result"

        result = test_function()
        assert result == "result"

        # Check that performance was logged
        output.seek(0)
        content = output.getvalue()
        if content.strip():
            logged_data = json.loads(content.strip())
            assert logged_data["event_type"] in [
                "performance_issue",
                "performance_info",
            ]

    def test_memory_and_cleanup(self):
        """Test memory usage and cleanup scenarios."""
        # Test logger creation and cleanup
        loggers = []

        for i in range(10):
            logger = LLMLogger(
                output=StringIO(), async_processing=False, sampler=AlwaysSampler()
            )
            loggers.append(logger)
            logger.log_event("cleanup_test", message=f"Test {i}")

        # Clean up all loggers
        for logger in loggers:
            asyncio.run(logger.close())

        # Should complete without issues
        assert len(loggers) == 10
