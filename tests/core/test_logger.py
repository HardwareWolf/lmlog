"""
Tests for LMLogger with Phase 2 features.
"""

import json
from lmlog import LMLogger
from lmlog.intelligence.classification import EventType, EventPriority
from lmlog.intelligence.aggregation import AggregatedEvent
from lmlog.intelligence.cost_aware import CostBudget
from lmlog.intelligence.sampling import AlwaysSampler


class TestLMLogger:
    """Test LMLogger functionality."""

    def test_basic_logging(self, tmp_path):
        """Test basic logging functionality."""
        log_file = tmp_path / "test.jsonl"
        print(f"Test log file: {log_file}")
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=True,
            enable_aggregation=False,
            enable_cost_awareness=False,
            sampler=AlwaysSampler(),  # Always sample
        )
        print(f"Logger enabled: {logger._enabled}")

        logger.log_event("test_event", level="info", message="Test message")
        logger.flush_buffer()

        # Check stats
        stats = logger.get_stats()
        print(f"Stats: {stats}")
        print(f"Files in tmp_path: {list(tmp_path.iterdir())}")
        print(f"Log file exists: {log_file.exists()}")

        # Check file was created
        assert log_file.exists()

        # Check content
        with open(log_file) as f:
            line = json.loads(f.readline())
            assert line["event_type"] == "test_event"
            assert line["level"] == "info"

    def test_classification_feature(self, tmp_path):
        """Test ML-based event classification."""
        log_file = tmp_path / "test_classification.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=True,
            enable_aggregation=False,
            enable_cost_awareness=False,
            sampler=AlwaysSampler(),
        )

        # Log an error event
        logger.log_event(
            "database_error",
            level="error",
            message="Database connection timeout",
            error="TimeoutError",
        )
        logger.flush_buffer()

        # Check classification
        with open(log_file) as f:
            line = json.loads(f.readline())
            assert "classification" in line
            assert line["classification"]["type"] == EventType.ERROR.value
            assert line["classification"]["priority"] >= EventPriority.HIGH.value
            assert line["classification"]["confidence"] > 0.5

    def test_aggregation_feature(self, tmp_path):
        """Test pattern detection and aggregation."""
        log_file = tmp_path / "test_aggregation.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=False,
            enable_aggregation=True,
            enable_cost_awareness=False,
            aggregation_window=60,
            aggregation_threshold=0.8,
            sampler=AlwaysSampler(),
        )

        # Log similar events to trigger aggregation
        for i in range(15):
            logger.log_event(
                "user_action",
                level="info",
                message=f"User {i} performed login",
            )
        logger.flush_buffer()

        # Check for aggregated event
        aggregated_found = False
        with open(log_file) as f:
            for line in f:
                event = json.loads(line)
                if event.get("event_type") == "aggregated_event":
                    aggregated_found = True
                    assert event["aggregation"]["count"] >= 10
                    assert (
                        "User <NUMBER> performed login"
                        in event["aggregation"]["pattern"]
                    )
                    break

        assert aggregated_found

    def test_cost_awareness_feature(self, tmp_path):
        """Test cost-aware logging."""
        log_file = tmp_path / "test_cost.jsonl"

        # Very restrictive budget
        budget = CostBudget(
            max_daily_bytes=1024,  # 1KB
            max_events_per_second=10,
        )

        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=False,
            enable_aggregation=False,
            enable_cost_awareness=True,
            cost_budget=budget,
        )

        # Log many events
        logged_count = 0
        for i in range(100):
            logger.log_event(
                "test_event",
                level="info",
                message=f"Event {i}" * 10,  # Make it larger
            )

        # Count actual logged events
        if log_file.exists():
            with open(log_file) as f:
                logged_count = sum(1 for _ in f)

        # Should have sampled down due to budget
        assert logged_count < 100

    def test_all_features_enabled(self, tmp_path):
        """Test with all Phase 2 features enabled."""
        log_file = tmp_path / "test_all_features.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=True,
            enable_aggregation=True,
            enable_cost_awareness=True,
        )

        # Log various types of events
        events = [
            ("error", "Database connection failed", {"error": "ConnectionError"}),
            ("warning", "High memory usage detected", {"memory_percent": 85}),
            ("info", "User login successful", {"user_id": "123"}),
            ("info", "User login successful", {"user_id": "456"}),
            ("info", "User login successful", {"user_id": "789"}),
        ]

        for level, message, context in events:
            logger.log_event(
                "system_event",
                level=level,
                message=message,
                context=context,
            )

        # Get comprehensive stats
        stats = logger.get_stats()

        assert "classification" in stats
        assert "aggregation" in stats
        assert "cost_metrics" in stats
        assert "features" in stats
        assert stats["features"]["classification"] is True
        assert stats["features"]["aggregation"] is True
        assert stats["features"]["cost_awareness"] is True

    def test_feature_toggle(self, tmp_path):
        """Test enabling and disabling features."""
        log_file = tmp_path / "test_toggle.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=False,
            enable_aggregation=False,
            enable_cost_awareness=False,
        )

        # Initially all features disabled
        stats = logger.get_stats()
        assert stats["features"]["classification"] is False

        # Enable classification
        logger.enable_feature("classification")
        stats = logger.get_stats()
        assert stats["features"]["classification"] is True

        # Disable classification
        logger.disable_feature("classification")
        stats = logger.get_stats()
        assert stats["features"]["classification"] is False

    def test_get_classification_stats(self, tmp_path):
        """Test getting classification statistics."""
        log_file = tmp_path / "test_class_stats.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=True,
        )

        # Log some events
        for i in range(5):
            logger.log_event("test_event", level="info", message=f"Event {i}")

        stats = logger.get_classification_stats()
        assert stats is not None
        assert "cache_size" in stats
        assert "event_frequencies" in stats

    def test_get_aggregation_stats(self, tmp_path):
        """Test getting aggregation statistics."""
        log_file = tmp_path / "test_agg_stats.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_aggregation=True,
            sampler=AlwaysSampler(),
        )

        # Log some events with longer messages for aggregation
        for i in range(10):
            logger.log_event(
                "test_event",
                message=f"Similar message pattern number {i} for aggregation testing",
            )
        logger.flush_buffer()

        stats = logger.get_aggregation_stats()
        assert stats is not None
        assert "total_events" in stats
        assert stats["total_events"] >= 10

    def test_get_cost_metrics(self, tmp_path):
        """Test getting cost metrics."""
        log_file = tmp_path / "test_cost_metrics.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_cost_awareness=True,
            sampler=AlwaysSampler(),
        )

        # Log some events
        for i in range(5):
            logger.log_event("test_event", message=f"Event {i}")
        logger.flush_buffer()

        metrics = logger.get_cost_metrics()
        assert metrics is not None
        assert metrics.events_written >= 5
        assert metrics.bytes_written > 0

    def test_get_cost_forecast(self, tmp_path):
        """Test getting cost forecast."""
        log_file = tmp_path / "test_forecast.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_cost_awareness=True,
        )

        # Log some events
        for i in range(10):
            logger.log_event("test_event", message=f"Event {i}" * 10)

        forecast = logger.get_cost_forecast()
        assert forecast is not None
        assert "daily_volume_gb" in forecast
        assert "monthly_cost_hot" in forecast
        assert "budget_usage" in forecast

    def test_get_aggregated_events(self, tmp_path):
        """Test getting current aggregated events."""
        log_file = tmp_path / "test_get_agg.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_aggregation=True,
            aggregation_window=60,
        )

        # Log similar events
        for i in range(20):
            logger.log_event(
                "api_call",
                message=f"API /users/{i} called",
                duration_ms=100 + i,
            )

        aggregated = logger.get_aggregated_events()
        assert len(aggregated) > 0
        assert isinstance(aggregated[0], AggregatedEvent)

    def test_classification_affects_sampling(self, tmp_path):
        """Test that classification affects sampling decisions."""
        log_file = tmp_path / "test_class_sampling.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=True,
            enable_cost_awareness=True,
            cost_budget=CostBudget(
                max_daily_bytes=1024,  # Very small budget to trigger sampling
                max_events_per_second=5,
            ),
        )

        # Log high priority (error) and low priority (debug) events
        for i in range(10):
            logger.log_event(
                "error_event",
                level="error",
                message="Critical error occurred",
                error="SystemError",
            )

            logger.log_event(
                "debug_event",
                level="debug",
                message="Debug information",
            )
        logger.flush_buffer()

        # Count logged events by type
        error_count = 0
        debug_count = 0

        with open(log_file) as f:
            for line in f:
                event = json.loads(line)
                if "error" in event.get("event_type", ""):
                    error_count += 1
                elif "debug" in event.get("event_type", ""):
                    debug_count += 1

        # Errors should be logged more than debug events
        assert error_count >= debug_count

    def test_backward_compatibility(self, tmp_path):
        """Test backward compatibility with base logger."""
        log_file = tmp_path / "test_compat.jsonl"
        logger = LMLogger(
            output=str(log_file),
            async_processing=False,
            enable_classification=True,
            enable_aggregation=True,
            enable_cost_awareness=True,
            sampler=AlwaysSampler(),
        )

        # Use all base logger methods
        logger.log_state_change(
            entity_type="user",
            entity_id="123",
            field="status",
            before="active",
            after="inactive",
        )

        logger.log_performance_issue(
            operation="database_query",
            duration_ms=5000,
            threshold_ms=1000,
        )

        try:
            raise ValueError("Test exception")
        except Exception as e:
            logger.log_exception(
                e,
                operation="test_operation",
                include_traceback=True,
            )
        logger.flush_buffer()

        # Check all events were logged
        events = []
        with open(log_file) as f:
            for line in f:
                events.append(json.loads(line))

        assert len(events) >= 3
        event_types = [e["event_type"] for e in events]
        assert "state_change" in event_types
        assert "performance_issue" in event_types
        assert "exception" in event_types


class TestBaseLoggerProtocols:
    """Test protocol implementations in base_logger.py."""

    def test_log_backend_protocol(self):
        """Test LogBackend protocol methods."""
        from lmlog.core.base_logger import LogBackend
        
        class TestBackend:
            def write(self, event):
                pass

            async def awrite(self, event):
                pass

            def flush(self):
                pass

            def close(self):
                pass

        backend = TestBackend()
        assert isinstance(backend, LogBackend)

        # Test all protocol methods
        backend.write({"test": "event"})
        backend.flush()
        backend.close()
        import asyncio
        asyncio.run(backend.awrite({"test": "event"}))

    def test_log_encoder_protocol(self):
        """Test LogEncoder protocol methods."""
        from lmlog.core.base_logger import LogEncoder
        
        class TestEncoder:
            def encode(self, event):
                return json.dumps(event).encode()

        encoder = TestEncoder()
        assert isinstance(encoder, LogEncoder)

        # Test encode method
        result = encoder.encode({"test": "data"})
        assert isinstance(result, bytes)

    def test_log_event_context_methods(self):
        """Test LogEventContext methods."""
        from lmlog.core.base_logger import LogEventContext
        from lmlog.intelligence.sampling import LogLevel
        
        context = LogEventContext(
            level=LogLevel.ERROR, event_type="test_event", context={"key": "value"}
        )

        assert context.get_event_type() == "test_event"
        assert context.get_context() == {"key": "value"}


class TestLLMLoggerSampling:
    """Test sampling functionality specific to LLMLogger."""

    def test_global_context_in_event(self, tmp_path):
        """Test global context inclusion."""
        from io import StringIO
        from lmlog import LLMLogger
        
        output = StringIO()
        logger = LLMLogger(
            output=output,
            global_context={"app": "test", "version": "1.0"},
            async_processing=False,
            sampler=AlwaysSampler(),
        )

        logger.log_event("test_event")
        logger.flush_buffer()

        output.seek(0)
        event = json.loads(output.getvalue())
        assert event["global_context"]["app"] == "test"
        assert event["global_context"]["version"] == "1.0"

    def test_event_disabled(self, tmp_path):
        """Test logging when disabled."""
        from io import StringIO
        from lmlog import LLMLogger
        
        output = StringIO()
        logger = LLMLogger(output=output, enabled=False, async_processing=False)

        logger.log_event("test_event")
        logger.flush_buffer()

        output.seek(0)
        assert output.getvalue() == ""

    def test_event_sampled_out(self, tmp_path):
        """Test event sampling."""
        from io import StringIO
        from lmlog import LLMLogger
        from lmlog.intelligence.sampling import NeverSampler
        
        output = StringIO()
        logger = LLMLogger(
            output=output,
            async_processing=False,
            sampler=NeverSampler(),  # Always sample out
        )

        logger.log_event("test_event")
        logger.flush_buffer()

        output.seek(0)
        assert output.getvalue() == ""

        stats = logger.get_stats()
        assert stats["events_sampled_out"] > 0


class TestLLMLoggerEntityFields:
    """Test entity field handling."""

    def test_entity_fields(self, tmp_path):
        """Test entity_type and entity_id handling."""
        from io import StringIO
        from lmlog import LLMLogger
        
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        logger.log_event("test_event", entity_type="user", entity_id="12345")
        logger.flush_buffer()

        output.seek(0)
        event = json.loads(output.getvalue())
        assert event["entity_type"] == "user"
        assert event["entity_id"] == "12345"


class TestLLMLoggerBuffering:
    """Test buffering functionality."""

    def test_buffer_without_auto_flush(self, tmp_path):
        """Test buffering without auto-flush."""
        from io import StringIO
        from lmlog import LLMLogger
        
        output = StringIO()
        logger = LLMLogger(
            output=output,
            buffer_size=5,
            auto_flush=False,
            async_processing=False,
            sampler=AlwaysSampler(),
        )

        # Log 3 events (less than buffer size)
        for i in range(3):
            logger.log_event(f"event_{i}")

        # Should not be written yet
        output.seek(0)
        assert output.getvalue() == ""

        # Log 2 more to reach buffer size
        for i in range(3, 5):
            logger.log_event(f"event_{i}")

        # Should trigger flush
        output.seek(0)
        lines = output.getvalue().strip().split("\n")
        assert len(lines) >= 5


class TestLLMLoggerSpecialMethods:
    """Test special logging methods."""

    def test_log_state_change_with_trigger(self, tmp_path):
        """Test log_state_change with trigger."""
        from io import StringIO
        from lmlog import LLMLogger
        
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        logger.log_state_change(
            entity_type="user",
            entity_id="123",
            field="status",
            before="active",
            after="suspended",
            trigger="admin_action",
        )
        logger.flush_buffer()

        output.seek(0)
        event = json.loads(output.getvalue())
        assert event["event_type"] == "state_change"
        assert event["context"]["trigger"] == "admin_action"


class TestLLMLoggerContextManagers:
    """Test context managers."""

    def test_operation_context_success(self, tmp_path):
        """Test operation_context success path."""
        from io import StringIO
        from lmlog import LLMLogger
        import time
        
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        with logger.operation_context("test_op", user="alice") as op_id:
            assert op_id == "test_op"
            time.sleep(0.1)

        logger.flush_buffer()

        output.seek(0)
        lines = output.getvalue().strip().split("\n")
        events = [json.loads(line) for line in lines]

        # Should have start and end events
        assert any(e["event_type"] == "operation_start" for e in events)
        assert any(e["event_type"] == "operation_end" for e in events)

        # End event should have duration
        end_events = [e for e in events if e["event_type"] == "operation_end"]
        assert end_events[0]["context"]["duration_ms"] > 0

    def test_operation_context_error(self, tmp_path):
        """Test operation_context error path."""
        from io import StringIO
        from lmlog import LLMLogger
        import pytest
        
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        with pytest.raises(ValueError):
            with logger.operation_context("failing_op"):
                raise ValueError("Test error")

        logger.flush_buffer()

        output.seek(0)
        lines = output.getvalue().strip().split("\n")
        events = [json.loads(line) for line in lines]

        # Should have error event
        error_events = [e for e in events if e["event_type"] == "operation_error"]
        assert len(error_events) > 0
        assert error_events[0]["context"]["error"] == "Test error"


class TestLLMLoggerCacheAndSettings:
    """Test cache and settings methods."""

    def test_set_sampler(self, tmp_path):
        """Test set_sampler method."""
        from io import StringIO
        from lmlog import LLMLogger
        
        output = StringIO()
        logger = LLMLogger(output=output, async_processing=False)

        new_sampler = AlwaysSampler()
        logger.set_sampler(new_sampler)

        assert logger._sampler == new_sampler

    def test_global_context_methods(self, tmp_path):
        """Test global context manipulation."""
        from io import StringIO
        from lmlog import LLMLogger
        
        logger = LLMLogger(output=StringIO())

        # Test add_global_context
        logger.add_global_context(env="production", version="2.0")
        assert logger._global_context["env"] == "production"
        assert logger._global_context["version"] == "2.0"

        # Test remove_global_context
        logger.remove_global_context("env")
        assert "env" not in logger._global_context
        assert "version" in logger._global_context

        # Remove non-existent key (should not raise)
        logger.remove_global_context("nonexistent")

        # Test clear_global_context
        logger.clear_global_context()
        assert len(logger._global_context) == 0


class TestLMLoggerCoverage:
    """Test uncovered lines in LMLogger."""

    def test_log_event_disabled(self, tmp_path):
        """Test log_event when disabled."""
        from io import StringIO
        
        logger = LMLogger(output=StringIO(), enabled=False, async_processing=False)

        logger.log_event("test_event")

        # Should return early, no events logged
        stats = logger.get_stats()
        assert stats["events_logged"] == 0

    def test_cost_aware_sampling(self, tmp_path):
        """Test cost-aware sampling."""
        from io import StringIO
        
        output = StringIO()
        logger = LMLogger(
            output=output,
            enable_cost_awareness=True,
            cost_budget=CostBudget(
                max_daily_bytes=100, max_events_per_second=1  # Very small budget
            ),
            async_processing=False,
        )

        # Log many large events to exceed budget
        for i in range(50):
            logger.log_event(
                "large_event",
                message="x" * 1000,  # Large message
                priority=0.1,  # Low priority
            )

        stats = logger.get_stats()
        assert stats["events_sampled_out"] > 0

    def test_get_methods_without_features(self, tmp_path):
        """Test get methods when features are disabled."""
        from io import StringIO
        
        logger = LMLogger(
            output=StringIO(),
            enable_classification=False,
            enable_aggregation=False,
            enable_cost_awareness=False,
            async_processing=False,
        )

        assert logger.get_classification_stats() is None
        assert logger.get_aggregation_stats() is None
        assert logger.get_cost_metrics() is None
        assert logger.get_cost_forecast() is None
        assert logger.get_aggregated_events() == []

    def test_enable_feature_aggregation(self, tmp_path):
        """Test enabling aggregation feature."""
        from io import StringIO
        
        logger = LMLogger(
            output=StringIO(), enable_aggregation=False, async_processing=False
        )

        assert logger._aggregator is None

        logger.enable_feature("aggregation")

        assert logger._enable_aggregation is True
        assert logger._aggregator is not None

    def test_enable_feature_cost_awareness(self, tmp_path):
        """Test enabling cost_awareness feature."""
        from io import StringIO
        
        logger = LMLogger(
            output=StringIO(), enable_cost_awareness=False, async_processing=False
        )

        assert logger._cost_manager is None

        logger.enable_feature("cost_awareness")

        assert logger._enable_cost_awareness is True
        assert logger._cost_manager is not None

    def test_disable_features(self, tmp_path):
        """Test disabling features."""
        from io import StringIO
        
        logger = LMLogger(
            output=StringIO(),
            enable_classification=True,
            enable_aggregation=True,
            enable_cost_awareness=True,
            async_processing=False,
        )

        # All features should be enabled
        assert logger._enable_classification is True
        assert logger._enable_aggregation is True
        assert logger._enable_cost_awareness is True

        # Disable each feature
        logger.disable_feature("classification")
        assert logger._enable_classification is False

        logger.disable_feature("aggregation")
        assert logger._enable_aggregation is False

        logger.disable_feature("cost_awareness")
        assert logger._enable_cost_awareness is False


class TestLLMLoggerStats:
    """Test statistics methods."""

    def test_get_stats_with_async_queue(self, tmp_path):
        """Test get_stats with async queue info."""
        from io import StringIO
        from unittest.mock import Mock
        from lmlog import LLMLogger
        
        output = StringIO()
        logger = LLMLogger(output=output, async_processing=True)

        # Mock async queue
        logger._async_queue = Mock()
        logger._async_queue.get_stats = Mock(return_value={"queued": 5})
        logger._async_queue.qsize = Mock(return_value=3)

        stats = logger.get_stats()

        assert stats["async_queue"]["queued"] == 5
        assert stats["async_queue_size"] == 3


class TestLLMLoggerBufferOperations:
    """Test buffer-related operations."""

    def test_flush_buffer_to_backend(self, tmp_path):
        """Test _flush_buffer_to_backend method."""
        from io import StringIO
        from lmlog import LLMLogger
        from lmlog.intelligence.sampling import AlwaysSampler
        
        output = StringIO()
        logger = LLMLogger(
            output=output,
            buffer_size=10,
            auto_flush=False,
            async_processing=False,
            sampler=AlwaysSampler(),
        )

        # Add events to buffer
        for i in range(3):
            logger.log_event(f"buffered_event_{i}")

        # Manually flush
        logger.flush_buffer()

        output.seek(0)
        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 3

    def test_flush_buffer_with_async_queue(self, tmp_path):
        """Test buffer flush with async queue."""
        from io import StringIO
        from unittest.mock import Mock
        from lmlog import LLMLogger
        from lmlog.intelligence.sampling import AlwaysSampler
        
        output = StringIO()
        logger = LLMLogger(
            output=output,
            buffer_size=10,
            auto_flush=False,
            async_processing=True,
            sampler=AlwaysSampler(),
        )

        # Mock async queue to simulate full queue
        logger._async_queue = Mock()
        logger._async_queue.put_nowait = Mock(return_value=False)
        logger._async_queue.is_running = True

        # Add events to buffer
        for i in range(3):
            logger.log_event(f"event_{i}")

        # Flush should handle queue errors
        logger.flush_buffer()

        stats = logger.get_stats()
        assert stats["async_queue_errors"] >= 3

    def test_clear_caches(self, tmp_path):
        """Test clear_caches method."""
        from io import StringIO
        from lmlog import LLMLogger
        
        output = StringIO()
        logger = LLMLogger(output=output, async_processing=False)

        # Add some cached data
        logger._get_caller_info_cached("test.py", 10, "test_func")
        logger._string_pool.intern("test_string")

        # Clear caches
        logger.clear_caches()

        # Verify caches are cleared
        assert logger._get_caller_info_cached.cache_info().currsize == 0

    @pytest.mark.asyncio
    async def test_close_with_async_queue(self):
        """Test close method with async queue."""
        from io import StringIO
        from unittest.mock import Mock
        from lmlog import LLMLogger
        import asyncio
        
        output = StringIO()
        logger = LLMLogger(output=output, async_processing=True)

        # Mock async queue
        logger._async_queue = Mock()
        logger._async_queue.stop = Mock(return_value=asyncio.Future())
        logger._async_queue.stop.return_value.set_result(None)

        await logger.close()

        logger._async_queue.stop.assert_called()

    def test_enabled_property(self, tmp_path):
        """Test enabled property."""
        from io import StringIO
        from lmlog import LLMLogger
        
        logger = LLMLogger(output=StringIO(), enabled=True)
        assert logger.enabled is True

        logger = LLMLogger(output=StringIO(), enabled=False)
        assert logger.enabled is False

    def test_global_context_property(self, tmp_path):
        """Test global_context property."""
        from io import StringIO
        from lmlog import LLMLogger
        
        logger = LLMLogger(output=StringIO(), global_context={"app": "test"})

        context = logger.global_context
        assert context["app"] == "test"

        # Should return a copy
        context["modified"] = True
        assert "modified" not in logger.global_context

    def test_buffer_properties(self, tmp_path):
        """Test buffer-related properties."""
        from io import StringIO
        from lmlog import LLMLogger
        
        logger = LLMLogger(output=StringIO(), buffer_size=100, auto_flush=True)

        assert logger.buffer_size == 100
        assert logger.auto_flush is True

        # Test get_buffer_size
        assert logger.get_buffer_size() == 0

        # Add to buffer
        logger._buffer = [{"event": 1}, {"event": 2}]
        assert logger.get_buffer_size() == 2

    def test_flush_and_clear_buffer(self, tmp_path):
        """Test flush_buffer and clear_buffer."""
        from io import StringIO
        from lmlog import LLMLogger
        from lmlog.intelligence.sampling import AlwaysSampler
        
        output = StringIO()
        logger = LLMLogger(
            output=output,
            buffer_size=10,
            auto_flush=False,
            async_processing=False,
            sampler=AlwaysSampler(),
        )

        # Add events to buffer
        logger._buffer = [
            {"event_type": "test1", "timestamp": "2024-01-01"},
            {"event_type": "test2", "timestamp": "2024-01-01"},
        ]

        # Test flush with empty buffer
        logger._buffer = []
        logger.flush_buffer()  # Should return early

        # Re-add events
        logger._buffer = [
            {"event_type": "test1", "timestamp": "2024-01-01"},
            {"event_type": "test2", "timestamp": "2024-01-01"},
        ]

        # Test clear_buffer
        logger.clear_buffer()
        assert len(logger._buffer) == 0

        output.seek(0)
        assert output.getvalue() == ""  # Nothing written

    def test_set_output(self, tmp_path):
        """Test set_output method."""
        from io import StringIO
        from lmlog import LLMLogger
        from lmlog.processing.backends import FileBackend, StreamBackend
        import tempfile
        
        # Test with file path
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            logger = LLMLogger(output=StringIO())
            logger.set_output(tmp.name)

            assert isinstance(logger._backend, FileBackend)

        # Test with stream
        new_output = StringIO()
        logger.set_output(new_output)
        assert isinstance(logger._backend, StreamBackend)

    def test_context_manager(self, tmp_path):
        """Test context manager methods."""
        from io import StringIO
        from lmlog import LLMLogger
        from lmlog.intelligence.sampling import AlwaysSampler
        
        output = StringIO()
        logger = LLMLogger(
            output=output,
            buffer_size=10,
            auto_flush=False,
            async_processing=False,
            sampler=AlwaysSampler(),
        )

        with logger as ctx_logger:
            assert ctx_logger == logger
            ctx_logger._buffer = [{"event": "buffered"}]

        # Buffer should be flushed on exit
        output.seek(0)
        assert len(output.getvalue()) > 0


class TestEdgeCasesAndIntegration:
    """Additional edge cases and integration tests."""

    def test_logger_with_path_object(self, tmp_path):
        """Test logger with Path object."""
        from pathlib import Path
        from lmlog import LLMLogger
        from lmlog.intelligence.sampling import AlwaysSampler
        
        log_path = tmp_path / "test.jsonl"
        logger = LLMLogger(
            output=log_path, async_processing=False, sampler=AlwaysSampler()
        )

        logger.log_event("test_event")
        logger.flush_buffer()

        assert log_path.exists()

    def test_complex_event_serialization(self, tmp_path):
        """Test complex event serialization."""
        from io import StringIO
        from lmlog import LMLogger
        from lmlog.intelligence.sampling import AlwaysSampler
        import json
        
        output = StringIO()
        logger = LMLogger(
            output=output,
            async_processing=False,
            enable_aggregation=True,
            enable_classification=False,  # Disable classification to avoid hashing issues
            sampler=AlwaysSampler(),
        )

        # Log events with complex data (avoid sets which aren't JSON serializable)
        logger.log_event(
            "complex_event",
            nested_data={
                "list": [1, 2, {"key": "value"}],
                "tuple_as_list": list((1, 2, 3)),  # Convert tuple to list
                "numbers": [1, 2, 3],
            },
        )
        logger.flush_buffer()

        output.seek(0)
        event = json.loads(output.getvalue())
        assert event["event_type"] == "complex_event"

    @pytest.mark.asyncio
    async def test_async_context_manager_integration(self):
        """Test async context manager with real async operations."""
        from io import StringIO
        from lmlog import LLMLogger
        from lmlog.intelligence.sampling import AlwaysSampler
        import asyncio
        import json
        
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        results = []

        async with logger.aoperation_context("batch_process") as op_id:
            # Simulate batch processing
            for i in range(3):
                await asyncio.sleep(0.01)
                results.append(i)
                await logger.alog_event("item_processed", item_id=i, operation_id=op_id)

        await logger.close()

        assert len(results) == 3

        output.seek(0)
        lines = output.getvalue().strip().split("\n")
        events = [json.loads(line) for line in lines]

        # Should have operation start, 3 items, and operation end
        assert len(events) >= 5
        assert events[0]["event_type"] == "operation_start"
        assert events[-1]["event_type"] == "operation_end"
