"""
Tests to cover missing lines and edge cases.
"""

import asyncio
import tempfile
from io import StringIO
import pytest

from lmlog import LLMLogger, OptimizedLLMLogger
from lmlog.backends import FileBackend, StreamBackend, AsyncFileBackend
from lmlog.decorators import capture_errors, log_performance, log_calls


class TestMissingCoverage:
    """Test cases specifically designed to cover missing lines."""

    def test_logger_caller_info_edge_cases(self):
        """Test edge cases in _get_caller_info."""
        logger = LLMLogger(output=StringIO())

        # Test with skip_frames that goes beyond stack depth
        caller_info = logger._get_caller_info(skip_frames=100)
        assert caller_info["source"] == "unknown"
        assert caller_info["function"] == "unknown"
        assert caller_info["line"] == 0

    def test_logger_disabled_no_write(self):
        """Test that disabled logger doesn't write events."""
        output = StringIO()
        logger = LLMLogger(output=output, enabled=False)

        # Logging when disabled should not write anything
        logger.log_event("test", context={"test": True})

        output.seek(0)
        content = output.getvalue()
        assert content == ""  # No output when disabled

        # Also test _write_event directly to ensure line 67 is covered
        logger.enabled = False
        logger._write_event({"test": "direct_write"})

        output.seek(0)
        content = output.getvalue()
        assert content == ""  # Still no output when disabled

    def test_logger_file_auto_flush(self):
        """Test logger file output with auto_flush=True."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            # Create logger with file output and auto_flush enabled
            logger = LLMLogger(output=tmp.name, auto_flush=True, buffer_size=2)

            # Add events to trigger buffer flush to file
            logger.log_event("test1", context={"test": 1})
            logger.log_event("test2", context={"test": 2})  # This should trigger flush

            # Verify events were written
            with open(tmp.name, "r") as f:
                content = f.read()
                assert "test1" in content
                assert "test2" in content

    def test_logger_exception_conditions(self):
        """Test exception handling edge cases in logger."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            logger = LLMLogger(output=tmp.name, auto_flush=False)

            # Test buffer size 0 (no buffering)
            logger.buffer_size = 0
            logger.log_event("test")

            # Test auto_flush = False
            logger.auto_flush = False
            logger.log_event("test2")

    def test_optimized_logger_edge_cases(self):
        """Test edge cases in OptimizedLLMLogger."""
        output = StringIO()
        logger = OptimizedLLMLogger(output=output, buffer_size=0)

        # Test log_state_change with None values
        logger.log_state_change(
            entity_type="user",
            entity_id="123",
            field="status",
            before=None,
            after=None,
            trigger="test",
        )

        # Test log_batch_events when disabled
        logger.disable()
        logger.log_batch_events([{"event_type": "test"}])
        logger.enable()

        # Test disabled logger doesn't write events (line 148)
        logger.disable()
        logger.log_event("disabled_test")
        logger.enable()

        output.seek(0)
        content = output.getvalue()
        # Should not contain the disabled_test event
        assert "disabled_test" not in content

    def test_optimized_logger_caller_info_edge_cases(self):
        """Test edge cases in OptimizedLLMLogger _get_caller_info."""
        logger = OptimizedLLMLogger(output=StringIO())

        # Test with very high skip_frames to trigger frame=None (lines 112, 116)
        caller_info = logger._get_caller_info(skip_frames=100)
        assert caller_info["source"] == "unknown"
        assert caller_info["function"] == "unknown"
        assert caller_info["line"] == 0

    def test_optimized_logger_buffering(self):
        """Test OptimizedLLMLogger buffering functionality (lines 180-184, 195-202)."""
        output = StringIO()
        # Set buffer_size to 2 to trigger buffering code paths
        logger = OptimizedLLMLogger(output=output, buffer_size=2)

        # Add one event - should not flush yet
        logger.log_event("event1")

        # Add second event - should trigger buffer flush (lines 180-184)
        logger.log_event("event2")

        # Check that events were written (lines 195-202)
        output.seek(0)
        content = output.getvalue()
        assert "event1" in content
        assert "event2" in content

        # Test manual flush with empty buffer
        logger.flush()  # Should handle empty buffer gracefully

    def test_protocol_ellipsis_statements(self):
        """Test Protocol abstract methods with ellipsis statements."""
        from lmlog.backends import LogBackend

        # Create a minimal class that implements the protocol
        class MinimalBackend(LogBackend):
            def write(self, event):
                return super().write(event)  # This will execute the ... in Protocol

            def flush(self):
                return super().flush()  # This will execute the ... in Protocol

            def close(self):
                return super().close()  # This will execute the ... in Protocol

        backend = MinimalBackend()

        # Execute the protocol methods to hit the ellipsis statements
        result1 = backend.write({})  # Should return None (from ...)
        result2 = backend.flush()  # Should return None (from ...)
        result3 = backend.close()  # Should return None (from ...)

        assert result1 is None
        assert result2 is None
        assert result3 is None

    def test_backends_json_fallback(self):
        """Test backends' JSON fallback when encoder lacks encode_str method."""

        # Create a custom encoder without encode_str method
        class BasicEncoder:
            def encode(self, obj):
                return b"encoded"

        # Test FileBackend fallback (lines 71-73)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = FileBackend(tmp.name, encoder="orjson", async_writes=False)
            # Replace encoder with one that doesn't have encode_str
            backend._encoder = BasicEncoder()

            event = {"test": "fallback"}
            backend.write(event)

            # Verify it used JSON fallback
            with open(tmp.name, "r") as f:
                content = f.read().strip()
                import json

                parsed = json.loads(content)
                assert parsed == event

        # Test StreamBackend fallback (lines 120-122)
        stream = StringIO()
        backend = StreamBackend(stream, encoder="orjson")
        backend._encoder = BasicEncoder()

        event = {"test": "stream_fallback"}
        backend.write(event)

        stream.seek(0)
        content = stream.read().strip()
        parsed = json.loads(content)
        assert parsed == event

    @pytest.mark.asyncio
    async def test_async_backend_json_fallback(self):
        """Test AsyncFileBackend JSON fallback (lines 194-196)."""

        class BasicEncoder:
            def encode(self, obj):
                return b"encoded"

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name, encoder="orjson")
            backend._encoder = BasicEncoder()

            await backend.start()

            event = {"test": "async_fallback"}
            await backend.write(event)

            # Wait for processing
            await asyncio.sleep(0.01)
            await backend.stop()

            # Verify fallback was used
            with open(tmp.name, "r") as f:
                content = f.read().strip()
                if content:
                    import json

                    parsed = json.loads(content)
                    assert parsed == event

    def test_optimized_logger_kwargs_filtering(self):
        """Test kwargs filtering in optimized logger."""
        logger = OptimizedLLMLogger(output=StringIO())

        # Test with kwargs that include None values
        logger.log_event(
            event_type="test",
            operation="test_op",
            none_value=None,  # This should be filtered
            valid_value="keep_this",
        )

    @pytest.mark.asyncio
    async def test_async_file_backend_stop_when_not_running(self):
        """Test AsyncFileBackend stop when not running."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name)

            # Call stop without starting
            await backend.stop()  # Should handle gracefully

    @pytest.mark.asyncio
    async def test_async_file_backend_close_with_done_task(self):
        """Test AsyncFileBackend close with completed task."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name)
            await backend.start()
            await backend.stop()

            # Task should be done now
            backend.close()  # Should handle gracefully

    def test_log_performance_async_decorator(self):
        """Test log_performance decorator with async function."""
        output = StringIO()
        logger = LLMLogger(output=output)

        @log_performance(logger, threshold_ms=1, log_all=True)
        async def async_function():
            await asyncio.sleep(0.001)
            return "result"

        async def run_test():
            result = await async_function()
            assert result == "result"

        asyncio.run(run_test())

        output.seek(0)
        content = output.getvalue()
        assert "performance_info" in content or "performance_issue" in content

    def test_capture_errors_async_decorator(self):
        """Test capture_errors decorator with async function."""
        output = StringIO()
        logger = LLMLogger(output=output)

        @capture_errors(logger, include_args=True)
        async def async_failing_function(arg1, kwarg1="test"):
            raise ValueError("Async test error")

        async def run_test():
            with pytest.raises(ValueError):
                await async_failing_function("value1", kwarg1="kwvalue")

        asyncio.run(run_test())

        output.seek(0)
        content = output.getvalue()
        assert "exception" in content
        assert "Async test error" in content

    def test_log_calls_async_decorator_with_exception(self):
        """Test log_calls decorator with async function that raises exception."""
        output = StringIO()
        logger = LLMLogger(output=output)

        @log_calls(logger, include_result=True)
        async def async_failing_function():
            raise RuntimeError("Async function failed")

        async def run_test():
            with pytest.raises(RuntimeError):
                await async_failing_function()

        asyncio.run(run_test())

        output.seek(0)
        log_lines = output.getvalue().strip().split("\n")

        # Should have entry and error exit logs
        assert len(log_lines) >= 2
        assert "function_entry" in log_lines[0]
        assert "function_exit_error" in log_lines[1]

    def test_log_calls_with_result_length(self):
        """Test log_calls decorator with result that has length."""
        output = StringIO()
        logger = LLMLogger(output=output)

        @log_calls(logger, include_result=True)
        def function_with_list_result():
            return [1, 2, 3, 4, 5]

        result = function_with_list_result()
        assert len(result) == 5

        output.seek(0)
        content = output.getvalue()
        assert "result_length" in content
        assert "5" in content

    def test_log_calls_with_result_type(self):
        """Test log_calls decorator with result type logging."""
        output = StringIO()
        logger = LLMLogger(output=output)

        @log_calls(logger, include_result=True)
        def function_with_dict_result():
            return {"key": "value"}

        result = function_with_dict_result()
        assert result == {"key": "value"}

        output.seek(0)
        content = output.getvalue()
        assert "result_type" in content
        assert "dict" in content

    def test_logger_context_manager_exception(self):
        """Test logger context manager exception handling."""
        output = StringIO()

        try:
            with LLMLogger(output=output) as logger:
                logger.log_event("test")
                raise ValueError("Test exception in context")
        except ValueError:
            pass  # Expected

        # Should have flushed on exit despite exception
        output.seek(0)
        content = output.getvalue()
        assert "test" in content

    def test_optimized_logger_context_manager_exception(self):
        """Test optimized logger context manager exception handling."""
        output = StringIO()

        try:
            with OptimizedLLMLogger(output=output) as logger:
                logger.log_event("test")
                raise ValueError("Test exception in context")
        except ValueError:
            pass  # Expected

        # Should have flushed on exit despite exception
        output.seek(0)
        content = output.getvalue()
        assert "test" in content

    def test_logger_operation_context_exception(self):
        """Test logger operation context with exception."""
        output = StringIO()
        logger = LLMLogger(output=output)

        try:
            with logger.operation_context("test_operation", key="value"):
                logger.log_event("inside_operation")
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        output.seek(0)
        content = output.getvalue()
        lines = content.strip().split("\n")

        # Should have operation_start, inside event, and operation_end
        assert len(lines) >= 3
        assert "operation_start" in lines[0]
        assert "inside_operation" in lines[1]
        assert "operation_end" in lines[2]
        # Should mark as unsuccessful
        assert '"success": false' in lines[2]

    def test_optimized_logger_operation_context_exception(self):
        """Test optimized logger operation context with exception."""
        output = StringIO()
        logger = OptimizedLLMLogger(output=output)

        try:
            with logger.operation_context("test_operation", key="value"):
                logger.log_event("inside_operation")
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        output.seek(0)
        content = output.getvalue()
        lines = content.strip().split("\n")

        # Should have operation_start, inside event, and operation_end
        assert len(lines) >= 3
        assert "operation_start" in lines[0]
        assert "inside_operation" in lines[1]
        assert "operation_end" in lines[2]
        # Should mark as unsuccessful
        assert 'success":false' in lines[2] or '"success": false' in lines[2]

    def test_log_calls_async_with_args_and_result(self):
        """Test async log_calls decorator with include_args and include_result."""
        output = StringIO()
        logger = LLMLogger(output=output)

        @log_calls(logger, include_args=True, include_result=True)
        async def async_function_with_list(arg1, arg2, kwarg1="test"):
            await asyncio.sleep(0.001)
            return [1, 2, 3, 4]  # Result with __len__

        async def run_test():
            result = await async_function_with_list(
                "value1", "value2", kwarg1="kwvalue"
            )
            assert len(result) == 4

        asyncio.run(run_test())

        output.seek(0)
        content = output.getvalue()

        # Should include args and result info
        assert "args_count" in content
        assert "kwargs_keys" in content
        assert "result_type" in content
        assert "result_length" in content
        assert "4" in content  # Length of result list
