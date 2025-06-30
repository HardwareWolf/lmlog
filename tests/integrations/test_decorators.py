"""
Tests for decorator functionality.
"""

import json
import time
from io import StringIO

import pytest

from lmlog import LLMLogger, capture_errors, log_performance, log_calls, AlwaysSampler


class TestDecorators:
    def test_capture_errors_decorator(self):
        """Test the capture_errors decorator."""
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @capture_errors(logger)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "exception"
        assert logged_data["context"]["operation"] == "failing_function"
        assert logged_data["context"]["exception_type"] == "ValueError"
        assert logged_data["context"]["exception_message"] == "Test error"
        assert logged_data["context"]["function"] == "failing_function"

    def test_capture_errors_with_args(self):
        """Test capture_errors decorator with argument logging."""
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @capture_errors(logger, include_args=True)
        def failing_function_with_args(arg1, arg2, kwarg1="test"):
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function_with_args("value1", "value2", kwarg1="kwvalue")

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["context"]["args_count"] == 2
        assert "kwarg1" in logged_data["context"]["kwargs_keys"]

    def test_log_performance_decorator(self):
        """Test the log_performance decorator."""
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @log_performance(logger, threshold_ms=100)
        def slow_function():
            time.sleep(0.15)  # 150ms
            return "result"

        result = slow_function()
        assert result == "result"

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "performance_issue"
        assert logged_data["context"]["operation"] == "slow_function"
        assert logged_data["context"]["duration_ms"] >= 100
        assert logged_data["context"]["threshold_ms"] == 100
        assert logged_data["context"]["function"] == "slow_function"

    def test_log_performance_under_threshold(self):
        """Test log_performance when execution is under threshold."""
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @log_performance(logger, threshold_ms=1000, log_all=True)
        def fast_function():
            return "quick result"

        result = fast_function()
        assert result == "quick result"

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "performance_info"
        assert logged_data["context"]["operation"] == "fast_function"
        assert logged_data["context"]["duration_ms"] < 1000

    def test_log_calls_decorator(self):
        """Test the log_calls decorator."""
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @log_calls(logger, include_args=True, include_result=True)
        def test_function(arg1, kwarg1="default"):
            return "test_result"

        result = test_function("value1", kwarg1="kwvalue")
        assert result == "test_result"

        output.seek(0)
        log_lines = output.getvalue().strip().split("\n")

        # Should have entry and exit logs
        assert len(log_lines) == 2

        entry_log = json.loads(log_lines[0])
        exit_log = json.loads(log_lines[1])

        # Check entry log
        assert entry_log["event_type"] == "function_entry"
        assert entry_log["operation"] == "test_function"
        assert entry_log["context"]["args_count"] == 1
        assert "kwarg1" in entry_log["context"]["kwargs_keys"]

        # Check exit log
        assert exit_log["event_type"] == "function_exit"
        assert exit_log["operation"] == "test_function"
        assert exit_log["context"]["result_type"] == "str"

    def test_log_calls_with_exception(self):
        """Test log_calls decorator when function raises exception."""
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @log_calls(logger)
        def failing_function():
            raise RuntimeError("Function failed")

        with pytest.raises(RuntimeError):
            failing_function()

        output.seek(0)
        log_lines = output.getvalue().strip().split("\n")

        # Should have entry and error exit logs
        assert len(log_lines) == 2

        entry_log = json.loads(log_lines[0])
        exit_log = json.loads(log_lines[1])

        # Check entry log
        assert entry_log["event_type"] == "function_entry"

        # Check error exit log
        assert exit_log["event_type"] == "function_exit_error"
        assert exit_log["error_info"]["exception_type"] == "RuntimeError"
        assert exit_log["error_info"]["message"] == "Function failed"

    def test_multiple_decorators(self):
        """Test using multiple decorators together."""
        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @capture_errors(logger)
        @log_performance(logger, threshold_ms=50)
        def decorated_function():
            time.sleep(0.1)  # 100ms
            return "success"

        result = decorated_function()
        assert result == "success"

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        # Should log performance issue
        assert logged_data["event_type"] == "performance_issue"
        assert logged_data["context"]["duration_ms"] >= 50


class TestAsyncDecorators:
    """Test async versions of decorators."""

    @pytest.mark.asyncio
    async def test_capture_errors_async(self):
        """Test async capture_errors decorator."""
        import asyncio
        from io import StringIO

        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @capture_errors(logger, include_args=True, include_traceback=True)
        async def async_failing_function(x, y=10):
            await asyncio.sleep(0.01)
            raise RuntimeError("Async error")

        with pytest.raises(RuntimeError):
            await async_failing_function(5, y=20)

        output.seek(0)
        event = json.loads(output.getvalue())

        assert event["event_type"] == "exception"
        assert event["context"]["function"] == "async_failing_function"
        assert event["context"]["args_count"] == 1
        assert event["context"]["kwargs_keys"] == ["y"]
        assert "traceback" in event["context"]

    @pytest.mark.asyncio
    async def test_log_performance_async(self):
        """Test async log_performance decorator."""
        import asyncio
        from io import StringIO

        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @log_performance(logger, threshold_ms=50, log_all=True)
        async def async_slow_function(delay=0.1):
            await asyncio.sleep(delay)
            return "async result"

        # Test slow execution
        result = await async_slow_function(0.06)  # 60ms
        assert result == "async result"

        output.seek(0)
        lines = output.getvalue().strip().split("\n")
        event = json.loads(lines[0])

        assert event["event_type"] == "performance_issue"
        assert event["context"]["operation"] == "async_slow_function"
        assert event["context"]["duration_ms"] >= 50

        # Test fast execution with log_all
        output.truncate(0)
        output.seek(0)

        result = await async_slow_function(0.01)  # 10ms

        output.seek(0)
        event = json.loads(output.getvalue())
        assert event["event_type"] == "performance_info"
        assert event["context"]["duration_ms"] < 50

    @pytest.mark.asyncio
    async def test_log_calls_async(self):
        """Test async log_calls decorator."""
        import asyncio
        from io import StringIO

        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @log_calls(logger, include_args=True, include_result=True)
        async def async_test_function(x, y=5):
            await asyncio.sleep(0.01)
            return [x, y]

        # Test successful execution
        result = await async_test_function(10, y=20)
        assert result == [10, 20]

        output.seek(0)
        lines = output.getvalue().strip().split("\n")

        entry_event = json.loads(lines[0])
        exit_event = json.loads(lines[1])

        # Check entry
        assert entry_event["event_type"] == "function_entry"
        assert entry_event["context"]["args_count"] == 1
        assert entry_event["context"]["kwargs_keys"] == ["y"]

        # Check exit
        assert exit_event["event_type"] == "function_exit"
        assert exit_event["context"]["result_type"] == "list"
        assert exit_event["context"]["result_length"] == 2

    @pytest.mark.asyncio
    async def test_log_calls_async_with_exception(self):
        """Test async log_calls with exception."""
        import asyncio
        from io import StringIO

        output = StringIO()
        logger = LLMLogger(
            output=output, async_processing=False, sampler=AlwaysSampler()
        )

        @log_calls(logger, log_exit=True)
        async def async_failing_function():
            await asyncio.sleep(0.01)
            raise KeyError("Missing key")

        with pytest.raises(KeyError):
            await async_failing_function()

        output.seek(0)
        lines = output.getvalue().strip().split("\n")

        exit_event = json.loads(lines[-1])
        assert exit_event["event_type"] == "function_exit_error"
        assert exit_event["error_info"]["exception_type"] == "KeyError"
        assert "Missing key" in exit_event["error_info"]["message"]
