"""
Tests for decorator functionality.
"""

import json
import time
from io import StringIO

import pytest

from lmlog import LLMLogger, capture_errors, log_performance, log_calls


class TestDecorators:
    def test_capture_errors_decorator(self):
        """Test the capture_errors decorator."""
        output = StringIO()
        logger = LLMLogger(output=output)

        @capture_errors(logger)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "exception"
        assert logged_data["operation"] == "failing_function"
        assert logged_data["error_info"]["exception_type"] == "ValueError"
        assert logged_data["error_info"]["message"] == "Test error"
        assert logged_data["context"]["function"] == "failing_function"

    def test_capture_errors_with_args(self):
        """Test capture_errors decorator with argument logging."""
        output = StringIO()
        logger = LLMLogger(output=output)

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
        logger = LLMLogger(output=output)

        @log_performance(logger, threshold_ms=100)
        def slow_function():
            time.sleep(0.15)  # 150ms
            return "result"

        result = slow_function()
        assert result == "result"

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "performance_issue"
        assert logged_data["operation"] == "slow_function"
        assert logged_data["performance"]["duration_ms"] >= 100
        assert logged_data["performance"]["threshold_ms"] == 100
        assert logged_data["context"]["function"] == "slow_function"

    def test_log_performance_under_threshold(self):
        """Test log_performance when execution is under threshold."""
        output = StringIO()
        logger = LLMLogger(output=output)

        @log_performance(logger, threshold_ms=1000, log_all=True)
        def fast_function():
            return "quick result"

        result = fast_function()
        assert result == "quick result"

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "performance_info"
        assert logged_data["operation"] == "fast_function"
        assert logged_data["performance"]["duration_ms"] < 1000

    def test_log_calls_decorator(self):
        """Test the log_calls decorator."""
        output = StringIO()
        logger = LLMLogger(output=output)

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
        logger = LLMLogger(output=output)

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
        logger = LLMLogger(output=output)

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
        assert logged_data["performance"]["duration_ms"] >= 50
