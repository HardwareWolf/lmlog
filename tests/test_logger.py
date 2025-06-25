"""
Tests for the LLMLogger class.
"""

import json
import tempfile
from pathlib import Path
from io import StringIO


from lmlog import LLMLogger


class TestLLMLogger:
    def test_basic_event_logging(self):
        """Test basic event logging functionality."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.log_event(
            event_type="test_event",
            entity_type="user",
            entity_id="123",
            operation="test_operation",
            context={"key": "value"},
        )

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "test_event"
        assert logged_data["entity"]["type"] == "user"
        assert logged_data["entity"]["id"] == "123"
        assert logged_data["operation"] == "test_operation"
        assert logged_data["context"]["key"] == "value"
        assert "timestamp" in logged_data
        assert "source" in logged_data

    def test_state_change_logging(self):
        """Test state change logging."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.log_state_change(
            entity_type="user",
            entity_id="123",
            field="status",
            before="active",
            after="suspended",
            trigger="fraud_detection",
        )

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "state_change"
        assert logged_data["entity"]["type"] == "user"
        assert logged_data["entity"]["id"] == "123"
        assert logged_data["state_change"]["field"] == "status"
        assert logged_data["state_change"]["before"] == "active"
        assert logged_data["state_change"]["after"] == "suspended"
        assert logged_data["state_change"]["trigger"] == "fraud_detection"

    def test_performance_issue_logging(self):
        """Test performance issue logging."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.log_performance_issue(
            operation="database_query",
            duration_ms=5000,
            threshold_ms=1000,
            context={"query_type": "user_lookup"},
        )

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "performance_issue"
        assert logged_data["operation"] == "database_query"
        assert logged_data["performance"]["duration_ms"] == 5000
        assert logged_data["performance"]["threshold_ms"] == 1000
        assert logged_data["performance"]["slowdown_factor"] == 5.0
        assert logged_data["context"]["query_type"] == "user_lookup"

    def test_exception_logging(self):
        """Test exception logging."""
        output = StringIO()
        logger = LLMLogger(output=output)

        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.log_exception(
                exception=e, operation="test_operation", context={"test": "data"}
            )

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "exception"
        assert logged_data["operation"] == "test_operation"
        assert logged_data["error_info"]["exception_type"] == "ValueError"
        assert logged_data["error_info"]["message"] == "Test error"
        assert "traceback" in logged_data["error_info"]
        assert logged_data["context"]["test"] == "data"

    def test_global_context(self):
        """Test global context functionality."""
        output = StringIO()
        logger = LLMLogger(output=output, global_context={"app": "test_app"})

        logger.add_global_context(version="1.0.0", env="test")

        logger.log_event(event_type="test_event")

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["app"] == "test_app"
        assert logged_data["version"] == "1.0.0"
        assert logged_data["env"] == "test"

    def test_enable_disable(self):
        """Test enable/disable functionality."""
        output = StringIO()
        logger = LLMLogger(output=output)

        # Test disable
        logger.disable()
        logger.log_event(event_type="disabled_event")

        output.seek(0)
        assert output.getvalue() == ""

        # Test enable
        logger.enable()
        logger.log_event(event_type="enabled_event")

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())
        assert logged_data["event_type"] == "enabled_event"

    def test_file_output(self):
        """Test file output functionality."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            temp_path = f.name

        try:
            logger = LLMLogger(output=temp_path)
            logger.log_event(event_type="file_test")

            with open(temp_path, "r") as f:
                logged_data = json.loads(f.read().strip())

            assert logged_data["event_type"] == "file_test"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_source_context_capture(self):
        """Test that source context is properly captured."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.log_event(event_type="source_test")

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert "source" in logged_data
        assert "function" in logged_data["source"]
        assert "line" in logged_data["source"]
        assert logged_data["source"]["function"] == "test_source_context_capture"
