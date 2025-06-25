import json
import asyncio
import tempfile
from pathlib import Path
from io import StringIO

import pytest

from lmlog import LLMLogger, capture_errors, log_performance, log_calls


class TestComprehensiveCoverage:
    def test_logger_init_with_all_params(self):
        """Test logger initialization with all parameters."""
        output = StringIO()
        logger = LLMLogger(
            output=output,
            enabled=False,
            global_context={"app": "test"},
            buffer_size=5,
            auto_flush=False,
        )

        assert logger.enabled is False
        assert logger.global_context == {"app": "test"}
        assert logger.buffer_size == 5
        assert logger.auto_flush is False
        assert logger.get_buffer_size() == 0

    def test_buffering_functionality(self):
        """Test event buffering."""
        output = StringIO()
        logger = LLMLogger(output=output, buffer_size=3, auto_flush=False)

        logger.log_event("test1")
        assert logger.get_buffer_size() == 1
        assert output.getvalue() == ""

        logger.log_event("test2")
        assert logger.get_buffer_size() == 2
        assert output.getvalue() == ""

        logger.log_event("test3")
        assert logger.get_buffer_size() == 0
        assert len(output.getvalue().strip().split("\n")) == 3

        logger.log_event("test4")
        assert logger.get_buffer_size() == 1

    def test_flush_and_clear_buffer(self):
        """Test buffer flush and clear operations."""
        output = StringIO()
        logger = LLMLogger(output=output, buffer_size=5)

        logger.log_event("test1")
        logger.log_event("test2")
        assert logger.get_buffer_size() == 2

        logger.flush()
        assert logger.get_buffer_size() == 0
        assert len(output.getvalue().strip().split("\n")) == 2

        logger.log_event("test3")
        assert logger.get_buffer_size() == 1

        logger.clear_buffer()
        assert logger.get_buffer_size() == 0
        assert len(output.getvalue().strip().split("\n")) == 2

    def test_business_rule_violation_logging(self):
        """Test business rule violation logging."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.log_business_rule_violation(
            rule="payment_limit",
            entity_type="user",
            entity_id="123",
            expected=1000,
            actual=1500,
            context={"transaction_id": "tx_456"},
        )

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "business_rule_violation"
        assert logged_data["entity"]["type"] == "user"
        assert logged_data["entity"]["id"] == "123"
        assert logged_data["violation"]["rule"] == "payment_limit"
        assert logged_data["violation"]["expected"] == 1000
        assert logged_data["violation"]["actual"] == 1500
        assert logged_data["context"]["transaction_id"] == "tx_456"

    def test_integration_failure_logging(self):
        """Test integration failure logging."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.log_integration_failure(
            service="payment_gateway",
            operation="process_payment",
            error_code="503",
            error_message="Service unavailable",
            context={"retry_count": 3},
        )

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "integration_failure"
        assert logged_data["integration"]["service"] == "payment_gateway"
        assert logged_data["integration"]["operation"] == "process_payment"
        assert logged_data["integration"]["error_code"] == "503"
        assert logged_data["integration"]["error_message"] == "Service unavailable"
        assert logged_data["context"]["retry_count"] == 3

    def test_authentication_issue_logging(self):
        """Test authentication issue logging."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.log_authentication_issue(
            auth_type="token",
            user_id="user_123",
            reason="expired",
            context={"token_age": 86400},
        )

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "authentication_issue"
        assert logged_data["entity"]["type"] == "user"
        assert logged_data["entity"]["id"] == "user_123"
        assert logged_data["authentication"]["type"] == "token"
        assert logged_data["authentication"]["reason"] == "expired"
        assert logged_data["context"]["token_age"] == 86400

    def test_user_behavior_anomaly_logging(self):
        """Test user behavior anomaly logging."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.log_user_behavior_anomaly(
            user_id="user_456",
            behavior_type="login_pattern",
            anomaly_score=0.95,
            context={"locations": ["NY", "Tokyo"], "time_diff_hours": 14},
        )

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["event_type"] == "user_behavior_anomaly"
        assert logged_data["entity"]["type"] == "user"
        assert logged_data["entity"]["id"] == "user_456"
        assert logged_data["behavior"]["type"] == "login_pattern"
        assert logged_data["behavior"]["anomaly_score"] == 0.95
        assert logged_data["context"]["time_diff_hours"] == 14

    def test_operation_context_manager(self):
        """Test operation context manager."""
        output = StringIO()
        logger = LLMLogger(output=output)

        with logger.operation_context(
            "user_registration", user_type="premium"
        ) as op_id:
            assert op_id.startswith("user_registration_")
            logger.log_event("step_completed", step="validation")

        output.seek(0)
        log_lines = output.getvalue().strip().split("\n")
        assert len(log_lines) == 3

        start_log = json.loads(log_lines[0])
        step_log = json.loads(log_lines[1])
        end_log = json.loads(log_lines[2])

        assert start_log["event_type"] == "operation_start"
        assert start_log["operation"] == "user_registration"
        assert start_log["operation_id"] == step_log["operation_id"]

        assert end_log["event_type"] == "operation_end"
        assert end_log["success"] is True
        assert "duration_ms" in end_log

    @pytest.mark.asyncio
    async def test_async_operation_context_manager(self):
        """Test async operation context manager."""
        output = StringIO()
        logger = LLMLogger(output=output)

        async with logger.aoperation_context(
            "async_processing", batch_size=100
        ) as op_id:
            assert op_id.startswith("async_processing_")
            await logger.alog_event("async_step", step="data_fetch")

        output.seek(0)
        log_lines = output.getvalue().strip().split("\n")
        assert len(log_lines) == 3

        start_log = json.loads(log_lines[0])
        end_log = json.loads(log_lines[2])

        assert start_log["event_type"] == "operation_start"
        assert end_log["event_type"] == "operation_end"
        assert end_log["success"] is True

    def test_set_output_destination(self):
        """Test changing output destination."""
        output1 = StringIO()
        output2 = StringIO()
        logger = LLMLogger(output=output1, buffer_size=0)

        logger.log_event("test1")
        logger.set_output(output2)
        logger.log_event("test2")

        assert output1.getvalue().strip() != ""
        assert output2.getvalue().strip() != ""

    def test_global_context_management(self):
        """Test global context add, remove, clear."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.add_global_context(app="test", version="1.0")
        logger.add_global_context(env="prod")

        logger.log_event("test1")

        logger.remove_global_context("version")
        logger.log_event("test2")

        logger.clear_global_context()
        logger.log_event("test3")

        output.seek(0)
        log_lines = output.getvalue().strip().split("\n")

        log1 = json.loads(log_lines[0])
        log2 = json.loads(log_lines[1])
        log3 = json.loads(log_lines[2])

        assert log1["app"] == "test"
        assert log1["version"] == "1.0"
        assert log1["env"] == "prod"

        assert log2["app"] == "test"
        assert "version" not in log2
        assert log2["env"] == "prod"

        assert "app" not in log3
        assert "env" not in log3

    def test_context_manager_usage(self):
        """Test logger as context manager."""
        output = StringIO()

        with LLMLogger(output=output, buffer_size=5) as logger:
            logger.log_event("test1")
            logger.log_event("test2")

        assert len(output.getvalue().strip().split("\n")) == 2

    def test_disabled_logger_no_output(self):
        """Test that disabled logger produces no output."""
        output = StringIO()
        logger = LLMLogger(output=output, enabled=False)

        logger.log_event("test")
        logger.log_state_change("user", "123", "status", "old", "new", "trigger")
        logger.log_performance_issue("op", 1000, 500)
        logger.log_business_rule_violation("rule")
        logger.log_integration_failure("service", "op")
        logger.log_authentication_issue("token")
        logger.log_user_behavior_anomaly("user", "pattern")

        assert output.getvalue() == ""

    def test_partial_parameters(self):
        """Test methods with minimal required parameters."""
        output = StringIO()
        logger = LLMLogger(output=output)

        logger.log_business_rule_violation("rule")
        logger.log_integration_failure("service", "operation")
        logger.log_authentication_issue("token")

        output.seek(0)
        log_lines = output.getvalue().strip().split("\n")
        assert len(log_lines) == 3

        for line in log_lines:
            data = json.loads(line)
            assert "timestamp" in data
            assert "source" in data

    @pytest.mark.asyncio
    async def test_async_decorators(self):
        """Test decorators work with async functions."""
        output = StringIO()
        logger = LLMLogger(output=output)

        @capture_errors(logger)
        @log_performance(logger, threshold_ms=50)
        @log_calls(logger)
        async def async_function():
            await asyncio.sleep(0.1)
            return "async_result"

        result = await async_function()
        assert result == "async_result"

        output.seek(0)
        log_lines = output.getvalue().strip().split("\n")
        assert len(log_lines) >= 2

        entry_log = json.loads(log_lines[0])
        assert entry_log["event_type"] == "function_entry"

    def test_file_output_with_path_object(self):
        """Test file output using Path object."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            temp_path = Path(f.name)

        logger = LLMLogger(output=temp_path)
        logger.log_event("file_test")

        with open(temp_path, "r") as f:
            logged_data = json.loads(f.read().strip())

        assert logged_data["event_type"] == "file_test"

        temp_path.unlink()

    def test_exception_without_traceback(self):
        """Test exception logging without traceback."""
        output = StringIO()
        logger = LLMLogger(output=output)

        exc = ValueError("test error")
        logger.log_exception(exc, include_traceback=False)

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert "traceback" not in logged_data["error_info"]
        assert logged_data["error_info"]["exception_type"] == "ValueError"

    def test_edge_cases_caller_info(self):
        """Test caller info extraction edge cases."""
        output = StringIO()
        logger = LLMLogger(output=output)

        def deep_call():
            def deeper_call():
                logger.log_event("deep_test")

            deeper_call()

        deep_call()

        output.seek(0)
        logged_data = json.loads(output.getvalue().strip())

        assert logged_data["source"]["function"] == "deeper_call"
        assert "test_comprehensive.py" in logged_data["source"]["file"]
