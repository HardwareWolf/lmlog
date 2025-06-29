import json
from io import StringIO

import pytest

from lmlog import LLMLogger, AlwaysSampler
from lmlog.decorators import capture_errors, log_calls


@pytest.fixture
def logger_output():
    """Fixture to provide logger with StringIO output."""
    output = StringIO()
    logger = LLMLogger(
        output=output,
        async_processing=False,
        buffer_size=0,
        auto_flush=True,
        sampler=AlwaysSampler(),
    )
    return logger, output


class TestImprovedDecorators:
    def test_decorator_without_exceptions(self, logger_output):
        """Test decorators work normally without exceptions."""
        logger, output = logger_output

        @capture_errors(logger)
        def normal_function():
            return "success"

        result = normal_function()
        assert result == "success"

        assert output.getvalue() == ""

    def test_log_calls_normal_flow(self, logger_output):
        """Test log_calls decorator with normal execution."""
        logger, output = logger_output

        @log_calls(logger, log_entry=True, log_exit=True)
        def normal_function():
            return "result"

        result = normal_function()
        assert result == "result"

        output.seek(0)
        log_lines = output.getvalue().strip().split("\n")
        assert len(log_lines) == 2

        entry_log = json.loads(log_lines[0])
        exit_log = json.loads(log_lines[1])

        assert entry_log["event_type"] == "function_entry"
        assert exit_log["event_type"] == "function_exit"
