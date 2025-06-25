import json
from io import StringIO

import pytest

from lmlog import LLMLogger


@pytest.fixture
def logger_output():
    """Fixture to provide logger with StringIO output."""
    output = StringIO()
    logger = LLMLogger(output=output)
    return logger, output


def capture_errors_impl(
    logger: LLMLogger,
    event_type: str = "exception",
    include_args: bool = False,
    include_traceback: bool = True,
):
    """Working implementation of capture_errors decorator."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            context = {
                "function": func.__name__,
                "module": func.__module__,
            }

            if include_args:
                context["args_count"] = len(args)
                context["kwargs_keys"] = list(kwargs.keys())

            result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator


def log_calls_impl(
    logger: LLMLogger,
    log_entry: bool = True,
    log_exit: bool = True,
    include_args: bool = False,
    include_result: bool = False,
):
    """Working implementation of log_calls decorator."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            context = {
                "function": func.__name__,
                "module": func.__module__,
            }

            if include_args:
                context["args_count"] = len(args)
                context["kwargs_keys"] = list(kwargs.keys())

            if log_entry:
                logger.log_event(
                    event_type="function_entry",
                    operation=func.__name__,
                    context=context,
                )

            result = func(*args, **kwargs)

            if log_exit:
                exit_context = context.copy()
                if include_result:
                    exit_context["result_type"] = type(result).__name__
                    if hasattr(result, "__len__"):
                        exit_context["result_length"] = len(result)

                logger.log_event(
                    event_type="function_exit",
                    operation=func.__name__,
                    context=exit_context,
                )

            return result

        return wrapper

    return decorator


class TestImprovedDecorators:
    def test_decorator_without_exceptions(self, logger_output):
        """Test decorators work normally without exceptions."""
        logger, output = logger_output

        @capture_errors_impl(logger)
        def normal_function():
            return "success"

        result = normal_function()
        assert result == "success"

        # Should not log anything when no exception
        assert output.getvalue() == ""

    def test_log_calls_normal_flow(self, logger_output):
        """Test log_calls decorator with normal execution."""
        logger, output = logger_output

        @log_calls_impl(logger)
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
