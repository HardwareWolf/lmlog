import time
import functools
import asyncio
from typing import Callable, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.base_logger import LLMLogger


def capture_errors(
    logger: "LLMLogger",
    event_type: str = "exception",
    include_args: bool = False,
    include_traceback: bool = True,
):
    """
    Decorator to automatically log exceptions.

    Args:
        logger: "LLMLogger" instance
        event_type: Event type to log
        include_args: Whether to include function arguments in context
        include_traceback: Whether to include stack trace
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                context: Dict[str, Any] = {
                    "function": func.__name__,
                    "module": func.__module__,
                }

                if include_args:
                    context["args_count"] = len(args)
                    context["kwargs_keys"] = list(kwargs.keys())

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as exc:
                    logger.log_exception(
                        exception=exc,
                        operation=func.__name__,
                        context=context,
                        include_traceback=include_traceback,
                    )
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                context: Dict[str, Any] = {
                    "function": func.__name__,
                    "module": func.__module__,
                }

                if include_args:
                    context["args_count"] = len(args)
                    context["kwargs_keys"] = list(kwargs.keys())

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as exc:
                    logger.log_exception(
                        exception=exc,
                        operation=func.__name__,
                        context=context,
                        include_traceback=include_traceback,
                    )
                    raise

            return wrapper

    return decorator


def log_performance(
    logger: "LLMLogger", threshold_ms: int = 1000, log_all: bool = False
):
    """
    Decorator to log performance issues.

    Args:
        logger: "LLMLogger" instance
        threshold_ms: Log if execution takes longer than this
        log_all: Log all executions, not just slow ones
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                if log_all or duration_ms > threshold_ms:
                    context: Dict[str, Any] = {
                        "function": func.__name__,
                        "module": func.__module__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                    }

                    if duration_ms > threshold_ms:
                        logger.log_performance_issue(
                            operation=func.__name__,
                            duration_ms=duration_ms,
                            threshold_ms=threshold_ms,
                            context=context,
                        )
                    else:
                        perf_info_context = {
                            "operation": func.__name__,
                            "duration_ms": duration_ms,
                            **context,
                        }
                        logger.log_event(
                            event_type="performance_info",
                            context=perf_info_context,
                        )

                return result

            return async_wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                if log_all or duration_ms > threshold_ms:
                    context: Dict[str, Any] = {
                        "function": func.__name__,
                        "module": func.__module__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                    }

                    if duration_ms > threshold_ms:
                        logger.log_performance_issue(
                            operation=func.__name__,
                            duration_ms=duration_ms,
                            threshold_ms=threshold_ms,
                            context=context,
                        )
                    else:
                        perf_info_context = {
                            "operation": func.__name__,
                            "duration_ms": duration_ms,
                            **context,
                        }
                        logger.log_event(
                            event_type="performance_info",
                            context=perf_info_context,
                        )

                return result

            return wrapper

    return decorator


def log_calls(
    logger: "LLMLogger",
    log_entry: bool = True,
    log_exit: bool = True,
    include_args: bool = False,
    include_result: bool = False,
):
    """
    Decorator to log function entry and exit.

    Args:
        logger: "LLMLogger" instance
        log_entry: Log when function is called
        log_exit: Log when function returns
        include_args: Include function arguments (be careful with sensitive data)
        include_result: Include return value (be careful with sensitive data)
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                context: Dict[str, Any] = {
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

                try:
                    result = await func(*args, **kwargs)

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
                except Exception as exc:
                    if log_exit:
                        logger.log_event(
                            event_type="function_exit_error",
                            operation=func.__name__,
                            error_info={
                                "exception_type": type(exc).__name__,
                                "message": str(exc),
                            },
                            context=context,
                        )
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                context: Dict[str, Any] = {
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

                try:
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
                except Exception as exc:
                    if log_exit:
                        logger.log_event(
                            event_type="function_exit_error",
                            operation=func.__name__,
                            error_info={
                                "exception_type": type(exc).__name__,
                                "message": str(exc),
                            },
                            context=context,
                        )
                    raise

            return wrapper

    return decorator
