import json
import inspect
import asyncio
from datetime import datetime, UTC
from typing import Any, Dict, Optional, Union, TextIO, List
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from threading import Lock


class LLMLogger:
    """
    LLM-optimized logger for structured debugging information.

    Designed to capture debugging context in a format that enables
    LLMs to effectively assist with troubleshooting across any Python application.
    """

    def __init__(
        self,
        output: Union[str, Path, TextIO] = "llm_log.jsonl",
        enabled: bool = True,
        global_context: Optional[Dict[str, Any]] = None,
        buffer_size: int = 0,
        auto_flush: bool = True,
    ):
        """
        Initialize the LLM logger.

        Args:
            output: Output destination - file path or file-like object
            enabled: Whether logging is enabled
            global_context: Context to include in all log events
            buffer_size: Number of events to buffer before writing (0 = no buffering)
            auto_flush: Whether to flush after each write
        """
        self.output = output
        self.enabled = enabled
        self.global_context = global_context or {}
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        self._buffer: List[Dict[str, Any]] = []
        self._lock = Lock()
        self._file_handle = None

    def _get_caller_info(self, skip_frames: int = 2) -> Dict[str, Any]:
        """Extract caller information from the stack."""
        frame = inspect.currentframe()
        for _ in range(skip_frames):
            if frame is None:
                break
            frame = frame.f_back

        if frame is None:
            return {"source": "unknown", "function": "unknown", "line": 0}

        return {
            "source": f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}",
            "function": frame.f_code.co_name,
            "line": frame.f_lineno,
            "file": str(Path(frame.f_code.co_filename)),
        }

    def _write_event(self, event: Dict[str, Any]) -> None:
        """Write event to output destination."""
        if not self.enabled:
            return

        with self._lock:
            if self.buffer_size > 0:
                self._buffer.append(event)
                if len(self._buffer) >= self.buffer_size:
                    self._flush_buffer()
            else:
                self._write_single_event(event)

    def _write_single_event(self, event: Dict[str, Any]) -> None:
        """Write a single event to output."""
        if isinstance(self.output, (str, Path)):
            with open(self.output, "a", encoding="utf-8") as f:
                json.dump(event, f, ensure_ascii=False, default=str)
                f.write("\n")
                if self.auto_flush:
                    f.flush()
        else:
            json.dump(event, self.output, ensure_ascii=False, default=str)
            self.output.write("\n")
            if self.auto_flush and hasattr(self.output, "flush"):
                self.output.flush()

    def _flush_buffer(self) -> None:
        """Flush buffered events to output."""
        if not self._buffer:
            return

        if isinstance(self.output, (str, Path)):
            with open(self.output, "a", encoding="utf-8") as f:
                for event in self._buffer:
                    json.dump(event, f, ensure_ascii=False, default=str)
                    f.write("\n")
                if self.auto_flush:
                    f.flush()
        else:
            for event in self._buffer:
                json.dump(event, self.output, ensure_ascii=False, default=str)
                self.output.write("\n")
            if self.auto_flush and hasattr(self.output, "flush"):
                self.output.flush()

        self._buffer.clear()

    def log_event(
        self,
        event_type: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Log a structured event.

        Args:
            event_type: Type of event (data_anomaly, performance_issue, etc.)
            entity_type: Type of entity involved (user, order, etc.)
            entity_id: ID of the specific entity
            operation: Operation being performed
            context: Additional context information
            **kwargs: Additional event-specific data
        """
        if not self.enabled:
            return

        caller_info = self._get_caller_info()

        event = {
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "source": caller_info,
            **self.global_context,
        }

        if entity_type or entity_id:
            entity: Dict[str, Any] = {}
            if entity_type:
                entity["type"] = entity_type
            if entity_id:
                entity["id"] = entity_id
            event["entity"] = entity

        if operation:
            event["operation"] = operation

        if context:
            event["context"] = context

        for key, value in kwargs.items():
            if key not in event:
                event[key] = value

        self._write_event(event)

    def log_state_change(
        self,
        entity_type: str,
        entity_id: str,
        field: str,
        before: Any,
        after: Any,
        trigger: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a state change event.

        Args:
            entity_type: Type of entity (user, order, etc.)
            entity_id: ID of the entity
            field: Field that changed
            before: Previous value
            after: New value
            trigger: What caused the change
            context: Additional context
        """
        self.log_event(
            event_type="state_change",
            entity_type=entity_type,
            entity_id=entity_id,
            state_change={
                "field": field,
                "before": before,
                "after": after,
                "trigger": trigger,
            },
            context=context,
        )

    def log_performance_issue(
        self,
        operation: str,
        duration_ms: int,
        threshold_ms: int = 1000,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Log a performance issue.

        Args:
            operation: Operation that was slow
            duration_ms: Actual duration in milliseconds
            threshold_ms: Expected threshold
            context: Additional context
            **kwargs: Additional performance metrics
        """
        self.log_event(
            event_type="performance_issue",
            operation=operation,
            performance={
                "duration_ms": duration_ms,
                "threshold_ms": threshold_ms,
                "slowdown_factor": duration_ms / threshold_ms,
                **kwargs,
            },
            context=context,
        )

    def enable(self) -> None:
        """Enable logging."""
        self.enabled = True

    def disable(self) -> None:
        """Disable logging."""
        self.enabled = False

    def log_exception(
        self,
        exception: Exception,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        include_traceback: bool = True,
    ) -> None:
        """
        Log an exception with full context.

        Args:
            exception: The exception that occurred
            operation: Operation that failed
            context: Additional context
            include_traceback: Whether to include stack trace
        """
        import traceback

        error_info: Dict[str, Any] = {
            "exception_type": type(exception).__name__,
            "message": str(exception),
        }

        if include_traceback:
            error_info["traceback"] = traceback.format_exc().splitlines()

        self.log_event(
            event_type="exception",
            operation=operation,
            error_info=error_info,
            context=context,
        )

    def flush(self) -> None:
        """Flush any buffered events to output."""
        with self._lock:
            self._flush_buffer()

    def clear_buffer(self) -> None:
        """Clear the buffer without writing."""
        with self._lock:
            self._buffer.clear()

    def log_business_rule_violation(
        self,
        rule: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a business rule violation.

        Args:
            rule: Name of the business rule violated
            entity_type: Type of entity involved
            entity_id: ID of the entity
            expected: Expected value
            actual: Actual value
            context: Additional context
        """
        violation_info = {"rule": rule}
        if expected is not None:
            violation_info["expected"] = expected
        if actual is not None:
            violation_info["actual"] = actual

        self.log_event(
            event_type="business_rule_violation",
            entity_type=entity_type,
            entity_id=entity_id,
            violation=violation_info,
            context=context,
        )

    def log_integration_failure(
        self,
        service: str,
        operation: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an integration failure with external service.

        Args:
            service: Name of the external service
            operation: Operation that failed
            error_code: Error code if available
            error_message: Error message if available
            context: Additional context
        """
        integration_info = {"service": service, "operation": operation}
        if error_code:
            integration_info["error_code"] = error_code
        if error_message:
            integration_info["error_message"] = error_message

        self.log_event(
            event_type="integration_failure",
            integration=integration_info,
            context=context,
        )

    def log_authentication_issue(
        self,
        auth_type: str,
        user_id: Optional[str] = None,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log authentication or authorization issues.

        Args:
            auth_type: Type of authentication (login, token, permission)
            user_id: User ID if available
            reason: Reason for failure
            context: Additional context
        """
        auth_info = {"type": auth_type}
        if reason:
            auth_info["reason"] = reason

        self.log_event(
            event_type="authentication_issue",
            entity_type="user" if user_id else None,
            entity_id=user_id,
            authentication=auth_info,
            context=context,
        )

    def log_user_behavior_anomaly(
        self,
        user_id: str,
        behavior_type: str,
        anomaly_score: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log unusual user behavior that might indicate security issues.

        Args:
            user_id: User ID
            behavior_type: Type of behavior (login_pattern, usage_spike, etc.)
            anomaly_score: Numerical score indicating severity
            context: Additional context
        """
        behavior_info: Dict[str, Any] = {"type": behavior_type}
        if anomaly_score is not None:
            behavior_info["anomaly_score"] = anomaly_score

        self.log_event(
            event_type="user_behavior_anomaly",
            entity_type="user",
            entity_id=user_id,
            behavior=behavior_info,
            context=context,
        )

    @contextmanager
    def operation_context(self, operation: str, **context):
        """Context manager for tracking operation execution."""
        start_time = datetime.now(UTC)
        operation_id = f"{operation}_{int(start_time.timestamp() * 1000000)}"

        self.log_event(
            event_type="operation_start",
            operation=operation,
            operation_id=operation_id,
            context=context,
        )

        operation_context = {"operation_id": operation_id}
        old_context = self.global_context.copy()
        self.global_context.update(operation_context)

        success = False

        try:
            yield operation_id
            success = True
        finally:
            end_time = datetime.now(UTC)
            self.global_context = old_context

            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            self.log_event(
                event_type="operation_end",
                operation=operation,
                operation_id=operation_id,
                success=success,
                duration_ms=duration_ms,
                context=context,
            )

    async def alog_event(
        self,
        event_type: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Async version of log_event."""

        import functools

        sync_callable = functools.partial(
            self.log_event,
            event_type,
            entity_type,
            entity_id,
            operation,
            context,
            **kwargs,
        )
        await asyncio.get_event_loop().run_in_executor(None, sync_callable)

    @asynccontextmanager
    async def aoperation_context(self, operation: str, **context):
        """Async context manager for tracking operation execution."""
        start_time = datetime.now(UTC)
        operation_id = f"{operation}_{int(start_time.timestamp() * 1000000)}"

        await self.alog_event(
            event_type="operation_start",
            operation=operation,
            operation_id=operation_id,
            context=context,
        )

        operation_context = {"operation_id": operation_id}
        old_context = self.global_context.copy()
        self.global_context.update(operation_context)

        success = False
        end_time = start_time

        yield operation_id

        success = True
        end_time = datetime.now(UTC)

        self.global_context = old_context

        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        await self.alog_event(
            event_type="operation_end",
            operation=operation,
            operation_id=operation_id,
            success=success,
            duration_ms=duration_ms,
            context=context,
        )

    def set_output(self, output: Union[str, Path, TextIO]) -> None:
        """Change the output destination."""
        self.flush()
        self.output = output

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    def add_global_context(self, **context) -> None:
        """Add context that will be included in all future log events."""
        self.global_context.update(context)

    def remove_global_context(self, *keys: str) -> None:
        """Remove keys from global context."""
        for key in keys:
            self.global_context.pop(key, None)

    def clear_global_context(self) -> None:
        """Clear all global context."""
        self.global_context.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.flush()
        return False
