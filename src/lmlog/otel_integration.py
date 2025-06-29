"""
OpenTelemetry integration for trace context correlation.
"""

from typing import Dict, Any, Optional
import threading
from contextlib import contextmanager

try:
    from importlib.metadata import version

    SDK_VERSION = version("lmlog")
except ImportError:
    SDK_VERSION = "unknown"


class ContextLocal(threading.local):
    """Thread-local storage with context attribute."""

    def __init__(self):
        super().__init__()
        self.context: Dict[str, Any] = {}


try:
    from opentelemetry import trace, baggage
    from opentelemetry.trace import Tracer
    from opentelemetry.metrics import Meter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    Tracer = None
    Meter = None


class TraceContextExtractor:
    """
    Extracts OpenTelemetry trace context for log correlation.
    """

    __slots__ = ("_tracer",)

    def __init__(self, tracer_name: str = "lmlog"):
        """
        Initialize trace context extractor.

        Args:
            tracer_name: Name for the tracer
        """
        if OTEL_AVAILABLE:
            self._tracer: Optional[Tracer] = trace.get_tracer(tracer_name)
        else:
            self._tracer: Optional[Tracer] = None

    def extract_context(self) -> Dict[str, Any]:
        """
        Extract current trace context.

        Returns:
            Dictionary with trace context information
        """
        if not OTEL_AVAILABLE or not self._tracer:
            return {}

        span = trace.get_current_span()
        context_data = {}

        if span and span.is_recording():
            span_context = span.get_span_context()

            context_data.update(
                {
                    "trace_id": format(span_context.trace_id, "032x"),
                    "span_id": format(span_context.span_id, "016x"),
                    "trace_flags": span_context.trace_flags,
                    "trace_state": (
                        dict(span_context.trace_state)
                        if span_context.trace_state
                        else {}
                    ),
                    "span_name": getattr(span, "name", ""),
                    "span_kind": (
                        getattr(kind, "name", str(kind))
                        if (kind := getattr(span, "kind", None)) is not None
                        else ""
                    ),
                }
            )

        current_baggage = baggage.get_all()
        if current_baggage:
            context_data["baggage"] = current_baggage

        return context_data

    def is_available(self) -> bool:
        """Check if OpenTelemetry is available."""
        return OTEL_AVAILABLE and self._tracer is not None

    @contextmanager
    def start_span(self, name: str, **kwargs):
        """
        Start a new span for logging operations.

        Args:
            name: Span name
            **kwargs: Additional span attributes
        """
        if not self._tracer:
            yield None
            return

        with self._tracer.start_as_current_span(name) as span:
            if kwargs:
                span.set_attributes(kwargs)
            yield span


class CorrelationContext:
    """
    Manages correlation context for distributed tracing.
    """

    __slots__ = ("_local", "_extractors")

    def __init__(self):
        """Initialize correlation context."""
        self._local = ContextLocal()
        self._extractors = [TraceContextExtractor()]

    def set_context(self, **context: Any) -> None:
        """
        Set correlation context.

        Args:
            **context: Context key-value pairs
        """
        self._local.context.update(context)

    def get_context(self) -> Dict[str, Any]:
        """
        Get current correlation context.

        Returns:
            Current context dictionary
        """
        context_data = {}
        context_data.update(self._local.context)

        for extractor in self._extractors:
            context_data.update(extractor.extract_context())

        return context_data

    def clear_context(self) -> None:
        """Clear current correlation context."""
        self._local.context.clear()

    def add_extractor(self, extractor: TraceContextExtractor) -> None:
        """
        Add a custom context extractor.

        Args:
            extractor: Context extractor
        """
        self._extractors.append(extractor)

    @contextmanager
    def with_context(self, **context: Any):
        """
        Context manager for temporary context.

        Args:
            **context: Temporary context key-value pairs
        """
        old_context = self.get_context()
        self.set_context(**context)
        try:
            yield
        finally:
            self.clear_context()
            if old_context:
                self.set_context(**old_context)


class MetricGenerator:
    """
    Generates OpenTelemetry metrics from log events.
    """

    __slots__ = ("_meter", "_counters", "_histograms", "_enabled")

    def __init__(self, meter_name: str = "lmlog_metrics"):
        """
        Initialize metric generator.

        Args:
            meter_name: Name for the meter
        """
        self._enabled = OTEL_AVAILABLE
        self._counters = {}
        self._histograms = {}

        if self._enabled:
            try:
                from opentelemetry import metrics

                self._meter: Optional[Meter] = metrics.get_meter(meter_name)
            except ImportError:
                self._enabled = False
                self._meter: Optional[Meter] = None
        else:
            self._meter: Optional[Meter] = None

    def increment_counter(
        self, name: str, value: int = 1, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Counter name
            value: Value to increment by
            attributes: Metric attributes
        """
        if not self._enabled or not self._meter:
            return

        if name not in self._counters:
            self._counters[name] = self._meter.create_counter(
                name=name, description=f"Count of {name} events"
            )

        self._counters[name].add(value, attributes or {})

    def record_histogram(
        self, name: str, value: float, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a histogram metric.

        Args:
            name: Histogram name
            value: Value to record
            attributes: Metric attributes
        """
        if not self._enabled or not self._meter:
            return

        if name not in self._histograms:
            self._histograms[name] = self._meter.create_histogram(
                name=name, description=f"Distribution of {name} values"
            )

        self._histograms[name].record(value, attributes or {})

    def generate_from_event(self, event: Dict[str, Any]) -> None:
        """
        Generate metrics from a log event.

        Args:
            event: Log event
        """
        if not self._enabled:
            return

        event_type = event.get("event_type", "unknown")
        level = event.get("level", "info")

        self.increment_counter(
            "lmlog_events_total", attributes={"event_type": event_type, "level": level}
        )

        if event_type == "performance_issue":
            duration = event.get("context", {}).get("duration_ms", 0)
            if duration > 0:
                self.record_histogram(
                    "lmlog_performance_duration_ms",
                    duration,
                    attributes={
                        "operation": event.get("context", {}).get(
                            "operation", "unknown"
                        )
                    },
                )

        if "error" in event.get("context", {}) or level in ["error", "critical"]:
            self.increment_counter(
                "lmlog_errors_total",
                attributes={"event_type": event_type, "level": level},
            )


class ResourceDetector:
    """
    Detects and provides resource information for OpenTelemetry.
    """

    __slots__ = ("_resource_info",)

    def __init__(self):
        """Initialize resource detector."""
        self._resource_info = self._detect_resource()

    def _detect_resource(self) -> Dict[str, Any]:
        """
        Detect resource information.

        Returns:
            Resource information dictionary
        """
        import platform
        import os

        resource = {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "lmlog-application"),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", "unknown"),
            "service.instance.id": os.getenv("HOSTNAME", platform.node()),
            "telemetry.sdk.name": "lmlog",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": SDK_VERSION,
            "host.name": platform.node(),
            "host.type": platform.machine(),
            "os.name": platform.system(),
            "os.version": platform.release(),
            "process.pid": os.getpid(),
        }

        if "KUBERNETES_SERVICE_HOST" in os.environ:
            k8s_resource = {
                k: v
                for k, v in {
                    "k8s.pod.name": os.getenv("HOSTNAME"),
                    "k8s.namespace.name": os.getenv("K8S_NAMESPACE"),
                    "k8s.cluster.name": os.getenv("K8S_CLUSTER_NAME"),
                }.items()
                if v is not None
            }
            resource.update(k8s_resource)

        return resource

    def get_resource_info(self) -> Dict[str, Any]:
        """
        Get resource information.

        Returns:
            Resource information dictionary
        """
        return self._resource_info.copy()


_global_correlation_context = CorrelationContext()
_global_trace_extractor = TraceContextExtractor()
_global_metric_generator = MetricGenerator()
_global_resource_detector = ResourceDetector()


def get_correlation_context() -> CorrelationContext:
    """Get the global correlation context."""
    return _global_correlation_context


def get_trace_extractor() -> TraceContextExtractor:
    """Get the global trace extractor."""
    return _global_trace_extractor


def get_metric_generator() -> MetricGenerator:
    """Get the global metric generator."""
    return _global_metric_generator


def get_resource_detector() -> ResourceDetector:
    """Get the global resource detector."""
    return _global_resource_detector


def extract_trace_context() -> Dict[str, Any]:
    """
    Extract current trace context.

    Returns:
        Trace context dictionary
    """
    return _global_trace_extractor.extract_context()


def is_otel_available() -> bool:
    """
    Check if OpenTelemetry is available.

    Returns:
        True if OpenTelemetry is available
    """
    return OTEL_AVAILABLE
