"""External integrations and decorators."""

from .otel import TraceContextExtractor, CorrelationContext, MetricGenerator
from .decorators import capture_errors, log_performance, log_calls

__all__ = [
    "TraceContextExtractor",
    "CorrelationContext",
    "MetricGenerator",
    "capture_errors",
    "log_performance",
    "log_calls",
]
