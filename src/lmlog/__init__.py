"""
LMLog - LLM-optimized logging library for Python applications.

This library provides structured logging specifically designed for LLM consumption,
enabling better debugging assistance across any Python codebase.
"""

from .logger import LLMLogger
from .decorators import capture_errors, log_performance, log_calls
from .config import LLMLoggerConfig
from .serializers import FastJSONEncoder, MsgSpecEncoder, EventSerializer
from .backends import FileBackend, StreamBackend, AsyncFileBackend, BatchingBackend
from .pools import ObjectPool, EventPool, StringPool, BufferPool
from .sampling import (
    Sampler,
    AlwaysSampler,
    NeverSampler,
    ProbabilisticSampler,
    RateLimitingSampler,
    AdaptiveSampler,
    CompositeSampler,
    LevelBasedSampler,
    ContextBasedSampler,
    create_smart_sampler,
)
from .async_processing import AsyncEventQueue, CircuitBreaker, BackpressureManager
from .otel_integration import extract_trace_context, is_otel_available

__version__ = "0.2.1"
__all__ = [
    "LLMLogger",
    "capture_errors",
    "log_performance",
    "log_calls",
    "LLMLoggerConfig",
    "FastJSONEncoder",
    "MsgSpecEncoder",
    "EventSerializer",
    "FileBackend",
    "StreamBackend",
    "AsyncFileBackend",
    "BatchingBackend",
    "ObjectPool",
    "EventPool",
    "StringPool",
    "BufferPool",
    "Sampler",
    "AlwaysSampler",
    "NeverSampler",
    "ProbabilisticSampler",
    "RateLimitingSampler",
    "AdaptiveSampler",
    "CompositeSampler",
    "LevelBasedSampler",
    "ContextBasedSampler",
    "create_smart_sampler",
    "AsyncEventQueue",
    "CircuitBreaker",
    "BackpressureManager",
    "extract_trace_context",
    "is_otel_available",
]
