"""
LMLog - LLM-optimized logging library for Python applications.

This library provides structured logging specifically designed for LLM consumption,
enabling better debugging assistance across any Python codebase.
"""

from .logger import LLMLogger
from .optimized_logger import OptimizedLLMLogger
from .decorators import capture_errors, log_performance, log_calls
from .config import LLMLoggerConfig
from .serializers import FastJSONEncoder, MsgSpecEncoder, EventSerializer
from .backends import FileBackend, StreamBackend, AsyncFileBackend, BatchingBackend

__version__ = "0.2.0"
__all__ = [
    "LLMLogger",
    "OptimizedLLMLogger",
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
]
