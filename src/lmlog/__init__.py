"""
LMLog - LLM-optimized logging library for Python applications.

This library provides structured logging specifically designed for LLM consumption,
enabling better debugging assistance across any Python codebase.
"""

# Core functionality
from .core.base_logger import LLMLogger
from .core.logger import LMLogger
from .core.config import LLMLoggerConfig
from .core.serializers import FastJSONEncoder, MsgSpecEncoder

# Processing
from .processing.async_processing import (
    AsyncEventQueue,
    CircuitBreaker,
    BackpressureManager,
)
from .processing.backends import (
    FileBackend,
    StreamBackend,
    AsyncFileBackend,
    BatchingBackend,
)
from .processing.pools import ObjectPool, EventPool, StringPool, BufferPool

# Intelligence
from .intelligence.classification import (
    IntelligentEventClassifier,
    EventType,
    EventPriority,
    EventClassification,
)
from .intelligence.aggregation import SmartAggregator, AggregatedEvent, PatternDetector
from .intelligence.cost_aware import (
    CostAwareLogger,
    CostBudget,
    CostMetrics,
    StorageTier,
    CompressionLevel,
)
from .intelligence.sampling import (
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

# Integrations
from .integrations.decorators import capture_errors, log_performance, log_calls
from .integrations.otel import extract_trace_context, is_otel_available

__version__ = "0.4.0"
__all__ = [
    # Core loggers
    "LLMLogger",
    "LMLogger",
    # Configuration
    "LLMLoggerConfig",
    # Serializers
    "FastJSONEncoder",
    "MsgSpecEncoder",
    # Processing
    "AsyncEventQueue",
    "CircuitBreaker",
    "BackpressureManager",
    "FileBackend",
    "StreamBackend",
    "AsyncFileBackend",
    "BatchingBackend",
    "ObjectPool",
    "EventPool",
    "StringPool",
    "BufferPool",
    # Intelligence
    "IntelligentEventClassifier",
    "EventType",
    "EventPriority",
    "EventClassification",
    "SmartAggregator",
    "AggregatedEvent",
    "PatternDetector",
    "CostAwareLogger",
    "CostBudget",
    "CostMetrics",
    "StorageTier",
    "CompressionLevel",
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
    # Integrations
    "capture_errors",
    "log_performance",
    "log_calls",
    "extract_trace_context",
    "is_otel_available",
]
