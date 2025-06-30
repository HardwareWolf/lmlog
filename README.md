# LMLog

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/HardwareWolf/lmlog/workflows/Tests/badge.svg)](https://github.com/HardwareWolf/lmlog/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/HardwareWolf/lmlog)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Intelligent structured logging for Python applications, optimized for LLM analysis

LMLog provides high-performance structured logging with built-in intelligence features including ML-based event classification, pattern detection, cost-aware sampling, and async processing. Designed specifically for LLM consumption to enable superior debugging assistance.

## Features

### Core Performance
- **10-80x Faster** - msgspec and orjson serialization
- **Async/Sync Dual API** - Full async/await support with context managers
- **Thread Safe** - Concurrent logging with intelligent buffering
- **Memory Optimized** - Object pooling, string interning, LRU caching
- **Zero Allocation** - Preallocated buffers in hot paths

### Intelligence Layer
- **ML Event Classification** - 9 event types with confidence scoring
- **Smart Aggregation** - Pattern detection and variable extraction
- **Cost-Aware Logging** - Budget management with dynamic sampling
- **Adaptive Sampling** - 8 sampling strategies with auto-selection
- **Anomaly Detection** - Statistical analysis for unusual patterns

### Processing Engine
- **Async Processing** - Non-blocking event queues with circuit breakers
- **Batch Operations** - Configurable batching for high throughput
- **Backpressure Management** - Automatic flow control
- **Multiple Backends** - File, stream, and async I/O support
- **Object Pooling** - Reusable buffers and event dictionaries

## Quick Start

### Basic Usage

```python
from lmlog import LLMLogger

# High-performance logger with msgspec encoding
logger = LLMLogger(
    output="debug.jsonl",
    encoder="msgspec",
    async_processing=True,
    buffer_size=100
)

# Add global context
logger.add_global_context(app="my_app", version="1.0", env="production")

# Log structured events
logger.log_event(
    event_type="data_anomaly",
    entity_type="user",
    entity_id="user_123",
    context={"expected": 100, "actual": 0, "table": "user_metrics"}
)

# Track state changes
logger.log_state_change(
    entity_type="order",
    entity_id="order_456", 
    field="status",
    before="pending",
    after="completed",
    trigger="payment_received"
)

# Monitor performance issues
logger.log_performance_issue(
    operation="database_query",
    duration_ms=5000,
    threshold_ms=1000,
    context={"query": "SELECT * FROM users", "rows": 10000}
)
```

### Enhanced Intelligence Features

```python
from lmlog import LMLogger
from lmlog.intelligence.cost_aware import CostBudget

# Enhanced logger with ML classification and cost awareness
logger = LMLogger(
    output="smart.jsonl",
    enable_classification=True,
    enable_aggregation=True,
    enable_cost_awareness=True,
    cost_budget=CostBudget(
        max_daily_bytes=1024 * 1024 * 1024,  # 1GB
        max_events_per_second=1000
    ),
    aggregation_window=60,
    aggregation_threshold=0.8
)

# Events are automatically classified and aggregated
logger.log_event("payment_failure",
                 context={"amount": 99.99, "card_type": "visa"})
logger.log_event("payment_failure",
                 context={"amount": 149.99, "card_type": "mastercard"})

# Get intelligence insights
stats = logger.get_classification_stats()
aggregated = logger.get_aggregated_events()
cost_metrics = logger.get_cost_metrics()
```

## Decorators

```python
from lmlog import LLMLogger, capture_errors, log_performance, log_calls

logger = LLMLogger("api.jsonl")

@capture_errors(logger, include_traceback=True)
@log_performance(logger, threshold_ms=2000)
@log_calls(logger, include_args=True, include_result=True)
async def process_payment(user_id: str, amount: float):
    # Automatic logging of entry, exit, performance, and errors
    await payment_service.charge(user_id, amount)
    return {"status": "success", "transaction_id": "tx_123"}
```

## Context Managers

```python
# Sync operations
with logger.operation_context("user_registration", user_type="premium") as op_id:
    validate_user(user_data)
    create_account(user_data)
    send_welcome_email(user_data)

# Async operations with correlation
async with logger.aoperation_context("batch_processing") as op_id:
    async for item in data_stream:
        await logger.alog_event("item_processed",
                               item_id=item.id,
                               operation_id=op_id)
        await process_item(item)
```

## Configuration

### From Code

```python
from lmlog import LLMLogger, LLMLoggerConfig

# Programmatic configuration
config = LLMLoggerConfig(
    output="app.jsonl",
    buffer_size=1000,
    auto_flush=True,
    global_context={"service": "api", "version": "2.1.0"}
)

logger = LLMLogger(**config.to_dict())
```

### From JSON File

```python
from lmlog import LLMLoggerConfig, LMLogger

# Load from file
config = LLMLoggerConfig.from_file("logging.json")
logger = LMLogger(**config.to_dict())
```

**logging.json:**
```json
{
  "output": "app.jsonl",
  "enabled": true,
  "buffer_size": 500,
  "auto_flush": true,
  "global_context": {
    "service": "payment-api",
    "environment": "production"
  }
}
```

## Sampling Strategies

```python
from lmlog import LMLogger, ProbabilisticSampler, RateLimitingSampler, create_smart_sampler

# Probabilistic sampling (10% of events)
logger = LMLogger(sampler=ProbabilisticSampler(0.1))

# Rate limiting (max 100 events/second)
logger = LMLogger(sampler=RateLimitingSampler(100))

# Smart adaptive sampling
sampler = create_smart_sampler(
    target_rate=1000,
    priority_weights={"ERROR": 1.0, "INFO": 0.1}
)
logger = LMLogger(sampler=sampler)
```

## Event Types

LMLog intelligently classifies events into these types:

- **ERROR** - Exceptions, failures, critical issues
- **WARNING** - Potential problems, deprecated usage
- **INFO** - General information, state changes
- **DEBUG** - Detailed debugging information
- **PERFORMANCE** - Slow operations, resource issues
- **SECURITY** - Authentication, authorization, threats
- **BUSINESS** - Business rule violations, logic errors
- **AUDIT** - Compliance, regulatory, tracking events
- **UNKNOWN** - Unclassified events

## Performance Optimizations

### Serialization
- **msgspec**: 10-80x faster than pydantic, minimal allocation
- **orjson**: 2-3x faster JSON encoding than stdlib
- **String interning**: Reduces memory for repeated values

### Memory Management
- **Object pooling**: Reusable event dictionaries and buffers
- **LRU caching**: Eliminates repeated caller info lookups
- **Preallocated buffers**: Avoids allocation in hot paths

### Async Processing
- **Non-blocking I/O**: Events processed in background
- **Circuit breakers**: Automatic failure detection and recovery
- **Batch processing**: Efficient bulk operations

## Installation

```bash
pip install lmlog
```

## Requirements

- Python 3.11+ (utilizes latest performance features)
- msgspec (high-performance serialization)
- orjson (fast JSON encoding)

## Why LMLog?

**Traditional Logging:**
```text
2025-06-30 10:30:15 ERROR: Payment failed for user 123
```

**LMLog Output:**
```json
{
  "event_type": "business_rule_violation",
  "timestamp": "2025-06-30T10:30:15.123Z",
  "entity_type": "payment",
  "entity_id": "pay_123",
  "classification": {
    "type": "BUSINESS",
    "priority": "HIGH",
    "confidence": 0.89
  },
  "context": {
    "user_id": "user_123",
    "amount": 1500.00,
    "limit": 1000.00,
    "payment_method": "credit_card",
    "retry_count": 3
  },
  "source": {
    "file": "payment.py",
    "line": 45,
    "function": "process_payment"
  },
  "trace": {
    "trace_id": "abc123def456",
    "span_id": "789xyz"
  },
  "aggregation": {
    "count": 1,
    "pattern": "Payment limit exceeded for amount $<AMOUNT>"
  }
}
```

The structured, intelligent format enables LLMs to quickly understand problems, suggest solutions, and identify patterns across your entire application stack.

## License

MIT License - see [LICENSE](LICENSE) file for details.