# LMLog

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/HardwareWolf/lmlog/workflows/Tests/badge.svg)](https://github.com/HardwareWolf/lmlog/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/HardwareWolf/lmlog)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checker](https://img.shields.io/badge/type%20checker-pyrefly-blue.svg)](https://pyrefly.org)

> LLM-optimized structured logging for Python applications

LMLog provides structured logging specifically designed for LLM consumption, enabling better debugging assistance across any Python codebase. Instead of human-readable log messages, it captures debugging context that helps LLMs understand and troubleshoot your code effectively.

## Features

- **10-80x Faster** - Uses msgspec and orjson for blazing fast serialization
- **Universal Event Types** - Works with any Python application (web, CLI, data pipelines, etc.)
- **Structured JSON Output** - Machine-readable logs optimized for LLM analysis
- **Async Support** - Full async/await compatibility with context managers
- **Modern Python 3.11+** - Leverages latest performance improvements
- **Thread Safe** - Concurrent logging with buffering support
- **Memory Efficient** - Uses generators, deques, and preallocated buffers
- **Decorators** - Automatic logging with @lru_cache optimization
- **Context Tracking** - Operation correlation and state change monitoring

## Quick Start

```python
from lmlog import OptimizedLLMLogger

# High-performance logger with modern optimizations
logger = OptimizedLLMLogger("debug.jsonl", encoder="msgspec")
logger.add_global_context(app="my_app", version="1.0")

# Log structured events
logger.log_event(
    event_type="data_anomaly",
    entity_type="user",
    entity_id="user_123",
    context={"expected": 100, "actual": 0}
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

# Monitor performance
logger.log_performance_issue(
    operation="database_query",
    duration_ms=5000,
    threshold_ms=1000
)
```

## Decorators

```python
from lmlog import capture_errors, log_performance, log_calls

@capture_errors(logger)
@log_performance(logger, threshold_ms=2000) 
@log_calls(logger)
async def process_data(data):
    # Your code here
    return result
```

## Context Managers

```python
# Sync operations
with logger.operation_context("user_registration") as op_id:
    validate_user(user_data)
    create_account(user_data)

# Async operations  
async with logger.aoperation_context("batch_processing") as op_id:
    await process_batch(items)
```

## Event Types

LMLog captures universal debugging patterns:

- `data_anomaly` - Unexpected values, validation failures
- `performance_issue` - Slow operations, resource problems
- `business_rule_violation` - Logic constraints violated
- `integration_failure` - External API/service errors
- `state_change` - Entity modifications
- `authentication_issue` - Auth/permission problems
- `user_behavior_anomaly` - Security-relevant patterns

## Configuration

```python
from lmlog import LLMLogger, LLMLoggerConfig

# From file
config = LLMLoggerConfig.from_file("logging.json")
logger = LLMLogger(**config.to_dict())

# Programmatic
config = LLMLoggerConfig(
    output="app.jsonl",
    buffer_size=100,
    global_context={"service": "api"}
)
logger = LLMLogger(**config.to_dict())
```

## Performance

LMLog uses cutting-edge optimizations:

- **msgspec** - Up to 80x faster than pydantic for serialization
- **orjson** - 2-3x faster JSON encoding than stdlib
- **functools.lru_cache** - Caches repeated operations
- **Preallocated buffers** - Avoids memory reallocation
- **Method caching** - Eliminates attribute lookups in hot paths
- **Batch operations** - Efficient bulk event processing

## Installation

```bash
pip install lmlog
```

## Requirements

- Python 3.11+ (leverages latest performance features)
- msgspec (high-performance serialization)
- orjson (fast JSON encoding)

## Development

```bash
git clone https://github.com/username/lmlog.git
cd lmlog
make install-dev
make test
make lint
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Why LMLog?

Traditional logs are designed for humans. LMLog is designed for LLMs to help you debug faster:

**Traditional Logging:**

```text
2025-06-25 10:30:15 ERROR: Payment failed for user 123
```

**LMLog Output:**

```json
{
  "event_type": "business_rule_violation",
  "timestamp": "2025-06-25T10:30:15Z",
  "entity": {"type": "user", "id": "123"},
  "violation": {
    "rule": "payment_limit_exceeded", 
    "expected": 1000,
    "actual": 1500
  },
  "context": {"payment_method": "credit_card", "retry_count": 3},
  "source": {"file": "payment.py", "line": 45, "function": "process_payment"}
}
```

The structured format enables LLMs to quickly understand the problem, suggest solutions, and identify patterns across your application.