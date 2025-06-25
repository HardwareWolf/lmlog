"""
Demo of LMLog performance and features.
"""

import time
from lmlog import OptimizedLLMLogger, capture_errors, log_performance


def main():
    """Demonstrate LMLog capabilities."""
    
    # Initialize high-performance logger
    logger = OptimizedLLMLogger(
        "demo.jsonl",
        encoder="msgspec",  # 10-80x faster than alternatives
        async_writes=True   # Non-blocking I/O
    )
    
    logger.add_global_context(
        app="demo_app",
        version="1.0.0",
        environment="development"
    )
    
    # Example 1: Structured event logging
    print("Example 1: Basic event logging")
    logger.log_event(
        event_type="user_action",
        entity_type="user",
        entity_id="user_123",
        operation="login",
        context={
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "attempt_number": 1
        }
    )
    
    # Example 2: State change tracking
    print("\nExample 2: State change tracking")
    logger.log_state_change(
        entity_type="order",
        entity_id="order_456",
        field="status",
        before="pending",
        after="processing",
        trigger="payment_received"
    )
    
    # Example 3: Performance monitoring
    print("\nExample 3: Performance monitoring")
    logger.log_performance_issue(
        operation="database_query",
        duration_ms=2500,
        threshold_ms=1000,
        context={
            "query": "SELECT * FROM users WHERE active = true",
            "result_count": 10000
        }
    )
    
    # Example 4: Batch operations for high throughput
    print("\nExample 4: Batch event logging")
    events = [
        {
            "event_type": "data_processing",
            "operation": "transform",
            "record_id": f"record_{i}",
            "status": "success"
        }
        for i in range(100)
    ]
    logger.log_batch_events(events)
    
    # Example 5: Operation context tracking
    print("\nExample 5: Operation context")
    with logger.operation_context("user_registration", user_type="premium") as op_id:
        print(f"Operation ID: {op_id}")
        
        # Simulate registration steps
        logger.log_event("step", name="validate_email", status="success")
        time.sleep(0.1)
        
        logger.log_event("step", name="create_account", status="success")
        time.sleep(0.1)
        
        logger.log_event("step", name="send_welcome_email", status="success")
    
    # Example 6: Error logging
    print("\nExample 6: Exception logging")
    import traceback
    
    logger.log_exception(
        exception=ValueError("Invalid user input"),
        operation="data_validation",
        context={
            "input_value": "abc123",
            "expected_format": "numeric"
        }
    )
    
    # Example 7: Performance comparison
    print("\nExample 7: Performance comparison")
    print("Testing serialization performance...")
    
    # Standard JSON approach (slow)
    import json
    start = time.time()
    for _ in range(1000):
        data = {
            "event": "test",
            "timestamp": "2025-06-25T10:00:00Z",
            "data": {"key": "value"}
        }
        json.dumps(data)
    standard_time = time.time() - start
    
    # LMLog with msgspec (fast)
    start = time.time()
    for _ in range(1000):
        logger.log_event(
            event_type="test",
            data={"key": "value"}
        )
    lmlog_time = time.time() - start
    
    print(f"Standard JSON: {standard_time:.4f}s")
    print(f"LMLog (msgspec): {lmlog_time:.4f}s")
    print(f"Speed improvement: {standard_time/lmlog_time:.1f}x faster")
    
    # Flush any remaining events
    logger.flush()
    print(f"\nLog written to: demo.jsonl")


if __name__ == "__main__":
    main()