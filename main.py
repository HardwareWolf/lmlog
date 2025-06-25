import time
from src.lmlog import LLMLogger, capture_errors, log_performance


def main():
    # Initialize logger
    logger = LLMLogger(output="example_log.jsonl")
    logger.add_global_context(app="example_app", version="1.0.0")

    # Example 1: Basic event logging
    logger.log_event(
        event_type="data_anomaly",
        entity_type="user",
        entity_id="user_12345",
        anomaly_type="duplicate_email",
        context={"email": "user@example.com", "existing_user_id": "user_67890"},
    )

    # Example 2: State change logging
    logger.log_state_change(
        entity_type="user_account",
        entity_id="user_12345",
        field="subscription_status",
        before="trial",
        after="premium",
        trigger="payment_processed",
    )

    # Example 3: Performance issue
    logger.log_performance_issue(
        operation="database_query",
        duration_ms=5000,
        threshold_ms=1000,
        context={"query_type": "user_profile", "result_count": 1},
    )

    # Example 4: Using decorators
    @capture_errors(logger)
    @log_performance(logger, threshold_ms=2000)
    def example_function():
        """Simulate some work that might fail or be slow."""
        time.sleep(2.5)  # Simulate slow operation
        if time.time() % 2 > 1:  # Randomly fail sometimes
            raise ValueError("Something went wrong")
        return "success"

    try:
        result = example_function()
        print(f"Function result: {result}")
    except Exception as e:
        print(f"Function failed: {e}")

    # Example 5: Exception logging
    try:
        raise RuntimeError("Example error for demonstration")
    except Exception as e:
        logger.log_exception(
            exception=e,
            operation="example_operation",
            context={"user_id": "user_12345", "action": "process_data"},
        )

    print("Example logging complete. Check 'example_log.jsonl' for output.")


if __name__ == "__main__":
    main()
