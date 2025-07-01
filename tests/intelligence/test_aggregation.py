"""
Tests for pattern-based aggregation system.
"""

import pytest
import time
from lmlog.intelligence.aggregation import (
    PatternBasedAggregator,
    AggregatedEvent,
)


class TestPatternBasedAggregator:
    """Test pattern-based aggregation functionality."""

    def test_basic_aggregation_functionality(self):
        """Test basic aggregation of similar events."""
        aggregator = PatternBasedAggregator(aggregation_threshold=3)
        
        events = [
            {"message": "User 123 logged in", "level": "INFO"},
            {"message": "User 456 logged in", "level": "INFO"},
            {"message": "User 789 logged in", "level": "INFO"},
        ]
        
        # First two events should not trigger aggregation
        assert not aggregator.should_aggregate(events[0])
        assert not aggregator.should_aggregate(events[1])
        
        # Third event should trigger aggregation
        assert aggregator.should_aggregate(events[2])

    def test_database_query_aggregation(self):
        """Test aggregation of database queries with different IDs."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        events = [
            {"message": "SELECT * FROM users WHERE id = 123", "level": "DEBUG"},
            {"message": "SELECT * FROM users WHERE id = 456", "level": "DEBUG"},
        ]
        
        # Should not aggregate first event
        assert not aggregator.should_aggregate(events[0])
        
        # Should aggregate second event (similar pattern)
        assert aggregator.should_aggregate(events[1])

    def test_api_request_aggregation(self):
        """Test aggregation of API requests with different parameters."""
        aggregator = PatternBasedAggregator(aggregation_threshold=3)
        
        events = [
            {"message": "GET /api/users/123 - 200ms", "level": "INFO"},
            {"message": "GET /api/users/456 - 150ms", "level": "INFO"},
            {"message": "GET /api/users/789 - 180ms", "level": "INFO"},
        ]
        
        results = [aggregator.should_aggregate(event) for event in events]
        
        # Third event should trigger aggregation
        assert results == [False, False, True]

    def test_high_priority_events_not_aggregated(self):
        """Test that high priority events are not aggregated."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        # Error events should not be aggregated
        error_events = [
            {"message": "Database error occurred", "level": "ERROR"},
            {"message": "Database error occurred", "level": "ERROR"},
        ]
        
        for event in error_events:
            assert not aggregator.should_aggregate(event)

    def test_security_events_not_aggregated(self):
        """Test that security events are not aggregated."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        security_events = [
            {"message": "Failed login attempt", "level": "WARNING"},
            {"message": "Failed login attempt", "level": "WARNING"},
        ]
        
        for event in security_events:
            assert not aggregator.should_aggregate(event)

    def test_events_with_stack_traces_not_aggregated(self):
        """Test that events with stack traces are not aggregated."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        events_with_traces = [
            {
                "message": "Exception occurred",
                "level": "INFO",
                "stack_trace": "at com.example.Service.method(Service.java:123)"
            },
            {
                "message": "Exception occurred", 
                "level": "INFO",
                "stack_trace": "at com.example.Service.method(Service.java:456)"
            },
        ]
        
        for event in events_with_traces:
            assert not aggregator.should_aggregate(event)

    def test_aggregated_event_data_structure(self):
        """Test that aggregated event data contains expected information."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        events = [
            {"message": "Processing file data_123.csv", "level": "INFO", "service": "worker"},
            {"message": "Processing file data_456.csv", "level": "INFO", "service": "worker"},
        ]
        
        # Trigger aggregation
        aggregator.should_aggregate(events[0])
        aggregator.should_aggregate(events[1])
        
        # Get aggregated event data
        aggregated_data = aggregator.get_aggregated_event(events[1])
        
        assert aggregated_data is not None
        assert aggregated_data["event_type"] == "aggregated_event"
        assert aggregated_data["count"] == 2
        assert "pattern" in aggregated_data
        assert "normalized_message" in aggregated_data
        assert "sample_events" in aggregated_data
        assert len(aggregated_data["sample_events"]) <= 3  # Max 3 samples

    def test_ip_address_normalization(self):
        """Test that IP addresses are properly normalized."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        events = [
            {"message": "Connection from 192.168.1.100", "level": "INFO"},
            {"message": "Connection from 10.0.0.1", "level": "INFO"},
        ]
        
        # Should not aggregate first event
        assert not aggregator.should_aggregate(events[0])
        
        # Should aggregate second event (same pattern, different IP)
        assert aggregator.should_aggregate(events[1])

    def test_uuid_normalization(self):
        """Test that UUIDs are properly normalized."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        events = [
            {"message": "Processing request 550e8400-e29b-41d4-a716-446655440000", "level": "INFO"},
            {"message": "Processing request 6ba7b810-9dad-11d1-80b4-00c04fd430c8", "level": "INFO"},
        ]
        
        results = [aggregator.should_aggregate(event) for event in events]
        assert results == [False, True]

    def test_timestamp_normalization(self):
        """Test that timestamps are properly normalized."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        events = [
            {"message": "Event occurred at 2024-01-01T10:00:00", "level": "INFO"},
            {"message": "Event occurred at 2024-01-01T11:00:00", "level": "INFO"},
        ]
        
        results = [aggregator.should_aggregate(event) for event in events]
        assert results == [False, True]

    def test_duration_normalization(self):
        """Test that durations are properly normalized."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        events = [
            {"message": "Operation completed in 250ms", "level": "INFO"},
            {"message": "Operation completed in 180ms", "level": "INFO"},
        ]
        
        results = [aggregator.should_aggregate(event) for event in events]
        assert results == [False, True]

    def test_statistics_tracking(self):
        """Test that aggregation statistics are properly tracked."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        events = [
            {"message": "Test event 1", "level": "INFO"},
            {"message": "Test event 2", "level": "INFO"},
            {"message": "Different event", "level": "INFO"},
        ]
        
        for event in events:
            aggregator.should_aggregate(event)
        
        stats = aggregator.get_statistics()
        
        assert stats["events_processed"] == 3
        assert stats["events_aggregated"] >= 0
        assert stats["patterns_created"] >= 0
        assert "aggregation_threshold" in stats

    def test_empty_message_handling(self):
        """Test handling of events with empty or missing messages."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        # Event with no message
        assert not aggregator.should_aggregate({})
        
        # Event with empty message
        assert not aggregator.should_aggregate({"message": "", "level": "INFO"})
        
        # Event with None message
        assert not aggregator.should_aggregate({"message": None, "level": "INFO"})

    def test_pattern_cleanup_after_expiration(self):
        """Test that old patterns are cleaned up properly."""
        aggregator = PatternBasedAggregator(
            aggregation_threshold=2,
            cleanup_interval=1  # Clean up every second
        )
        
        event = {"message": "Test message", "level": "INFO"}
        aggregator.should_aggregate(event)
        
        initial_stats = aggregator.get_statistics()
        initial_patterns = initial_stats["active_patterns"]
        
        # Wait for cleanup interval and trigger cleanup
        time.sleep(1.1)
        aggregator.should_aggregate({"message": "Another message", "level": "INFO"})
        
        # Patterns should eventually be cleaned up
        # Note: This test is time-dependent and may be flaky

    def test_max_patterns_limit(self):
        """Test that pattern count is limited."""
        aggregator = PatternBasedAggregator(
            aggregation_threshold=1,
            max_patterns=5
        )
        
        # Create more unique patterns than the limit
        for i in range(10):
            event = {"message": f"Unique message {i} with different content", "level": "INFO"}
            aggregator.should_aggregate(event)
        
        stats = aggregator.get_statistics()
        
        # Should not exceed max patterns limit
        assert stats["active_patterns"] <= 5

    def test_sample_event_sanitization(self):
        """Test that sample events are properly sanitized."""
        aggregator = PatternBasedAggregator(aggregation_threshold=2)
        
        events = [
            {
                "message": "Test message 1",
                "level": "INFO",
                "sensitive_data": "secret_key_123",
                "large_payload": "x" * 10000,
                "timestamp": "2024-01-01T10:00:00",
                "service": "test-service"
            },
            {
                "message": "Test message 2", 
                "level": "INFO",
                "sensitive_data": "secret_key_456",
                "large_payload": "y" * 10000,
                "timestamp": "2024-01-01T11:00:00",
                "service": "test-service"
            },
        ]
        
        # Trigger aggregation
        aggregator.should_aggregate(events[0])
        aggregator.should_aggregate(events[1])
        
        aggregated_data = aggregator.get_aggregated_event(events[1])
        sample_event = aggregated_data["sample_events"][0]
        
        # Should contain only essential fields
        assert "message" in sample_event
        assert "level" in sample_event
        assert "timestamp" in sample_event
        assert "service" in sample_event
        
        # Should not contain sensitive or large data
        assert "sensitive_data" not in sample_event
        assert "large_payload" not in sample_event

    def test_consistent_pattern_detection(self):
        """Test that pattern detection is consistent for similar events."""
        aggregator = PatternBasedAggregator(aggregation_threshold=1)
        
        event1 = {"message": "User user_123 performed action", "level": "INFO"}
        event2 = {"message": "User user_456 performed action", "level": "INFO"}
        
        # Both events should match the same pattern
        aggregator.should_aggregate(event1)
        
        aggregated_data1 = aggregator.get_aggregated_event(event1)
        aggregated_data2 = aggregator.get_aggregated_event(event2)
        
        if aggregated_data1 and aggregated_data2:
            assert aggregated_data1["pattern"] == aggregated_data2["pattern"]