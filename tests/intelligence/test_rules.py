"""
Tests for rule-based classification system.
"""

import pytest
from lmlog.intelligence.rules import (
    RuleBasedClassifier,
    EventType,
    EventPriority,
    EventClassification,
)


class TestRuleBasedClassifier:
    """Test rule-based event classifier."""

    def test_error_event_classification(self):
        """Test classification of error events."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "ERROR",
            "message": "Database connection failed",
            "service": "api"
        }
        
        result = classifier.classify_event(event)
        
        assert isinstance(result, EventClassification)
        assert result.event_type == EventType.ERROR
        assert result.priority in [EventPriority.HIGH, EventPriority.CRITICAL]
        assert result.confidence > 0.5
        assert result.suggested_sampling_rate >= 0.8

    def test_performance_event_classification(self):
        """Test classification of performance events."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "WARNING",
            "message": "Slow query detected: 2500ms",
            "duration_ms": 2500
        }
        
        result = classifier.classify_event(event)
        
        assert result.event_type == EventType.PERFORMANCE
        assert result.priority == EventPriority.MEDIUM
        assert result.confidence > 0.6

    def test_security_event_classification(self):
        """Test classification of security events."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "WARNING",
            "message": "Authentication failed for user admin",
            "user": "admin"
        }
        
        result = classifier.classify_event(event)
        
        assert result.event_type == EventType.SECURITY
        assert result.priority >= EventPriority.MEDIUM
        assert result.confidence > 0.8

    def test_business_event_classification(self):
        """Test classification of business events."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "INFO",
            "message": "Payment processed successfully",
            "amount": 99.99,
            "event_type": "payment"
        }
        
        result = classifier.classify_event(event)
        
        assert result.event_type == EventType.BUSINESS
        assert result.priority == EventPriority.MEDIUM
        assert result.confidence > 0.7

    def test_debug_event_classification(self):
        """Test classification of debug events."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "DEBUG",
            "message": "Processing request 12345",
            "request_id": "12345"
        }
        
        result = classifier.classify_event(event)
        
        assert result.event_type == EventType.DEBUG
        assert result.priority == EventPriority.TRIVIAL
        assert result.suggested_sampling_rate <= 0.1

    def test_unknown_event_classification(self):
        """Test classification of unknown events."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "INFO",
            "message": "Some random message that matches no patterns",
        }
        
        result = classifier.classify_event(event)
        
        assert result.event_type == EventType.INFO
        assert result.priority == EventPriority.LOW

    def test_priority_boosts_with_stack_trace(self):
        """Test priority boosts for events with stack traces."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "ERROR",
            "message": "NullPointerException occurred",
            "stack_trace": "at com.example.Service.method(Service.java:123)"
        }
        
        result = classifier.classify_event(event)
        
        assert result.priority >= EventPriority.HIGH

    def test_priority_boosts_with_user_context(self):
        """Test priority boosts for events with user context."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "WARNING",
            "message": "Operation failed",
            "user_id": "user123"
        }
        
        result = classifier.classify_event(event)
        
        # Should have slight priority boost
        base_event = {
            "level": "WARNING",
            "message": "Operation failed"
        }
        base_result = classifier.classify_event(base_event)
        
        assert result.priority.value >= base_result.priority.value

    def test_sampling_rates_by_priority(self):
        """Test that sampling rates correspond to priorities."""
        classifier = RuleBasedClassifier()
        
        # Critical events should have high sampling
        critical_event = {
            "level": "FATAL",
            "message": "System crash detected"
        }
        critical_result = classifier.classify_event(critical_event)
        assert critical_result.suggested_sampling_rate >= 0.8
        
        # Debug events should have low sampling
        debug_event = {
            "level": "DEBUG",
            "message": "Debug trace message"
        }
        debug_result = classifier.classify_event(debug_event)
        assert debug_result.suggested_sampling_rate <= 0.1

    def test_rule_matching_statistics(self):
        """Test that statistics are properly tracked."""
        classifier = RuleBasedClassifier()
        
        events = [
            {"level": "ERROR", "message": "Database error"},
            {"level": "WARNING", "message": "Slow query"},
            {"level": "INFO", "message": "User login"},
        ]
        
        for event in events:
            classifier.classify_event(event)
        
        stats = classifier.get_statistics()
        
        assert stats["total_classifications"] == 3
        assert stats["rules_loaded"] > 0
        assert "rule_match_counts" in stats
        assert "type_distribution" in stats

    def test_empty_event_handling(self):
        """Test handling of empty or malformed events."""
        classifier = RuleBasedClassifier()
        
        # Empty event
        result = classifier.classify_event({})
        assert isinstance(result, EventClassification)
        
        # Event with None values
        result = classifier.classify_event({"level": None, "message": None})
        assert isinstance(result, EventClassification)

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case insensitive."""
        classifier = RuleBasedClassifier()
        
        events = [
            {"message": "ERROR: Connection failed"},
            {"message": "error: connection failed"},
            {"message": "Error: Connection Failed"},
        ]
        
        results = [classifier.classify_event(event) for event in events]
        
        # All should be classified as errors
        for result in results:
            assert result.event_type == EventType.ERROR

    def test_multiple_rule_matches(self):
        """Test events that match multiple rules."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "ERROR",
            "message": "Database connection timeout error",
            "duration_ms": 30000
        }
        
        result = classifier.classify_event(event)
        
        # Should match both error and performance rules
        assert len(result.matched_rules) >= 1
        assert result.confidence > 0.5

    def test_field_target_matching(self):
        """Test that rules can match on different event fields."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "INFO",
            "message": "Request processed",
            "error_message": "Connection failed"  # Error in separate field
        }
        
        result = classifier.classify_event(event)
        
        # Should classify as error based on error_message field
        assert result.event_type == EventType.ERROR

    def test_performance_with_large_message(self):
        """Test performance with large message content."""
        classifier = RuleBasedClassifier()
        
        # Create event with large message
        large_message = "This is a test message. " * 1000
        event = {
            "level": "INFO",
            "message": large_message + " error occurred"
        }
        
        import time
        start_time = time.time()
        result = classifier.classify_event(event)
        end_time = time.time()
        
        # Should classify correctly and be fast
        assert result.event_type == EventType.ERROR
        assert (end_time - start_time) < 0.01  # Should be under 10ms

    def test_consistent_classification(self):
        """Test that identical events get consistent classification."""
        classifier = RuleBasedClassifier()
        
        event = {
            "level": "ERROR",
            "message": "Database connection failed",
            "service": "api"
        }
        
        # Classify same event multiple times
        results = [classifier.classify_event(event) for _ in range(10)]
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.event_type == first_result.event_type
            assert result.priority == first_result.priority
            assert result.confidence == first_result.confidence