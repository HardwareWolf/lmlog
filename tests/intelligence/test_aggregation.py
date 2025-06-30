"""
Tests for pattern detection and aggregation system.
"""

import time
from lmlog.intelligence.aggregation import (
    PatternDetector,
    EventAggregator,
    SmartAggregator,
    AggregatedEvent,
)


class TestPatternDetector:
    """Test pattern detection functionality."""

    def test_detect_simple_pattern(self):
        """Test detection of simple patterns."""
        detector = PatternDetector()

        message1 = "User 123 logged in successfully"
        message2 = "User 456 logged in successfully"

        pattern_id1 = detector.detect_pattern(message1)
        pattern_id2 = detector.detect_pattern(message2)

        assert pattern_id1 == pattern_id2
        assert pattern_id1 is not None

    def test_normalize_message_with_numbers(self):
        """Test message normalization with numbers."""
        detector = PatternDetector()

        messages = [
            "Request took 100ms to complete",
            "Request took 250ms to complete",
            "Request took 500ms to complete",
        ]

        pattern_ids = [detector.detect_pattern(msg) for msg in messages]

        # All should have same pattern
        assert len(set(pattern_ids)) == 1

        pattern = detector.get_pattern(pattern_ids[0])
        assert "<NUMBER>" in pattern.pattern_template

    def test_normalize_message_with_ips(self):
        """Test message normalization with IP addresses."""
        detector = PatternDetector()

        messages = [
            "Connection from 192.168.1.1",
            "Connection from 10.0.0.1",
            "Connection from 172.16.0.1",
        ]

        pattern_ids = [detector.detect_pattern(msg) for msg in messages]

        assert len(set(pattern_ids)) == 1

        pattern = detector.get_pattern(pattern_ids[0])
        assert "<IP>" in pattern.pattern_template

    def test_normalize_message_with_paths(self):
        """Test message normalization with file paths."""
        detector = PatternDetector()

        messages = [
            "Error reading file /var/log/app.log",
            "Error reading file /home/user/data.txt",
            "Error reading file /tmp/test.json",
        ]

        pattern_ids = [detector.detect_pattern(msg) for msg in messages]

        assert len(set(pattern_ids)) == 1

        pattern = detector.get_pattern(pattern_ids[0])
        assert "<PATH>" in pattern.pattern_template

    def test_pattern_similarity_threshold(self):
        """Test pattern similarity threshold."""
        detector = PatternDetector(similarity_threshold=0.9)

        message1 = "Database query executed successfully"
        message2 = "Database query failed with error"

        pattern_id1 = detector.detect_pattern(message1)
        pattern_id2 = detector.detect_pattern(message2)

        # Should be different patterns due to high threshold
        assert pattern_id1 != pattern_id2

    def test_pattern_cleanup(self):
        """Test pattern cleanup when limit reached."""
        detector = PatternDetector(max_patterns=5)

        # Create more patterns than limit
        for i in range(10):
            message = f"Unique message pattern {i}"
            detector.detect_pattern(message)

        patterns = detector.get_patterns()
        assert len(patterns) <= 5

    def test_get_patterns_sorted_by_frequency(self):
        """Test getting patterns sorted by frequency."""
        detector = PatternDetector()

        # Create patterns with different frequencies
        for i in range(5):
            detector.detect_pattern("Pattern A occurs frequently")

        for i in range(3):
            detector.detect_pattern("Pattern B occurs sometimes")

        detector.detect_pattern("Pattern C occurs rarely")

        patterns = detector.get_patterns()

        assert len(patterns) == 3
        assert patterns[0].event_count == 5
        assert patterns[1].event_count == 3
        assert patterns[2].event_count == 1


class TestEventAggregator:
    """Test event aggregation functionality."""

    def test_add_event_basic(self):
        """Test adding events for aggregation."""
        aggregator = EventAggregator(window_seconds=60)

        event = {
            "message": "User login successful",
            "user_id": "123",
            "timestamp": time.time(),
        }

        pattern_id = aggregator.add_event(event)
        assert pattern_id is not None

        stats = aggregator.get_statistics()
        assert stats["total_events"] == 1
        assert stats["aggregated_events"] == 1

    def test_aggregate_similar_events(self):
        """Test aggregating similar events."""
        aggregator = EventAggregator(window_seconds=60)

        # Add similar events
        for i in range(10):
            event = {
                "message": f"User {i} performed action",
                "event_type": "user_action",
                "timestamp": time.time(),
            }
            aggregator.add_event(event)

        aggregated = aggregator.get_aggregated_events()

        assert len(aggregated) == 1
        assert aggregated[0].count == 10
        assert "User <NUMBER> performed action" in aggregated[0].pattern

    def test_window_separation(self):
        """Test events in different windows."""
        aggregator = EventAggregator(window_seconds=1)

        # Add event in first window
        event1 = {
            "message": "Event in window 1",
            "timestamp": time.time(),
        }
        aggregator.add_event(event1)

        # Wait for new window
        time.sleep(1.1)

        # Add event in second window
        event2 = {
            "message": "Event in window 2",
            "timestamp": time.time(),
        }
        aggregator.add_event(event2)

        # Should have events in different windows
        stats = aggregator.get_statistics()
        assert stats["active_windows"] == 2

    def test_max_unique_patterns(self):
        """Test maximum unique patterns per window."""
        aggregator = EventAggregator(window_seconds=60, max_unique_patterns=3)

        # Add more unique patterns than limit
        messages = [
            "Database connection failed",
            "Authentication service timeout",
            "Cache memory limit exceeded",
            "Network partition detected",
            "File system disk full",
        ]
        for i in range(5):
            event = {
                "message": messages[i],
                "timestamp": time.time(),
            }
            pattern_id = aggregator.add_event(event)

            # After limit, should return None
            if i >= 3:
                assert pattern_id is None

        aggregated = aggregator.get_aggregated_events()
        assert len(aggregated) == 3

    def test_calculate_statistics(self):
        """Test statistics calculation for aggregated events."""
        aggregator = EventAggregator(window_seconds=60)

        # Add events with various attributes
        events = [
            {
                "message": "API call completed",
                "user_id": "user1",
                "duration_ms": 100,
                "timestamp": time.time(),
            },
            {
                "message": "API call completed",
                "user_id": "user2",
                "duration_ms": 200,
                "timestamp": time.time(),
            },
            {
                "message": "API call completed",
                "user_id": "user1",
                "duration_ms": 150,
                "timestamp": time.time(),
            },
            {
                "message": "API call completed",
                "session_id": "sess1",
                "error_code": "E001",
                "timestamp": time.time(),
            },
        ]

        for event in events:
            aggregator.add_event(event)

        aggregated = aggregator.get_aggregated_events()
        assert len(aggregated) == 1

        stats = aggregated[0].statistics
        assert stats["unique_users"] == 2
        assert stats["avg_duration_ms"] == 150
        assert stats["min_duration_ms"] == 100
        assert stats["max_duration_ms"] == 200

    def test_cleanup_old_windows(self):
        """Test cleanup of old windows."""
        aggregator = EventAggregator(window_seconds=1)

        # Add events
        for i in range(3):
            event = {
                "message": f"Event {i}",
                "timestamp": time.time() - i * 2,
            }
            aggregator.add_event(event)

        # Cleanup old windows
        aggregator.cleanup_old_windows()

        stats = aggregator.get_statistics()
        assert stats["active_windows"] < 3


class TestSmartAggregator:
    """Test smart aggregation functionality."""

    def test_should_aggregate_logic(self):
        """Test logic for determining if event should be aggregated."""
        aggregator = SmartAggregator()

        # Should aggregate normal events
        event1 = {
            "message": "Normal log message that is long enough to aggregate",
            "level": "INFO",
        }
        assert aggregator.should_aggregate(event1) is True

        # Should not aggregate errors
        event2 = {"message": "Error occurred", "level": "ERROR"}
        assert aggregator.should_aggregate(event2) is False

        # Should not aggregate critical events
        event3 = {"message": "Critical issue", "level": "CRITICAL"}
        assert aggregator.should_aggregate(event3) is False

        # Should respect aggregate flag
        event4 = {"message": "Do not aggregate", "aggregate": False}
        assert aggregator.should_aggregate(event4) is False

    def test_process_event_below_threshold(self):
        """Test processing events below auto-aggregation threshold."""
        aggregator = SmartAggregator(auto_aggregate_threshold=5)

        # Add events below threshold
        for i in range(3):
            event = {
                "message": f"User {i} logged in",
                "timestamp": time.time(),
            }
            result = aggregator.process_event(event)
            assert result is None  # Not aggregated yet

    def test_process_event_above_threshold(self):
        """Test processing events above auto-aggregation threshold."""
        aggregator = SmartAggregator(auto_aggregate_threshold=3)

        # Add events above threshold
        for i in range(5):
            event = {
                "message": f"User {i} logged in successfully with session tracking",
                "timestamp": time.time(),
            }
            result = aggregator.process_event(event)

            # Should return aggregated event after threshold
            if i >= 2:
                assert isinstance(result, AggregatedEvent)
                assert result.count >= 3

    def test_enable_disable_aggregation(self):
        """Test enabling and disabling aggregation."""
        aggregator = SmartAggregator()

        # Disable aggregation
        aggregator.disable()

        event = {"message": "Test message that is long enough to be aggregated"}
        assert aggregator.should_aggregate(event) is False

        # Enable aggregation
        aggregator.enable()
        assert aggregator.should_aggregate(event) is True

    def test_get_statistics(self):
        """Test getting aggregation statistics."""
        aggregator = SmartAggregator()

        # Process some events
        for i in range(10):
            event = {
                "message": f"Event type {i % 3} with sufficient length for aggregation",
                "timestamp": time.time(),
            }
            aggregator.process_event(event)

        stats = aggregator.get_statistics()

        assert "enabled" in stats
        assert "auto_threshold" in stats
        assert "total_events" in stats
        assert stats["total_events"] == 10

    def test_aggregation_with_complex_messages(self):
        """Test aggregation with complex log messages."""
        aggregator = SmartAggregator(similarity_threshold=0.8)

        messages = [
            "2024-01-01 10:00:00 [INFO] Processing order #12345 for user john@example.com",
            "2024-01-01 10:00:01 [INFO] Processing order #12346 for user jane@example.com",
            "2024-01-01 10:00:02 [INFO] Processing order #12347 for user bob@example.com",
        ]

        for i, msg in enumerate(messages):
            event = {
                "message": msg,
                "timestamp": time.time(),
            }
            aggregator.process_event(event)

        # Should aggregate similar messages
        aggregated_events = aggregator.get_aggregated_events()
        assert len(aggregated_events) > 0

        # Check pattern contains placeholders
        pattern = aggregated_events[0].pattern
        assert "<NUMBER>" in pattern
        assert "<EMAIL>" in pattern
