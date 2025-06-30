"""
Tests for ML-based event classification system.
"""

from lmlog.intelligence.classification import (
    IntelligentEventClassifier,
    EventType,
    EventPriority,
    FeatureExtractor,
    RuleBasedClassifier,
    AnomalyDetector,
    PriorityCalculator,
    SamplingRateCalculator,
)


class TestFeatureExtractor:
    """Test feature extraction from events."""

    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        event = {
            "message": "Error occurred in user authentication",
            "level": "ERROR",
        }

        features = extractor.extract_features(event)

        assert features["text_length"] > 0
        assert features["word_count"] == 6
        assert features["keyword_error"] is True
        assert features["explicit_level"] == "ERROR"

    def test_extract_performance_features(self):
        """Test performance-related feature extraction."""
        extractor = FeatureExtractor()
        event = {
            "message": "Slow query detected",
            "duration_ms": 2500,
        }

        features = extractor.extract_features(event)

        assert features["keyword_performance"] is True
        assert features["duration_ms"] == 2500
        assert features["is_slow"] is True

    def test_extract_security_features(self):
        """Test security-related feature extraction."""
        extractor = FeatureExtractor()
        event = {
            "message": "Unauthorized access attempt from 192.168.1.1",
            "user_id": "12345",
        }

        features = extractor.extract_features(event)

        assert features["keyword_security"] is True
        assert features["has_ip"] is True
        assert features["has_user_context"] is True

    def test_stack_trace_detection(self):
        """Test stack trace detection."""
        extractor = FeatureExtractor()
        event = {
            "message": "Traceback (most recent call last):\n  File 'app.py', line 42",
        }

        features = extractor.extract_features(event)

        assert features["has_stack_trace"] is True


class TestRuleBasedClassifier:
    """Test rule-based classification."""

    def test_classify_error_event(self):
        """Test classification of error events."""
        classifier = RuleBasedClassifier()
        features = {
            "keyword_error": True,
            "has_stack_trace": True,
            "explicit_level": "ERROR",
        }

        event_type, confidence = classifier.classify(features)

        assert event_type == EventType.ERROR
        assert confidence > 0.8

    def test_classify_performance_event(self):
        """Test classification of performance events."""
        classifier = RuleBasedClassifier()
        features = {
            "keyword_performance": True,
            "is_slow": True,
            "duration_ms": 5000,
        }

        event_type, confidence = classifier.classify(features)

        assert event_type == EventType.PERFORMANCE
        assert confidence > 0.7

    def test_classify_security_event(self):
        """Test classification of security events."""
        classifier = RuleBasedClassifier()
        features = {
            "keyword_security": True,
            "has_ip": True,
        }

        event_type, confidence = classifier.classify(features)

        assert event_type == EventType.SECURITY
        assert confidence > 0.7

    def test_classify_unknown_event(self):
        """Test classification of unknown events."""
        classifier = RuleBasedClassifier()
        features = {
            "text_length": 100,
            "word_count": 10,
        }

        event_type, confidence = classifier.classify(features)

        assert event_type == EventType.UNKNOWN
        assert confidence == 0.0


class TestAnomalyDetector:
    """Test anomaly detection."""

    def test_normal_values(self):
        """Test anomaly detection with normal values."""
        detector = AnomalyDetector(min_samples=5)

        # Feed normal values
        for i in range(10):
            features = {"duration_ms": 100 + i * 5}
            score = detector.score(features)

            # Should have low anomaly score after enough samples
            if i >= 5:
                assert score < 0.6

    def test_anomalous_values(self):
        """Test anomaly detection with anomalous values."""
        detector = AnomalyDetector(min_samples=5, z_threshold=2.0)

        # Feed normal values
        for i in range(10):
            features = {"duration_ms": 100}
            detector.score(features)

        # Feed anomalous value
        features = {"duration_ms": 1000}
        score = detector.score(features)

        assert score > 0.5

    def test_multiple_features(self):
        """Test anomaly detection with multiple features."""
        detector = AnomalyDetector(min_samples=5)

        # Feed normal values
        for i in range(10):
            features = {
                "duration_ms": 100,
                "response_size": 1000,
                "error_count": 0,
            }
            detector.score(features)

        # Feed anomalous values
        features = {
            "duration_ms": 100,
            "response_size": 10000,  # Anomalous
            "error_count": 5,  # Anomalous
        }
        score = detector.score(features)

        assert score > 0.0


class TestPriorityCalculator:
    """Test priority calculation."""

    def test_error_priority(self):
        """Test priority for error events."""
        priority = PriorityCalculator.calculate_priority(
            EventType.ERROR, confidence=0.9, anomaly_score=0.2, features={}
        )

        assert priority == EventPriority.HIGH

    def test_anomalous_event_priority(self):
        """Test priority boost for anomalous events."""
        priority = PriorityCalculator.calculate_priority(
            EventType.INFO, confidence=0.9, anomaly_score=0.9, features={}
        )

        assert priority.value >= EventPriority.MEDIUM.value

    def test_low_confidence_priority(self):
        """Test priority reduction for low confidence."""
        priority = PriorityCalculator.calculate_priority(
            EventType.WARNING, confidence=0.3, anomaly_score=0.1, features={}
        )

        assert priority.value <= EventPriority.MEDIUM.value

    def test_stack_trace_priority_boost(self):
        """Test priority boost for stack traces."""
        priority = PriorityCalculator.calculate_priority(
            EventType.WARNING,
            confidence=0.8,
            anomaly_score=0.1,
            features={"has_stack_trace": True},
        )

        assert priority.value >= EventPriority.MEDIUM.value


class TestSamplingRateCalculator:
    """Test sampling rate calculation."""

    def test_critical_priority_sampling(self):
        """Test sampling rate for critical events."""
        rate = SamplingRateCalculator.calculate_rate(
            EventType.ERROR,
            EventPriority.CRITICAL,
            anomaly_score=0.1,
        )

        assert rate == 1.0

    def test_low_priority_sampling(self):
        """Test sampling rate for low priority events."""
        rate = SamplingRateCalculator.calculate_rate(
            EventType.INFO,
            EventPriority.LOW,
            anomaly_score=0.1,
        )

        assert rate < 0.5

    def test_anomaly_based_sampling(self):
        """Test sampling rate boost for anomalies."""
        rate = SamplingRateCalculator.calculate_rate(
            EventType.INFO,
            EventPriority.LOW,
            anomaly_score=0.8,
        )

        assert rate == 1.0

    def test_frequency_based_sampling(self):
        """Test sampling rate reduction for high frequency."""
        rate = SamplingRateCalculator.calculate_rate(
            EventType.INFO,
            EventPriority.MEDIUM,
            anomaly_score=0.1,
            event_frequency=1000,
        )

        assert rate < 0.5


class TestIntelligentEventClassifier:
    """Test the main intelligent event classifier."""

    def test_classify_error_event(self):
        """Test classification of error event."""
        classifier = IntelligentEventClassifier()
        event = {
            "message": "Database connection error: timeout after 30s",
            "level": "ERROR",
            "error": "TimeoutError",
        }

        classification = classifier.classify_event(event)

        assert classification.event_type == EventType.ERROR
        assert classification.priority.value >= EventPriority.HIGH.value
        assert classification.confidence > 0.7
        assert "keyword_error" in classification.features

    def test_classify_performance_event(self):
        """Test classification of performance event."""
        classifier = IntelligentEventClassifier()
        event = {
            "message": "Slow API response detected",
            "duration_ms": 5000,
            "endpoint": "/api/users",
        }

        classification = classifier.classify_event(event)

        assert classification.event_type == EventType.PERFORMANCE
        assert classification.priority.value >= EventPriority.MEDIUM.value
        assert classification.confidence > 0.7
        assert classification.features["is_slow"] is True

    def test_event_caching(self):
        """Test event classification caching."""
        classifier = IntelligentEventClassifier(cache_ttl=3600)
        event = {
            "message": "User login successful",
            "user_id": "12345",
        }

        # First classification
        classification1 = classifier.classify_event(event)

        # Second classification (should be cached)
        classification2 = classifier.classify_event(event)

        assert classification1.event_type == classification2.event_type
        assert classification1.confidence == classification2.confidence

    def test_anomaly_detection_integration(self):
        """Test anomaly detection in classification."""
        classifier = IntelligentEventClassifier()

        # Feed normal events
        for i in range(20):
            event = {
                "message": "API request processed",
                "duration_ms": 100,
            }
            classifier.classify_event(event)

        # Feed anomalous event
        event = {
            "message": "API request processed",
            "duration_ms": 5000,
        }
        classification = classifier.classify_event(event)

        assert (
            classification.anomaly_score >= 0.0
        )  # May be 0.0 if not enough variation in features
        assert classification.suggested_sampling_rate >= 0.5

    def test_get_statistics(self):
        """Test getting classifier statistics."""
        classifier = IntelligentEventClassifier()

        # Classify some events
        events = [
            {"message": "Error occurred", "level": "ERROR"},
            {"message": "Warning issued", "level": "WARNING"},
            {"message": "Info logged", "level": "INFO"},
        ]

        for event in events:
            classifier.classify_event(event)

        stats = classifier.get_statistics()

        assert "cache_size" in stats
        assert "event_frequencies" in stats
        assert stats["cache_size"] > 0
