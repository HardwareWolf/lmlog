"""
ML-based event classification system for intelligent log processing.
"""

import re
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import Counter, deque
import math
import json


class EventType(Enum):
    """Event type classifications."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    AUDIT = "audit"
    UNKNOWN = "unknown"


class EventPriority(Enum):
    """Event priority levels."""

    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


@dataclass
class EventClassification:
    """Classification result for an event."""

    event_type: EventType
    priority: EventPriority
    confidence: float
    anomaly_score: float
    features: Dict[str, Any]
    suggested_sampling_rate: float


class FeatureExtractor:
    """Extract features from log events for classification."""

    __slots__ = (
        "_keyword_patterns",
        "_numeric_pattern",
        "_ip_pattern",
        "_url_pattern",
        "_timestamp_pattern",
    )

    def __init__(self):
        """Initialize feature extractor with common patterns."""
        self._keyword_patterns = {
            EventType.ERROR: re.compile(
                r"\b(error|exception|fail|fatal|crash|abort|panic)\b", re.I
            ),
            EventType.WARNING: re.compile(
                r"\b(warn|warning|deprecated|unstable|retry)\b", re.I
            ),
            EventType.PERFORMANCE: re.compile(
                r"\b(slow|latency|timeout|performance|duration|elapsed)\b", re.I
            ),
            EventType.SECURITY: re.compile(
                r"\b(auth|security|permission|denied|unauthorized|forbidden)\b", re.I
            ),
            EventType.BUSINESS: re.compile(
                r"\b(payment|order|transaction|customer|revenue|purchase)\b", re.I
            ),
            EventType.AUDIT: re.compile(
                r"\b(audit|compliance|regulation|policy|access|change)\b", re.I
            ),
        }

        self._numeric_pattern = re.compile(r"\b\d+\.?\d*\b")
        self._ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        self._url_pattern = re.compile(r"https?://[^\s]+")
        self._timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}")

    def extract_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from an event for classification.

        Args:
            event: Event dictionary

        Returns:
            Feature dictionary
        """
        text = self._event_to_text(event)

        features = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "has_stack_trace": self._has_stack_trace(text),
            "numeric_count": len(self._numeric_pattern.findall(text)),
            "has_ip": bool(self._ip_pattern.search(text)),
            "has_url": bool(self._url_pattern.search(text)),
            "has_timestamp": bool(self._timestamp_pattern.search(text)),
            "uppercase_ratio": self._uppercase_ratio(text),
        }

        for event_type, pattern in self._keyword_patterns.items():
            features[f"keyword_{event_type.value}"] = bool(pattern.search(text))

        if "level" in event:
            features["explicit_level"] = event["level"]

        if "duration_ms" in event:
            features["duration_ms"] = event["duration_ms"]
            features["is_slow"] = event["duration_ms"] > 1000

        if "error" in event or "exception" in event:
            features["has_error_field"] = True

        if "user_id" in event or "session_id" in event:
            features["has_user_context"] = True

        return features

    def _event_to_text(self, event: Dict[str, Any]) -> str:
        """Convert event to searchable text."""
        parts = []

        for key, value in event.items():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (int, float)):
                parts.append(f"{key}={value}")
            elif isinstance(value, dict):
                parts.append(json.dumps(value))

        return " ".join(parts)

    def _has_stack_trace(self, text: str) -> bool:
        """Check if text contains a stack trace."""
        stack_indicators = [
            "Traceback",
            "at ",
            "Exception in",
            ".java:",
            ".py:",
            ".js:",
            "at line",
        ]
        return any(indicator in text for indicator in stack_indicators)

    def _uppercase_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase letters."""
        if not text:
            return 0.0

        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0

        uppercase = sum(1 for c in letters if c.isupper())
        return uppercase / len(letters)


class RuleBasedClassifier:
    """Rule-based classifier for event classification."""

    __slots__ = ("_rules", "_feature_extractor")

    def __init__(self):
        """Initialize rule-based classifier."""
        self._feature_extractor = FeatureExtractor()
        self._rules = self._create_rules()

    def classify(self, features: Dict[str, Any]) -> Tuple[EventType, float]:
        """
        Classify event based on features.

        Args:
            features: Feature dictionary

        Returns:
            Tuple of (event_type, confidence)
        """
        scores = Counter()

        for rule_func, event_type, weight in self._rules:
            if rule_func(features):
                scores[event_type] += weight

        if not scores:
            return EventType.UNKNOWN, 0.0

        total_score = sum(scores.values())
        best_type = scores.most_common(1)[0][0]
        confidence = scores[best_type] / total_score

        return best_type, confidence

    def _create_rules(self) -> List[Tuple]:
        """Create classification rules."""
        rules = []

        rules.append(
            (
                lambda f: f.get("keyword_error") or f.get("has_error_field"),
                EventType.ERROR,
                3.0,
            )
        )

        rules.append((lambda f: f.get("keyword_warning"), EventType.WARNING, 2.0))

        rules.append(
            (
                lambda f: f.get("keyword_performance") or f.get("is_slow"),
                EventType.PERFORMANCE,
                2.5,
            )
        )

        rules.append((lambda f: f.get("keyword_security"), EventType.SECURITY, 3.0))

        rules.append((lambda f: f.get("keyword_business"), EventType.BUSINESS, 2.0))

        rules.append((lambda f: f.get("keyword_audit"), EventType.AUDIT, 2.0))

        rules.append((lambda f: f.get("has_stack_trace"), EventType.ERROR, 2.0))

        rules.append(
            (lambda f: f.get("explicit_level") == "ERROR", EventType.ERROR, 4.0)
        )

        rules.append(
            (lambda f: f.get("explicit_level") == "WARNING", EventType.WARNING, 3.0)
        )

        rules.append(
            (lambda f: f.get("uppercase_ratio", 0) > 0.3, EventType.ERROR, 1.5)
        )

        return rules


class AnomalyDetector:
    """Detect anomalies in log events using statistical methods."""

    __slots__ = (
        "_feature_stats",
        "_window_size",
        "_min_samples",
        "_z_threshold",
    )

    def __init__(
        self,
        window_size: int = 1000,
        min_samples: int = 100,
        z_threshold: float = 3.0,
    ):
        """
        Initialize anomaly detector.

        Args:
            window_size: Size of sliding window for statistics
            min_samples: Minimum samples before detecting anomalies
            z_threshold: Z-score threshold for anomaly detection
        """
        self._feature_stats: Dict[str, deque] = {}
        self._window_size = window_size
        self._min_samples = min_samples
        self._z_threshold = z_threshold

    def score(self, features: Dict[str, Any]) -> float:
        """
        Calculate anomaly score for features.

        Args:
            features: Feature dictionary

        Returns:
            Anomaly score (0.0 = normal, 1.0 = highly anomalous)
        """
        numeric_features = {
            k: v for k, v in features.items() if isinstance(v, (int, float))
        }

        if not numeric_features:
            return 0.0

        anomaly_scores = []

        for feature_name, value in numeric_features.items():
            if feature_name not in self._feature_stats:
                self._feature_stats[feature_name] = deque(maxlen=self._window_size)

            stats = self._feature_stats[feature_name]
            stats.append(value)

            if len(stats) < self._min_samples:
                continue

            mean = sum(stats) / len(stats)
            variance = sum((x - mean) ** 2 for x in stats) / len(stats)
            std_dev = math.sqrt(variance) if variance > 0 else 1.0

            z_score = abs(value - mean) / std_dev if std_dev > 0 else 0

            anomaly_score = min(z_score / self._z_threshold, 1.0)
            anomaly_scores.append(anomaly_score)

        return max(anomaly_scores) if anomaly_scores else 0.0


class PriorityCalculator:
    """Calculate event priority based on various factors."""

    @staticmethod
    def calculate_priority(
        event_type: EventType,
        confidence: float,
        anomaly_score: float,
        features: Dict[str, Any],
    ) -> EventPriority:
        """
        Calculate event priority.

        Args:
            event_type: Classified event type
            confidence: Classification confidence
            anomaly_score: Anomaly score
            features: Event features

        Returns:
            Event priority
        """
        base_priority = {
            EventType.ERROR: 4,
            EventType.SECURITY: 4,
            EventType.WARNING: 3,
            EventType.PERFORMANCE: 3,
            EventType.BUSINESS: 3,
            EventType.AUDIT: 2,
            EventType.INFO: 2,
            EventType.DEBUG: 1,
            EventType.UNKNOWN: 2,
        }

        score = base_priority.get(event_type, 2)

        if anomaly_score > 0.8:
            score += 1
        elif anomaly_score > 0.5:
            score += 0.5

        if confidence < 0.5:
            score -= 0.5

        if features.get("has_stack_trace"):
            score += 0.5

        if features.get("has_user_context"):
            score += 0.25

        score = max(1, min(5, round(score)))

        return EventPriority(score)


class SamplingRateCalculator:
    """Calculate optimal sampling rate for events."""

    @staticmethod
    def calculate_rate(
        event_type: EventType,
        priority: EventPriority,
        anomaly_score: float,
        event_frequency: Optional[float] = None,
    ) -> float:
        """
        Calculate suggested sampling rate.

        Args:
            event_type: Event type
            priority: Event priority
            anomaly_score: Anomaly score
            event_frequency: Event frequency (events/second)

        Returns:
            Sampling rate (0.0 to 1.0)
        """
        base_rates = {
            EventPriority.CRITICAL: 1.0,
            EventPriority.HIGH: 1.0,
            EventPriority.MEDIUM: 0.5,
            EventPriority.LOW: 0.1,
            EventPriority.TRIVIAL: 0.01,
        }

        rate = base_rates.get(priority, 0.1)

        if anomaly_score > 0.7:
            rate = 1.0
        elif anomaly_score > 0.5:
            rate = max(rate, 0.5)

        if event_frequency and event_frequency > 100:
            rate *= min(100 / event_frequency, 1.0)

        if event_type in [EventType.ERROR, EventType.SECURITY]:
            rate = max(rate, 0.8)

        return min(1.0, max(0.001, rate))


class IntelligentEventClassifier:
    """
    Main classifier that combines rule-based and statistical approaches.
    """

    __slots__ = (
        "_feature_extractor",
        "_rule_classifier",
        "_anomaly_detector",
        "_pattern_cache",
        "_event_frequencies",
        "_cache_ttl",
        "_cache_size",
        "_max_frequency_count",
    )

    def __init__(
        self,
        cache_size: int = 10000,
        cache_ttl: float = 3600,
        max_frequency_count: int = 10000,
    ):
        """
        Initialize intelligent event classifier.

        Args:
            cache_size: Maximum cache size
            cache_ttl: Cache TTL in seconds
            max_frequency_count: Maximum frequency count before decay
        """
        self._feature_extractor = FeatureExtractor()
        self._rule_classifier = RuleBasedClassifier()
        self._anomaly_detector = AnomalyDetector()
        self._pattern_cache: Dict[str, Tuple[EventClassification, float]] = {}
        self._event_frequencies: Counter = Counter()
        self._cache_ttl = cache_ttl
        self._cache_size = cache_size
        self._max_frequency_count = max_frequency_count

    def classify_event(self, event: Dict[str, Any]) -> EventClassification:
        """
        Classify a log event.

        Args:
            event: Event dictionary

        Returns:
            Event classification
        """
        event_hash = self._hash_event(event)

        cached = self._pattern_cache.get(event_hash)
        if cached:
            classification, timestamp = cached
            if time.time() - timestamp < self._cache_ttl:
                return classification

        features = self._feature_extractor.extract_features(event)

        event_type, confidence = self._rule_classifier.classify(features)

        anomaly_score = self._anomaly_detector.score(features)

        priority = PriorityCalculator.calculate_priority(
            event_type, confidence, anomaly_score, features
        )

        self._update_frequency(event_type)
        frequency = self._get_frequency(event_type)

        sampling_rate = SamplingRateCalculator.calculate_rate(
            event_type, priority, anomaly_score, frequency
        )

        classification = EventClassification(
            event_type=event_type,
            priority=priority,
            confidence=confidence,
            anomaly_score=anomaly_score,
            features=features,
            suggested_sampling_rate=sampling_rate,
        )

        self._pattern_cache[event_hash] = (classification, time.time())

        if len(self._pattern_cache) > self._cache_size:
            self._cleanup_cache()

        return classification

    def _hash_event(self, event: Dict[str, Any]) -> str:
        """Generate hash for event caching."""
        key_parts = []

        for k, v in sorted(event.items()):
            if k in ["timestamp", "id", "trace_id", "span_id"]:
                continue

            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}:{v}")
            elif isinstance(v, dict):
                key_parts.append(f"{k}:{json.dumps(v, sort_keys=True)}")

        key = "|".join(key_parts)
        return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()

    def _update_frequency(self, event_type: EventType) -> None:
        """Update event frequency statistics."""
        self._event_frequencies[event_type] += 1

        total = sum(self._event_frequencies.values())
        if total > self._max_frequency_count:
            for et in self._event_frequencies:
                self._event_frequencies[et] //= 2

    def _get_frequency(self, event_type: EventType) -> float:
        """Get event frequency (events per second estimate)."""
        count = self._event_frequencies.get(event_type, 0)
        return count / 60.0

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired = [
            k
            for k, (_, timestamp) in self._pattern_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]

        for k in expired[: len(expired) // 2]:
            del self._pattern_cache[k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            "cache_size": len(self._pattern_cache),
            "event_frequencies": dict(self._event_frequencies),
            "cache_hit_rate": len(self._pattern_cache)
            / max(1, sum(self._event_frequencies.values())),
        }
