"""
Pattern-based log aggregation for noise reduction.
"""

import re
import time
from typing import Dict, Any, Optional, List, Set, Pattern
from dataclasses import dataclass, field
from collections import Counter


@dataclass(frozen=True)
class AggregationPattern:
    """Defines how similar events should be aggregated."""

    name: str
    normalizer: Pattern[str]
    placeholder: str
    min_occurrences: int
    time_window_seconds: int


@dataclass
class AggregatedEvent:
    """Represents a group of similar events that have been aggregated."""

    pattern: str
    normalized_message: str
    count: int
    first_seen: float
    last_seen: float
    sample_events: List[Dict[str, Any]]
    variables_seen: Set[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregated event to dictionary for logging."""
        return {
            "event_type": "aggregated_event",
            "pattern": self.pattern,
            "normalized_message": self.normalized_message,
            "count": self.count,
            "duration_seconds": self.last_seen - self.first_seen,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "sample_events": self.sample_events[:3],  # Keep only first 3 samples
            "unique_variables": len(self.variables_seen),
        }


class PatternBasedAggregator:
    """
    Fast pattern-based log aggregation using pre-compiled regex patterns.
    
    Groups similar log messages by normalizing variable parts and counting
    occurrences within time windows.
    """

    __slots__ = (
        "_patterns",
        "_active_aggregations",
        "_aggregation_threshold",
        "_max_patterns",
        "_stats",
        "_cleanup_interval",
        "_last_cleanup",
    )

    def __init__(
        self,
        aggregation_threshold: int = 5,
        max_patterns: int = 1000,
        cleanup_interval: int = 300,
    ):
        """
        Initialize pattern-based aggregator.

        Args:
            aggregation_threshold: Minimum occurrences to trigger aggregation
            max_patterns: Maximum number of patterns to track
            cleanup_interval: How often to clean up expired patterns (seconds)
        """
        self._patterns = self._build_aggregation_patterns()
        self._active_aggregations: Dict[str, AggregatedEvent] = {}
        self._aggregation_threshold = aggregation_threshold
        self._max_patterns = max_patterns
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._stats = {
            "events_processed": 0,
            "events_aggregated": 0,
            "patterns_created": 0,
            "patterns_expired": 0,
        }

    def should_aggregate(self, event: Dict[str, Any]) -> bool:
        """
        Determine if an event should be aggregated.

        Args:
            event: Log event to check

        Returns:
            True if event should be aggregated, False if it should be logged normally
        """
        self._stats["events_processed"] += 1

        # Skip aggregation for high-priority events
        if self._is_high_priority_event(event):
            return False

        # Get message for pattern matching
        message = self._extract_message(event)
        if not message:
            return False

        # Find matching pattern and normalize
        pattern_key, normalized_message, variables = self._find_and_normalize_pattern(message)
        if not pattern_key:
            return False

        # Update or create aggregation
        current_time = time.time()
        
        if pattern_key in self._active_aggregations:
            # Update existing aggregation
            aggregation = self._active_aggregations[pattern_key]
            aggregation.count += 1
            aggregation.last_seen = current_time
            aggregation.variables_seen.update(variables)
            
            # Add sample event if we don't have many yet
            if len(aggregation.sample_events) < 3:
                aggregation.sample_events.append(self._sanitize_event_for_sample(event))
                
        else:
            # Create new aggregation
            self._active_aggregations[pattern_key] = AggregatedEvent(
                pattern=pattern_key,
                normalized_message=normalized_message,
                count=1,
                first_seen=current_time,
                last_seen=current_time,
                sample_events=[self._sanitize_event_for_sample(event)],
                variables_seen=set(variables),
            )
            self._stats["patterns_created"] += 1

        # Check if we should emit aggregated event
        aggregation = self._active_aggregations[pattern_key]
        if aggregation.count >= self._aggregation_threshold:
            self._stats["events_aggregated"] += 1
            return True

        # Cleanup old patterns periodically
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired_patterns()

        return False

    def get_aggregated_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get the aggregated event data for a pattern.

        Args:
            event: Original event that triggered aggregation

        Returns:
            Aggregated event data or None if not found
        """
        message = self._extract_message(event)
        if not message:
            return None

        pattern_key, _, _ = self._find_and_normalize_pattern(message)
        if not pattern_key or pattern_key not in self._active_aggregations:
            return None

        aggregation = self._active_aggregations[pattern_key]
        return aggregation.to_dict()

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            "events_processed": self._stats["events_processed"],
            "events_aggregated": self._stats["events_aggregated"],
            "patterns_created": self._stats["patterns_created"],
            "patterns_expired": self._stats["patterns_expired"],
            "active_patterns": len(self._active_aggregations),
            "aggregation_threshold": self._aggregation_threshold,
        }

    def _extract_message(self, event: Dict[str, Any]) -> str:
        """Extract message from event."""
        message = event.get("message", "")
        return message if isinstance(message, str) else ""

    def _is_high_priority_event(self, event: Dict[str, Any]) -> bool:
        """Check if event is high priority and should not be aggregated."""
        # Don't aggregate error or critical events
        level = event.get("level", "").upper()
        if level in ["ERROR", "FATAL", "CRITICAL"]:
            return True

        # Don't aggregate events with stack traces
        if any(field in event for field in ["stack_trace", "stacktrace", "exception"]):
            return True

        # Don't aggregate security-related events
        message = self._extract_message(event).lower()
        security_keywords = ["auth", "login", "security", "unauthorized", "forbidden", "breach"]
        if any(keyword in message for keyword in security_keywords):
            return True

        return False

    def _find_and_normalize_pattern(self, message: str) -> tuple[Optional[str], str, List[str]]:
        """
        Find matching pattern and normalize message.

        Returns:
            Tuple of (pattern_key, normalized_message, extracted_variables)
        """
        message_lower = message.lower()
        
        for pattern in self._patterns:
            matches = list(pattern.normalizer.finditer(message_lower))
            if matches:
                # Extract variables and normalize message
                variables = []
                normalized = message_lower
                
                # Replace matches with placeholders (in reverse order to maintain positions)
                for match in reversed(matches):
                    variables.append(match.group())
                    normalized = (
                        normalized[:match.start()] +
                        pattern.placeholder +
                        normalized[match.end():]
                    )
                
                # Create pattern key by combining pattern name and normalized message
                pattern_key = f"{pattern.name}:{normalized}"
                
                return pattern_key, normalized, variables

        # No pattern matched, return None
        return None, message_lower, []

    def _sanitize_event_for_sample(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Create a sanitized version of event for sample storage."""
        # Keep only essential fields to avoid memory bloat
        sample = {}
        
        keep_fields = [
            "message", "level", "timestamp", "service", "user_id", 
            "duration_ms", "status_code", "event_type"
        ]
        
        for field in keep_fields:
            if field in event:
                sample[field] = event[field]
                
        return sample

    def _cleanup_expired_patterns(self) -> None:
        """Remove expired aggregation patterns."""
        current_time = time.time()
        expired_keys = []
        
        for pattern_key, aggregation in self._active_aggregations.items():
            # Remove patterns older than 5 minutes
            if current_time - aggregation.last_seen > 300:
                expired_keys.append(pattern_key)
        
        for key in expired_keys:
            del self._active_aggregations[key]
            self._stats["patterns_expired"] += 1
        
        # If we still have too many patterns, remove oldest ones
        if len(self._active_aggregations) > self._max_patterns:
            # Sort by last seen time and remove oldest
            sorted_patterns = sorted(
                self._active_aggregations.items(),
                key=lambda x: x[1].last_seen
            )
            
            excess_count = len(self._active_aggregations) - self._max_patterns
            for key, _ in sorted_patterns[:excess_count]:
                del self._active_aggregations[key]
                self._stats["patterns_expired"] += 1
        
        self._last_cleanup = current_time

    def _build_aggregation_patterns(self) -> List[AggregationPattern]:
        """Build the set of aggregation patterns."""
        return [
            # Database query patterns
            AggregationPattern(
                name="db_query",
                normalizer=re.compile(r"\b\d+\b"),  # Replace numbers
                placeholder="<NUM>",
                min_occurrences=5,
                time_window_seconds=60,
            ),
            AggregationPattern(
                name="db_query_ids",
                normalizer=re.compile(r"\b(id|user_id|order_id)\s*=\s*['\"]?[\w-]+['\"]?"),
                placeholder="<ID>",
                min_occurrences=5,
                time_window_seconds=60,
            ),
            
            # API request patterns
            AggregationPattern(
                name="api_request",
                normalizer=re.compile(r"/api/\w+/[\w-]+"),  # API paths with IDs
                placeholder="/api/<RESOURCE>/<ID>",
                min_occurrences=10,
                time_window_seconds=60,
            ),
            AggregationPattern(
                name="api_params",
                normalizer=re.compile(r"[?&]\w+=[^&\s]+"),  # Query parameters
                placeholder="<PARAM>",
                min_occurrences=5,
                time_window_seconds=60,
            ),
            
            # File operation patterns
            AggregationPattern(
                name="file_ops",
                normalizer=re.compile(r"/[\w/.-]+\.(log|tmp|json|csv|xml)"),  # File paths
                placeholder="<FILE>",
                min_occurrences=5,
                time_window_seconds=120,
            ),
            
            # IP address patterns
            AggregationPattern(
                name="ip_addresses",
                normalizer=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
                placeholder="<IP>",
                min_occurrences=10,
                time_window_seconds=60,
            ),
            
            # UUID/Hash patterns
            AggregationPattern(
                name="uuids",
                normalizer=re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
                placeholder="<UUID>",
                min_occurrences=5,
                time_window_seconds=60,
            ),
            AggregationPattern(
                name="hashes",
                normalizer=re.compile(r"\b[0-9a-f]{32,64}\b"),  # MD5/SHA hashes
                placeholder="<HASH>",
                min_occurrences=5,
                time_window_seconds=60,
            ),
            
            # Timestamp patterns
            AggregationPattern(
                name="timestamps",
                normalizer=re.compile(r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}"),
                placeholder="<TIMESTAMP>",
                min_occurrences=10,
                time_window_seconds=30,
            ),
            
            # Duration/size patterns
            AggregationPattern(
                name="durations",
                normalizer=re.compile(r"\b\d+\s*(ms|seconds?|minutes?|hours?)\b"),
                placeholder="<DURATION>",
                min_occurrences=5,
                time_window_seconds=60,
            ),
            AggregationPattern(
                name="sizes",
                normalizer=re.compile(r"\b\d+\s*(bytes?|kb|mb|gb)\b"),
                placeholder="<SIZE>",
                min_occurrences=5,
                time_window_seconds=60,
            ),
        ]