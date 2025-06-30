"""
Pattern detection and intelligent log aggregation system.
"""

import hashlib
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class LogPattern:
    """Represents a detected log pattern."""

    pattern_id: str
    pattern_template: str
    variable_positions: List[Tuple[int, int]]
    event_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    sample_events: List[Dict[str, Any]] = field(default_factory=list)
    variable_values: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )


@dataclass
class AggregatedEvent:
    """Represents an aggregated log event."""

    pattern_id: str
    event_type: str
    pattern: str
    count: int
    time_window: Dict[str, float]
    sample_events: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    variables: Dict[str, List[str]]


class PatternDetector:
    """Detect patterns in log messages using various techniques."""

    __slots__ = (
        "_patterns",
        "_pattern_index",
        "_similarity_threshold",
        "_min_pattern_length",
        "_variable_patterns",
        "_max_patterns",
    )

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_pattern_length: int = 10,
        max_patterns: int = 1000,
    ):
        """
        Initialize pattern detector.

        Args:
            similarity_threshold: Minimum similarity for pattern matching
            min_pattern_length: Minimum length for pattern consideration
            max_patterns: Maximum number of patterns to track
        """
        self._patterns: Dict[str, LogPattern] = {}
        self._pattern_index: Dict[str, List[str]] = defaultdict(list)
        self._similarity_threshold = similarity_threshold
        self._min_pattern_length = min_pattern_length
        self._max_patterns = max_patterns

        self._variable_patterns = [
            (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
            (re.compile(r"\d+"), "<NUMBER>"),
            (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<HEX>"),
            (
                re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
                "<EMAIL>",
            ),
            (
                re.compile(
                    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
                ),
                "<UUID>",
            ),
            (re.compile(r"/[a-zA-Z0-9/_.-]+"), "<PATH>"),
            (re.compile(r'"[^"]*"'), "<STRING>"),
            (re.compile(r"'[^']*'"), "<STRING>"),
        ]

    def detect_pattern(self, message: str) -> Optional[str]:
        """
        Detect or match pattern for a log message.

        Args:
            message: Log message

        Returns:
            Pattern ID if found or created
        """
        if len(message) < self._min_pattern_length:
            return None

        normalized, variables = self._normalize_message(message)

        pattern_id = self._find_similar_pattern(normalized)

        if pattern_id:
            self._update_pattern(pattern_id, message, variables)
        else:
            pattern_id = self._create_pattern(normalized, message, variables)

        return pattern_id

    def _normalize_message(self, message: str) -> Tuple[str, Dict[str, str]]:
        """
        Normalize message by replacing variables with placeholders.

        Args:
            message: Original message

        Returns:
            Tuple of (normalized message, variable mappings)
        """
        normalized = message
        variables = {}

        for pattern, placeholder in self._variable_patterns:
            matches = list(pattern.finditer(normalized))
            for i, match in enumerate(reversed(matches)):
                var_name = f"{placeholder}_{i}"
                variables[var_name] = match.group()
                normalized = (
                    normalized[: match.start()]
                    + placeholder
                    + normalized[match.end() :]
                )

        return normalized, variables

    def _find_similar_pattern(self, normalized: str) -> Optional[str]:
        """Find similar existing pattern."""
        tokens = normalized.split()[:3]

        if not tokens:
            return None

        key = " ".join(tokens)
        candidates = self._pattern_index.get(key, [])

        for pattern_id in candidates:
            pattern = self._patterns.get(pattern_id)
            if not pattern:
                continue

            similarity = SequenceMatcher(
                None, pattern.pattern_template, normalized
            ).ratio()

            if similarity >= self._similarity_threshold:
                return pattern_id

        return None

    def _create_pattern(
        self, normalized: str, original: str, variables: Dict[str, str]
    ) -> str:
        """Create new pattern."""
        if len(self._patterns) >= self._max_patterns:
            self._cleanup_patterns()

        pattern_id = hashlib.md5(
            normalized.encode(), usedforsecurity=False
        ).hexdigest()[:16]

        variable_positions = []
        for placeholder in [
            "<NUMBER>",
            "<HEX>",
            "<IP>",
            "<EMAIL>",
            "<UUID>",
            "<PATH>",
            "<STRING>",
        ]:
            pos = 0
            while True:
                pos = normalized.find(placeholder, pos)
                if pos == -1:
                    break
                variable_positions.append((pos, pos + len(placeholder)))
                pos += len(placeholder)

        pattern = LogPattern(
            pattern_id=pattern_id,
            pattern_template=normalized,
            variable_positions=variable_positions,
            event_count=1,
            sample_events=[{"message": original}][:5],
        )

        for var_name, var_value in variables.items():
            pattern.variable_values[var_name].add(var_value)

        self._patterns[pattern_id] = pattern

        tokens = normalized.split()[:3]
        if tokens:
            key = " ".join(tokens)
            self._pattern_index[key].append(pattern_id)

        return pattern_id

    def _update_pattern(
        self, pattern_id: str, message: str, variables: Dict[str, str]
    ) -> None:
        """Update existing pattern with new occurrence."""
        pattern = self._patterns.get(pattern_id)
        if not pattern:
            return

        pattern.event_count += 1
        pattern.last_seen = time.time()

        if len(pattern.sample_events) < 5:
            pattern.sample_events.append({"message": message})

        for var_name, var_value in variables.items():
            if len(pattern.variable_values[var_name]) < 100:
                pattern.variable_values[var_name].add(var_value)

    def _cleanup_patterns(self) -> None:
        """Remove least recently used patterns."""
        sorted_patterns = sorted(self._patterns.items(), key=lambda x: x[1].last_seen)

        to_remove = len(sorted_patterns) // 4

        for pattern_id, _ in sorted_patterns[:to_remove]:
            del self._patterns[pattern_id]

            for patterns in self._pattern_index.values():
                if pattern_id in patterns:
                    patterns.remove(pattern_id)

    def get_pattern(self, pattern_id: str) -> Optional[LogPattern]:
        """Get pattern by ID."""
        return self._patterns.get(pattern_id)

    def get_patterns(self) -> List[LogPattern]:
        """Get all patterns sorted by frequency."""
        return sorted(
            self._patterns.values(), key=lambda p: p.event_count, reverse=True
        )


class EventAggregator:
    """Aggregate events based on patterns and time windows."""

    __slots__ = (
        "_window_seconds",
        "_max_unique_patterns",
        "_active_windows",
        "_pattern_detector",
        "_aggregation_stats",
    )

    def __init__(
        self,
        window_seconds: int = 60,
        max_unique_patterns: int = 100,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize event aggregator.

        Args:
            window_seconds: Time window for aggregation
            max_unique_patterns: Maximum unique patterns per window
            similarity_threshold: Pattern similarity threshold
        """
        self._window_seconds = window_seconds
        self._max_unique_patterns = max_unique_patterns
        self._pattern_detector = PatternDetector(
            similarity_threshold=similarity_threshold
        )
        self._active_windows: Dict[str, Dict[str, Any]] = {}
        self._aggregation_stats = {
            "total_events": 0,
            "aggregated_events": 0,
            "unique_patterns": 0,
        }

    def add_event(self, event: Dict[str, Any]) -> Optional[str]:
        """
        Add event for aggregation.

        Args:
            event: Event to aggregate

        Returns:
            Pattern ID if aggregated, None otherwise
        """
        self._aggregation_stats["total_events"] += 1

        message = self._extract_message(event)
        if not message:
            return None

        pattern_id = self._pattern_detector.detect_pattern(message)
        if not pattern_id:
            return None

        window_key = self._get_window_key(event.get("timestamp", time.time()))

        if window_key not in self._active_windows:
            self._active_windows[window_key] = {}

        window = self._active_windows[window_key]

        if pattern_id not in window:
            if len(window) >= self._max_unique_patterns:
                return None

            window[pattern_id] = {
                "events": [],
                "count": 0,
                "first_timestamp": time.time(),
                "last_timestamp": time.time(),
                "event_type": event.get("event_type", "unknown"),
            }

        window_data = window[pattern_id]
        window_data["count"] += 1
        window_data["last_timestamp"] = time.time()

        if len(window_data["events"]) < 5:
            window_data["events"].append(event)

        self._aggregation_stats["aggregated_events"] += 1

        return pattern_id

    def get_aggregated_events(
        self, window_key: Optional[str] = None
    ) -> List[AggregatedEvent]:
        """
        Get aggregated events for a window.

        Args:
            window_key: Window key (default: current window)

        Returns:
            List of aggregated events
        """
        if window_key is None:
            window_key = self._get_window_key(time.time())

        window = self._active_windows.get(window_key, {})
        aggregated = []

        for pattern_id, data in window.items():
            pattern = self._pattern_detector.get_pattern(pattern_id)
            if not pattern:
                continue

            statistics = self._calculate_statistics(data["events"])

            variables = {}
            for var_name, var_values in pattern.variable_values.items():
                variables[var_name] = list(var_values)[:10]

            aggregated.append(
                AggregatedEvent(
                    pattern_id=pattern_id,
                    event_type=data["event_type"],
                    pattern=pattern.pattern_template,
                    count=data["count"],
                    time_window={
                        "start": data["first_timestamp"],
                        "end": data["last_timestamp"],
                    },
                    sample_events=data["events"],
                    statistics=statistics,
                    variables=variables,
                )
            )

        return sorted(aggregated, key=lambda x: x.count, reverse=True)

    def cleanup_old_windows(self) -> None:
        """Remove old aggregation windows."""
        current_time = time.time()
        cutoff_time = current_time - (self._window_seconds * 2)

        to_remove = []
        for window_key in self._active_windows:
            window_time = float(window_key.split("_")[1])
            if window_time < cutoff_time:
                to_remove.append(window_key)

        for window_key in to_remove:
            del self._active_windows[window_key]

    def _extract_message(self, event: Dict[str, Any]) -> Optional[str]:
        """Extract message from event."""
        for key in ["message", "msg", "text", "error", "description"]:
            if key in event and isinstance(event[key], str):
                return event[key]

        if "event_type" in event and "context" in event:
            parts = [str(event["event_type"])]
            if isinstance(event["context"], dict):
                parts.extend(f"{k}={v}" for k, v in event["context"].items())
            return " ".join(parts)

        return None

    def _get_window_key(self, timestamp: Union[float, str]) -> str:
        """Get window key for timestamp."""
        if isinstance(timestamp, str):
            import datetime

            ts = datetime.datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            ).timestamp()
        else:
            ts = timestamp
        window_start = int(ts // self._window_seconds) * self._window_seconds
        return f"window_{window_start}"

    def _calculate_statistics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for aggregated events."""
        stats: Dict[str, Any] = {
            "unique_users": len(
                set(e.get("user_id") for e in events if e.get("user_id"))
            ),
            "unique_sessions": len(
                set(e.get("session_id") for e in events if e.get("session_id"))
            ),
        }

        durations = []
        for e in events:
            duration = e.get("duration_ms")
            if duration is not None:
                durations.append(float(duration))

        if durations:
            stats["avg_duration_ms"] = sum(durations) / len(durations)
            stats["min_duration_ms"] = min(durations)
            stats["max_duration_ms"] = max(durations)

        error_codes = [
            str(e.get("error_code")) for e in events if e.get("error_code") is not None
        ]

        if error_codes:
            error_counts = Counter(error_codes)
            stats["top_errors"] = error_counts.most_common(5)

        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        stats = self._aggregation_stats.copy()
        stats["active_windows"] = len(self._active_windows)
        stats["total_patterns"] = sum(
            len(window) for window in self._active_windows.values()
        )
        stats["unique_patterns"] = len(self._pattern_detector._patterns)
        return stats


class SmartAggregator:
    """
    High-level aggregator combining pattern detection and time-based aggregation.
    """

    __slots__ = (
        "_aggregator",
        "_enabled",
        "_auto_aggregate_threshold",
        "_last_cleanup",
    )

    def __init__(
        self,
        window_seconds: int = 60,
        similarity_threshold: float = 0.8,
        max_unique_patterns: int = 100,
        auto_aggregate_threshold: int = 10,
    ):
        """
        Initialize smart aggregator.

        Args:
            window_seconds: Aggregation window size
            similarity_threshold: Pattern similarity threshold
            max_unique_patterns: Maximum patterns per window
            auto_aggregate_threshold: Minimum events for auto-aggregation
        """
        self._aggregator = EventAggregator(
            window_seconds=window_seconds,
            max_unique_patterns=max_unique_patterns,
            similarity_threshold=similarity_threshold,
        )
        self._enabled = True
        self._auto_aggregate_threshold = auto_aggregate_threshold
        self._last_cleanup = time.time()

    def should_aggregate(self, event: Dict[str, Any]) -> bool:
        """
        Determine if event should be aggregated.

        Args:
            event: Event to check

        Returns:
            True if event should be aggregated
        """
        if not self._enabled:
            return False

        if event.get("level") in ["ERROR", "CRITICAL"]:
            return False

        if event.get("aggregate", True) is False:
            return False

        message = self._extract_message(event)
        if not message or len(message) < 20:
            return False

        return True

    def process_event(self, event: Dict[str, Any]) -> Optional[AggregatedEvent]:
        """
        Process event and return aggregated version if applicable.

        Args:
            event: Event to process

        Returns:
            Aggregated event if threshold met, None otherwise
        """
        if not self.should_aggregate(event):
            return None

        pattern_id = self._aggregator.add_event(event)
        if not pattern_id:
            return None

        if time.time() - self._last_cleanup > 300:
            self._aggregator.cleanup_old_windows()
            self._last_cleanup = time.time()

        window_key = self._aggregator._get_window_key(
            event.get("timestamp", time.time())
        )
        window = self._aggregator._active_windows.get(window_key, {})
        pattern_data = window.get(pattern_id, {})

        if pattern_data.get("count", 0) >= self._auto_aggregate_threshold:
            aggregated_events = self._aggregator.get_aggregated_events(window_key)
            for agg_event in aggregated_events:
                if agg_event.pattern_id == pattern_id:
                    return agg_event

        return None

    def _extract_message(self, event: Dict[str, Any]) -> Optional[str]:
        """Extract message from event."""
        return self._aggregator._extract_message(event)

    def enable(self) -> None:
        """Enable aggregation."""
        self._enabled = True

    def disable(self) -> None:
        """Disable aggregation."""
        self._enabled = False

    def get_aggregated_events(self) -> List[AggregatedEvent]:
        """Get all aggregated events from current window."""
        return self._aggregator.get_aggregated_events()

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        stats = self._aggregator.get_statistics()
        stats["enabled"] = self._enabled
        stats["auto_threshold"] = self._auto_aggregate_threshold
        return stats
