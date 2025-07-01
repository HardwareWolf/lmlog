"""
Rule-based event classification system for fast, predictable log processing.
"""

import re
import time
from typing import Dict, Any, Optional, List, Tuple, Pattern
from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True)
class EventClassification:
    """Classification result for an event."""

    event_type: EventType
    priority: EventPriority
    confidence: float
    suggested_sampling_rate: float
    matched_rules: List[str]


@dataclass(frozen=True)
class ClassificationRule:
    """A single classification rule with pattern and scoring."""

    name: str
    pattern: Pattern[str]
    event_type: EventType
    base_priority: EventPriority
    confidence: float
    weight: float
    field_targets: List[str]


class RuleBasedClassifier:
    """
    Fast rule-based event classifier using pre-compiled patterns.
    
    Provides sub-millisecond classification with predictable behavior
    and no external dependencies or training requirements.
    """

    __slots__ = (
        "_rules",
        "_level_mappings",
        "_sampling_rates",
        "_priority_boosts",
        "_stats",
    )

    def __init__(self):
        """Initialize rule-based classifier with pre-defined patterns."""
        self._rules = self._build_classification_rules()
        self._level_mappings = self._build_level_mappings()
        self._sampling_rates = self._build_sampling_rates()
        self._priority_boosts = self._build_priority_boosts()
        self._stats = {
            "classifications": 0,
            "rule_matches": Counter(),
            "type_distributions": Counter(),
        }

    def classify_event(self, event: Dict[str, Any]) -> EventClassification:
        """
        Classify an event using rule-based pattern matching.

        Args:
            event: Log event dictionary

        Returns:
            EventClassification with type, priority, and confidence
        """
        self._stats["classifications"] += 1

        # Extract key fields for analysis
        level = self._extract_level(event)
        message = self._extract_message(event)
        
        # Apply level-based classification first
        base_type, base_priority = self._classify_by_level(level)
        
        # Apply pattern rules for refinement
        matched_rules = []
        rule_scores = []
        
        for rule in self._rules:
            if self._rule_matches(rule, event, message):
                matched_rules.append(rule.name)
                rule_scores.append((rule.event_type, rule.base_priority, rule.weight, rule.confidence))
                self._stats["rule_matches"][rule.name] += 1

        # Determine final classification
        final_type, final_priority, confidence = self._resolve_classification(
            base_type, base_priority, rule_scores
        )

        # Apply priority boosts based on event characteristics
        boosted_priority = self._apply_priority_boosts(final_priority, event, message)

        # Calculate sampling rate
        sampling_rate = self._calculate_sampling_rate(final_type, boosted_priority, event)

        self._stats["type_distributions"][final_type] += 1

        return EventClassification(
            event_type=final_type,
            priority=boosted_priority,
            confidence=confidence,
            suggested_sampling_rate=sampling_rate,
            matched_rules=matched_rules,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier performance statistics."""
        return {
            "total_classifications": self._stats["classifications"],
            "rule_match_counts": dict(self._stats["rule_matches"]),
            "type_distribution": dict(self._stats["type_distributions"]),
            "rules_loaded": len(self._rules),
        }

    def _extract_level(self, event: Dict[str, Any]) -> str:
        """Extract log level from event."""
        level = event.get("level", "")
        if isinstance(level, str):
            return level.upper().strip()
        return ""

    def _extract_message(self, event: Dict[str, Any]) -> str:
        """Extract message content from event."""
        message = event.get("message", "")
        if isinstance(message, str):
            return message.lower()
        return ""

    def _classify_by_level(self, level: str) -> Tuple[EventType, EventPriority]:
        """Get base classification from log level."""
        return self._level_mappings.get(level, (EventType.UNKNOWN, EventPriority.LOW))

    def _rule_matches(self, rule: ClassificationRule, event: Dict[str, Any], message: str) -> bool:
        """Check if a rule matches the given event."""
        # Check message content first (most common case)
        if rule.pattern.search(message):
            return True

        # Check additional target fields if specified
        for field in rule.field_targets:
            field_value = event.get(field, "")
            if isinstance(field_value, str) and rule.pattern.search(field_value.lower()):
                return True

        return False

    def _resolve_classification(
        self, 
        base_type: EventType, 
        base_priority: EventPriority, 
        rule_scores: List[Tuple[EventType, EventPriority, float, float]]
    ) -> Tuple[EventType, EventPriority, float]:
        """Resolve final classification from base classification and rule matches."""
        if not rule_scores:
            return base_type, base_priority, 0.5

        # Calculate weighted scores for each event type
        type_scores = {}
        total_weight = 0

        for event_type, priority, weight, confidence in rule_scores:
            if event_type not in type_scores:
                type_scores[event_type] = {"weight": 0, "priority_sum": 0, "confidence_sum": 0, "count": 0}
            
            type_scores[event_type]["weight"] += weight
            type_scores[event_type]["priority_sum"] += priority.value * weight
            type_scores[event_type]["confidence_sum"] += confidence * weight
            type_scores[event_type]["count"] += 1
            total_weight += weight

        # Find the event type with highest weighted score
        best_type = max(type_scores.keys(), key=lambda t: type_scores[t]["weight"])
        best_scores = type_scores[best_type]

        # Calculate final priority and confidence
        avg_priority_value = best_scores["priority_sum"] / best_scores["weight"]
        final_priority = EventPriority(max(1, min(5, round(avg_priority_value))))
        
        # Confidence based on rule weight relative to total and number of matching rules
        confidence = min(1.0, (best_scores["weight"] / total_weight) * (1 + best_scores["count"] * 0.1))

        return best_type, final_priority, confidence

    def _apply_priority_boosts(
        self, 
        base_priority: EventPriority, 
        event: Dict[str, Any], 
        message: str
    ) -> EventPriority:
        """Apply priority boosts based on event characteristics."""
        priority_value = base_priority.value

        # Apply boosts from priority boost patterns
        for boost_pattern, boost_amount in self._priority_boosts.items():
            if boost_pattern.search(message):
                priority_value += boost_amount

        # Check for stack traces (high priority boost)
        if self._has_stack_trace(event):
            priority_value += 1

        # Check for user context (slight boost)
        if self._has_user_context(event):
            priority_value += 0.5

        # Check for duration/performance indicators
        duration_ms = event.get("duration_ms", 0)
        if isinstance(duration_ms, (int, float)) and duration_ms > 5000:
            priority_value += 1

        # Ensure priority stays within bounds
        final_priority_value = max(1, min(5, round(priority_value)))
        return EventPriority(final_priority_value)

    def _has_stack_trace(self, event: Dict[str, Any]) -> bool:
        """Check if event contains stack trace information."""
        stack_indicators = ["stack_trace", "stacktrace", "traceback", "exception"]
        
        for field in stack_indicators:
            if field in event and event[field]:
                return True

        # Check message for stack trace patterns
        message = self._extract_message(event)
        stack_patterns = [
            r"at\s+\w+\.\w+\(\w+\.java:\d+\)",
            r"traceback\s*\(most recent call last\)",
            r"file\s+\"[^\"]+\",\s+line\s+\d+",
            r"\w+error:\s*\w+",
        ]
        
        for pattern in stack_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True

        return False

    def _has_user_context(self, event: Dict[str, Any]) -> bool:
        """Check if event has user context information."""
        user_fields = ["user_id", "user", "username", "email", "customer_id"]
        return any(field in event and event[field] for field in user_fields)

    def _calculate_sampling_rate(
        self, 
        event_type: EventType, 
        priority: EventPriority, 
        event: Dict[str, Any]
    ) -> float:
        """Calculate suggested sampling rate based on event characteristics."""
        base_rate = self._sampling_rates[priority]

        # Critical events always get full sampling
        if priority == EventPriority.CRITICAL:
            return 1.0

        # High priority error and security events get high sampling
        if event_type in [EventType.ERROR, EventType.SECURITY] and priority.value >= 4:
            return max(base_rate, 0.8)

        # Business events get medium sampling
        if event_type == EventType.BUSINESS:
            return max(base_rate, 0.3)

        # Debug events get very low sampling unless high priority
        if event_type == EventType.DEBUG and priority.value < 3:
            return min(base_rate, 0.05)

        return base_rate

    def _build_classification_rules(self) -> List[ClassificationRule]:
        """Build the complete set of classification rules."""
        rules = []

        # Error patterns
        rules.extend([
            ClassificationRule(
                name="explicit_error",
                pattern=re.compile(r"\b(error|exception|fail|failure|fatal|crash|abort|panic)\b"),
                event_type=EventType.ERROR,
                base_priority=EventPriority.HIGH,
                confidence=0.9,
                weight=3.0,
                field_targets=["message", "error_message", "exception_type"],
            ),
            ClassificationRule(
                name="database_error",
                pattern=re.compile(r"\b(connection\s+(failed|refused|timeout)|database\s+(error|down)|sql\s+(error|exception))\b"),
                event_type=EventType.ERROR,
                base_priority=EventPriority.HIGH,
                confidence=0.85,
                weight=2.5,
                field_targets=["message"],
            ),
            ClassificationRule(
                name="http_error",
                pattern=re.compile(r"\b(50[0-9]|40[0-9])\b|\b(internal\s+server\s+error|bad\s+gateway|service\s+unavailable)\b"),
                event_type=EventType.ERROR,
                base_priority=EventPriority.MEDIUM,
                confidence=0.8,
                weight=2.0,
                field_targets=["message", "status_code"],
            ),
        ])

        # Performance patterns
        rules.extend([
            ClassificationRule(
                name="slow_operation",
                pattern=re.compile(r"\b(slow|latency|timeout|performance|duration|elapsed|delay)\b"),
                event_type=EventType.PERFORMANCE,
                base_priority=EventPriority.MEDIUM,
                confidence=0.7,
                weight=2.0,
                field_targets=["message"],
            ),
            ClassificationRule(
                name="memory_pressure",
                pattern=re.compile(r"\b(memory|heap|gc|garbage\s+collect|out\s+of\s+memory|oom)\b"),
                event_type=EventType.PERFORMANCE,
                base_priority=EventPriority.HIGH,
                confidence=0.8,
                weight=2.5,
                field_targets=["message"],
            ),
        ])

        # Security patterns
        rules.extend([
            ClassificationRule(
                name="authentication_failure",
                pattern=re.compile(r"\b(auth|login|authentication)\s+(fail|error|denied|invalid)\b"),
                event_type=EventType.SECURITY,
                base_priority=EventPriority.HIGH,
                confidence=0.9,
                weight=3.0,
                field_targets=["message"],
            ),
            ClassificationRule(
                name="permission_denied",
                pattern=re.compile(r"\b(permission|access)\s+(denied|forbidden|unauthorized)\b"),
                event_type=EventType.SECURITY,
                base_priority=EventPriority.MEDIUM,
                confidence=0.85,
                weight=2.5,
                field_targets=["message"],
            ),
            ClassificationRule(
                name="suspicious_activity",
                pattern=re.compile(r"\b(suspicious|malicious|attack|intrusion|breach|violation)\b"),
                event_type=EventType.SECURITY,
                base_priority=EventPriority.CRITICAL,
                confidence=0.95,
                weight=4.0,
                field_targets=["message"],
            ),
        ])

        # Business patterns
        rules.extend([
            ClassificationRule(
                name="payment_transaction",
                pattern=re.compile(r"\b(payment|transaction|purchase|order|billing|invoice|charge)\b"),
                event_type=EventType.BUSINESS,
                base_priority=EventPriority.MEDIUM,
                confidence=0.8,
                weight=2.0,
                field_targets=["message", "event_type"],
            ),
            ClassificationRule(
                name="user_registration",
                pattern=re.compile(r"\b(register|signup|account\s+creat|user\s+creat)\b"),
                event_type=EventType.BUSINESS,
                base_priority=EventPriority.MEDIUM,
                confidence=0.75,
                weight=1.5,
                field_targets=["message", "event_type"],
            ),
        ])

        # Audit patterns
        rules.extend([
            ClassificationRule(
                name="compliance_event",
                pattern=re.compile(r"\b(audit|compliance|regulation|policy|gdpr|sox|pci)\b"),
                event_type=EventType.AUDIT,
                base_priority=EventPriority.MEDIUM,
                confidence=0.8,
                weight=2.0,
                field_targets=["message", "event_type"],
            ),
            ClassificationRule(
                name="data_access",
                pattern=re.compile(r"\b(access|view|download|export|data\s+retriev)\b"),
                event_type=EventType.AUDIT,
                base_priority=EventPriority.LOW,
                confidence=0.6,
                weight=1.0,
                field_targets=["message", "event_type"],
            ),
        ])

        return rules

    def _build_level_mappings(self) -> Dict[str, Tuple[EventType, EventPriority]]:
        """Build mapping from log levels to default event types and priorities."""
        return {
            "FATAL": (EventType.ERROR, EventPriority.CRITICAL),
            "ERROR": (EventType.ERROR, EventPriority.HIGH),
            "WARN": (EventType.WARNING, EventPriority.MEDIUM),
            "WARNING": (EventType.WARNING, EventPriority.MEDIUM),
            "INFO": (EventType.INFO, EventPriority.LOW),
            "DEBUG": (EventType.DEBUG, EventPriority.TRIVIAL),
            "TRACE": (EventType.DEBUG, EventPriority.TRIVIAL),
        }

    def _build_sampling_rates(self) -> Dict[EventPriority, float]:
        """Build default sampling rates by priority level."""
        return {
            EventPriority.CRITICAL: 1.0,
            EventPriority.HIGH: 0.8,
            EventPriority.MEDIUM: 0.3,
            EventPriority.LOW: 0.1,
            EventPriority.TRIVIAL: 0.02,
        }

    def _build_priority_boosts(self) -> Dict[Pattern[str], float]:
        """Build patterns that boost event priority."""
        return {
            re.compile(r"\b(critical|urgent|emergency|immediate)\b"): 2.0,
            re.compile(r"\b(corruption|data\s+loss|system\s+down)\b"): 1.5,
            re.compile(r"\b(retry|attempt|retrying)\b"): 0.5,
            re.compile(r"\b(timeout|deadline|expired)\b"): 1.0,
        }


from collections import Counter