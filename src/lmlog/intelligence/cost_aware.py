"""
Cost-aware logging controls for budget management.
"""

import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from collections import deque


class CostTier(Enum):
    """Cost tiers for different log retention periods."""

    HOT = "hot"      # 7 days, fast access
    WARM = "warm"    # 30 days, medium access
    COLD = "cold"    # 90 days, slow access
    ARCHIVE = "archive"  # 365+ days, very slow access


@dataclass
class CostBudget:
    """Simple budget configuration for cost control."""

    max_events_per_second: int
    max_daily_events: int
    alert_threshold: float  # 0.0 to 1.0
    enable_throttling: bool = True
    enable_alerts: bool = True


@dataclass
class CostMetrics:
    """Current cost and usage metrics."""

    events_logged_today: int
    current_events_per_second: float
    budget_utilization: float
    throttling_active: bool
    last_reset: float


class CostController:
    """
    Simple cost controller for budget-aware logging.
    
    Provides basic rate limiting and budget tracking without
    complex ML or prediction algorithms.
    """

    __slots__ = (
        "_budget",
        "_event_counts",
        "_rate_tracker",
        "_daily_count",
        "_daily_reset_time",
        "_alert_callbacks",
        "_stats",
    )

    def __init__(self, budget: CostBudget):
        """
        Initialize simple cost controller.

        Args:
            budget: Budget configuration
        """
        self._budget = budget
        self._event_counts = deque(maxlen=60)  # Track events per second for last minute
        self._rate_tracker = deque(maxlen=10)  # Track 10-second averages
        self._daily_count = 0
        self._daily_reset_time = self._get_daily_reset_time()
        self._alert_callbacks: List[Callable[[CostMetrics], None]] = []
        self._stats = {
            "events_throttled": 0,
            "alerts_sent": 0,
            "budget_resets": 0,
        }

    def should_log_event(self, event: Dict[str, Any], priority_level: int = 3) -> bool:
        """
        Determine if an event should be logged based on budget constraints.

        Args:
            event: Log event to check
            priority_level: Event priority (1=trivial, 5=critical)

        Returns:
            True if event should be logged, False if throttled
        """
        current_time = time.time()
        
        # Reset daily counters if needed
        if current_time >= self._daily_reset_time:
            self._reset_daily_counters()

        # Always log critical events (priority 5)
        if priority_level >= 5:
            self._record_event()
            return True

        # Check daily budget
        if self._daily_count >= self._budget.max_daily_events:
            if self._budget.enable_throttling:
                self._stats["events_throttled"] += 1
                # Only log high priority events when over daily budget
                if priority_level >= 4:
                    self._record_event()
                    return True
                return False

        # Check rate limits
        current_rate = self._calculate_current_rate()
        if current_rate > self._budget.max_events_per_second:
            if self._budget.enable_throttling:
                self._stats["events_throttled"] += 1
                # Apply priority-based throttling
                if priority_level >= 4:
                    self._record_event()
                    return True
                elif priority_level >= 3 and current_rate < self._budget.max_events_per_second * 1.5:
                    self._record_event()
                    return True
                return False

        # Check budget utilization for alerts
        utilization = self._daily_count / self._budget.max_daily_events
        if (utilization >= self._budget.alert_threshold and 
            self._budget.enable_alerts and 
            len(self._alert_callbacks) > 0):
            self._send_budget_alert(utilization)

        # Log the event
        self._record_event()
        return True

    def get_cost_tier(self, event: Dict[str, Any], priority_level: int = 3) -> CostTier:
        """
        Determine appropriate cost tier for event storage.

        Args:
            event: Log event
            priority_level: Event priority level

        Returns:
            Appropriate cost tier for storage
        """
        # Critical and high priority events go to HOT tier
        if priority_level >= 4:
            return CostTier.HOT

        # Error events go to WARM tier for medium-term access
        level = event.get("level", "").upper()
        if level in ["ERROR", "FATAL"]:
            return CostTier.WARM

        # Security and audit events go to WARM tier for compliance
        event_type = event.get("event_type", "").lower()
        if any(keyword in event_type for keyword in ["security", "audit", "auth"]):
            return CostTier.WARM

        # Performance events go to COLD tier
        if "performance" in event_type or level == "WARNING":
            return CostTier.COLD

        # Everything else goes to ARCHIVE tier
        return CostTier.ARCHIVE

    def get_metrics(self) -> CostMetrics:
        """Get current cost metrics."""
        current_rate = self._calculate_current_rate()
        utilization = self._daily_count / max(1, self._budget.max_daily_events)
        throttling_active = (
            current_rate > self._budget.max_events_per_second or
            self._daily_count >= self._budget.max_daily_events
        )

        return CostMetrics(
            events_logged_today=self._daily_count,
            current_events_per_second=current_rate,
            budget_utilization=utilization,
            throttling_active=throttling_active,
            last_reset=self._daily_reset_time,
        )

    def add_alert_callback(self, callback: Callable[[CostMetrics], None]) -> None:
        """Add callback function for budget alerts."""
        self._alert_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cost controller statistics."""
        metrics = self.get_metrics()
        return {
            "events_throttled": self._stats["events_throttled"],
            "alerts_sent": self._stats["alerts_sent"],
            "budget_resets": self._stats["budget_resets"],
            "current_daily_count": self._daily_count,
            "budget_utilization": metrics.budget_utilization,
            "current_rate": metrics.current_events_per_second,
            "throttling_active": metrics.throttling_active,
        }

    def _record_event(self) -> None:
        """Record that an event was logged."""
        current_time = time.time()
        self._event_counts.append(current_time)
        self._daily_count += 1

    def _calculate_current_rate(self) -> float:
        """Calculate current events per second rate."""
        if not self._event_counts:
            return 0.0

        current_time = time.time()
        
        # Remove events older than 1 second
        while self._event_counts and current_time - self._event_counts[0] > 1.0:
            self._event_counts.popleft()

        return len(self._event_counts)

    def _reset_daily_counters(self) -> None:
        """Reset daily counters at midnight."""
        self._daily_count = 0
        self._daily_reset_time = self._get_daily_reset_time()
        self._stats["budget_resets"] += 1

    def _get_daily_reset_time(self) -> float:
        """Get timestamp for next daily reset (midnight)."""
        import datetime
        
        # Get current date and set time to midnight tomorrow
        now = datetime.datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        return tomorrow.timestamp()

    def _send_budget_alert(self, utilization: float) -> None:
        """Send budget utilization alert."""
        metrics = self.get_metrics()
        
        for callback in self._alert_callbacks:
            callback(metrics)
        
        self._stats["alerts_sent"] += 1


class Throttler:
    """
    Simple throttling mechanism for rate limiting.
    
    Uses token bucket algorithm for smooth rate limiting.
    """

    __slots__ = (
        "_max_rate",
        "_bucket_size",
        "_tokens",
        "_last_refill",
    )

    def __init__(self, max_rate: float, bucket_size: Optional[int] = None):
        """
        Initialize throttler.

        Args:
            max_rate: Maximum events per second
            bucket_size: Token bucket size (defaults to max_rate * 2)
        """
        self._max_rate = max_rate
        self._bucket_size = bucket_size or int(max_rate * 2)
        self._tokens = self._bucket_size
        self._last_refill = time.time()

    def should_allow(self, tokens_requested: int = 1) -> bool:
        """
        Check if request should be allowed.

        Args:
            tokens_requested: Number of tokens to consume

        Returns:
            True if request is allowed, False if throttled
        """
        current_time = time.time()
        
        # Refill tokens based on elapsed time
        elapsed = current_time - self._last_refill
        tokens_to_add = elapsed * self._max_rate
        self._tokens = min(self._bucket_size, self._tokens + tokens_to_add)
        self._last_refill = current_time

        # Check if we have enough tokens
        if self._tokens >= tokens_requested:
            self._tokens -= tokens_requested
            return True

        return False

    def get_available_tokens(self) -> float:
        """Get number of available tokens."""
        current_time = time.time()
        elapsed = current_time - self._last_refill
        tokens_to_add = elapsed * self._max_rate
        return min(self._bucket_size, self._tokens + tokens_to_add)


class CostSampler:
    """
    Simple sampling mechanism for reducing log volume.
    
    Uses deterministic sampling based on hash values for consistency.
    """

    __slots__ = ("_rate", "_counter", "_threshold")

    def __init__(self, sampling_rate: float):
        """
        Initialize sampler.

        Args:
            sampling_rate: Rate to sample (0.0 to 1.0)
        """
        self._rate = max(0.0, min(1.0, sampling_rate))
        self._counter = 0
        self._threshold = int(1.0 / self._rate) if self._rate > 0 else 0

    def should_sample(self, event: Dict[str, Any]) -> bool:
        """
        Determine if event should be sampled.

        Args:
            event: Event to check

        Returns:
            True if event should be included in sample
        """
        if self._rate >= 1.0:
            return True
        
        if self._rate <= 0.0:
            return False

        # Use deterministic sampling based on message hash
        message = event.get("message", "")
        event_hash = hash(message) % 100
        
        return event_hash < (self._rate * 100)

    def update_rate(self, new_rate: float) -> None:
        """Update sampling rate."""
        self._rate = max(0.0, min(1.0, new_rate))
        self._threshold = int(1.0 / self._rate) if self._rate > 0 else 0