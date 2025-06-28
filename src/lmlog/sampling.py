"""
Adaptive sampling strategies for intelligent log volume management.
"""

import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Optional, Protocol, Callable
import random
from enum import Enum


class LogLevel(Enum):
    """Log levels for sampling decisions."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class SamplingDecision:
    """
    Result of a sampling decision.
    """

    __slots__ = ("should_sample", "sample_rate", "reason")

    def __init__(self, should_sample: bool, sample_rate: float, reason: str):
        """
        Initialize sampling decision.

        Args:
            should_sample: Whether to sample this event
            sample_rate: Rate used for decision
            reason: Reason for the decision
        """
        self.should_sample = should_sample
        self.sample_rate = sample_rate
        self.reason = reason


class SamplingContext(Protocol):
    """Protocol for sampling context information."""

    def get_level(self) -> LogLevel:
        """Get the log level."""
        ...

    def get_event_type(self) -> str:
        """Get the event type."""
        ...

    def get_context(self) -> Dict[str, Any]:
        """Get additional context."""
        ...


class Sampler(ABC):
    """
    Base class for sampling strategies.
    """

    @abstractmethod
    def should_sample(self, context: SamplingContext) -> SamplingDecision:
        """
        Determine if an event should be sampled.

        Args:
            context: Context information for sampling decision

        Returns:
            Sampling decision
        """
        pass


class AlwaysSampler(Sampler):
    """
    Sampler that always samples all events.
    """

    def should_sample(self, context: SamplingContext) -> SamplingDecision:
        """Always sample."""
        return SamplingDecision(True, 1.0, "always_sample")


class NeverSampler(Sampler):
    """
    Sampler that never samples any events.
    """

    def should_sample(self, context: SamplingContext) -> SamplingDecision:
        """Never sample."""
        return SamplingDecision(False, 0.0, "never_sample")


class ProbabilisticSampler(Sampler):
    """
    Sampler that uses fixed probability.
    """

    __slots__ = ("_probability",)

    def __init__(self, probability: float):
        """
        Initialize probabilistic sampler.

        Args:
            probability: Sampling probability (0.0 to 1.0)
        """
        self._probability = max(0.0, min(1.0, probability))

    def should_sample(self, context: SamplingContext) -> SamplingDecision:
        """Sample based on probability."""
        should_sample = random.random() < self._probability
        return SamplingDecision(
            should_sample, self._probability, f"probabilistic({self._probability})"
        )


class RateLimitingSampler(Sampler):
    """
    Sampler that limits events per second.
    """

    __slots__ = ("_max_rate", "_window_size", "_events", "_lock")

    def __init__(self, max_events_per_second: int, window_size: float = 1.0):
        """
        Initialize rate limiting sampler.

        Args:
            max_events_per_second: Maximum events per second
            window_size: Time window in seconds
        """
        self._max_rate = max_events_per_second
        self._window_size = window_size
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def should_sample(self, context: SamplingContext) -> SamplingDecision:
        """Sample based on rate limit."""
        now = time.time()

        with self._lock:
            while self._events and self._events[0] < now - self._window_size:
                self._events.popleft()

            if len(self._events) < self._max_rate:
                self._events.append(now)
                current_rate = len(self._events) / self._window_size
                return SamplingDecision(
                    True,
                    current_rate / self._max_rate,
                    f"rate_limit({len(self._events)}/{self._max_rate})",
                )

        return SamplingDecision(False, 0.0, "rate_limit_exceeded")


class LevelBasedSampler(Sampler):
    """
    Sampler that uses different rates for different log levels.
    """

    __slots__ = ("_level_rates",)

    def __init__(self, level_rates: Optional[Dict[LogLevel, float]] = None):
        """
        Initialize level-based sampler.

        Args:
            level_rates: Sampling rates per log level
        """
        self._level_rates = level_rates or {
            LogLevel.DEBUG: 0.1,
            LogLevel.INFO: 0.5,
            LogLevel.WARNING: 0.8,
            LogLevel.ERROR: 1.0,
            LogLevel.CRITICAL: 1.0,
        }

    def should_sample(self, context: SamplingContext) -> SamplingDecision:
        """Sample based on log level."""
        level = context.get_level()
        rate = self._level_rates.get(level, 1.0)

        should_sample = random.random() < rate
        return SamplingDecision(
            should_sample, rate, f"level_based({level.name}={rate})"
        )


class AdaptiveSampler(Sampler):
    """
    Sampler that adapts based on current load and system state.
    """

    __slots__ = (
        "_target_rate",
        "_window_size",
        "_events",
        "_current_probability",
        "_adjustment_factor",
        "_min_probability",
        "_max_probability",
        "_lock",
        "_last_adjustment",
    )

    def __init__(
        self,
        target_events_per_second: int = 1000,
        window_size: float = 10.0,
        adjustment_factor: float = 0.1,
        min_probability: float = 0.001,
        max_probability: float = 1.0,
    ):
        """
        Initialize adaptive sampler.

        Args:
            target_events_per_second: Target events per second
            window_size: Adjustment window in seconds
            adjustment_factor: How aggressively to adjust (0.0 to 1.0)
            min_probability: Minimum sampling probability
            max_probability: Maximum sampling probability
        """
        self._target_rate = target_events_per_second
        self._window_size = window_size
        self._events: deque[float] = deque()
        self._current_probability = 1.0
        self._adjustment_factor = adjustment_factor
        self._min_probability = min_probability
        self._max_probability = max_probability
        self._lock = threading.Lock()
        self._last_adjustment = time.time()

    def should_sample(self, context: SamplingContext) -> SamplingDecision:
        """Sample with adaptive probability."""
        now = time.time()

        with self._lock:
            while self._events and self._events[0] < now - self._window_size:
                self._events.popleft()

            current_rate = len(self._events) / self._window_size

            if now - self._last_adjustment > 1.0:
                self._adjust_probability(current_rate)
                self._last_adjustment = now

            should_sample = random.random() < self._current_probability

            if should_sample:
                self._events.append(now)

            return SamplingDecision(
                should_sample,
                self._current_probability,
                f"adaptive(rate={current_rate:.1f}, prob={self._current_probability:.3f})",
            )

    def _adjust_probability(self, current_rate: float) -> None:
        """Adjust sampling probability based on current rate."""
        if current_rate > self._target_rate * 1.1:
            self._current_probability *= 1.0 - self._adjustment_factor
        elif current_rate < self._target_rate * 0.9:
            self._current_probability *= 1.0 + self._adjustment_factor

        self._current_probability = max(
            self._min_probability, min(self._max_probability, self._current_probability)
        )

    def get_current_probability(self) -> float:
        """Get current sampling probability."""
        return self._current_probability


class CompositeSampler(Sampler):
    """
    Sampler that combines multiple sampling strategies.
    """

    __slots__ = ("_samplers",)

    def __init__(self, samplers: list[Sampler]):
        """
        Initialize composite sampler.

        Args:
            samplers: List of samplers to combine (AND logic)
        """
        self._samplers = samplers

    def should_sample(self, context: SamplingContext) -> SamplingDecision:
        """Sample only if all samplers agree."""
        reasons = []
        min_rate = 1.0

        for sampler in self._samplers:
            decision = sampler.should_sample(context)
            reasons.append(decision.reason)
            min_rate = min(min_rate, decision.sample_rate)

            if not decision.should_sample:
                return SamplingDecision(
                    False, min_rate, f"composite_reject({' AND '.join(reasons)})"
                )

        return SamplingDecision(
            True, min_rate, f"composite_accept({' AND '.join(reasons)})"
        )


class ContextBasedSampler(Sampler):
    """
    Sampler that makes decisions based on event context.
    """

    __slots__ = ("_rules", "_default_sampler")

    def __init__(self, default_sampler: Optional[Sampler] = None):
        """
        Initialize context-based sampler.

        Args:
            default_sampler: Default sampler when no rules match
        """
        self._rules: list[tuple[Callable[[SamplingContext], bool], Sampler]] = []
        self._default_sampler = default_sampler or ProbabilisticSampler(1.0)

    def add_rule(
        self, condition: Callable[[SamplingContext], bool], sampler: Sampler
    ) -> None:
        """
        Add a sampling rule.

        Args:
            condition: Function that returns True if rule should apply
            sampler: Sampler to use when condition is met
        """
        self._rules.append((condition, sampler))

    def should_sample(self, context: SamplingContext) -> SamplingDecision:
        """Sample based on context rules."""
        for condition, sampler in self._rules:
            if condition(context):
                decision = sampler.should_sample(context)
                return SamplingDecision(
                    decision.should_sample,
                    decision.sample_rate,
                    f"context_rule({decision.reason})",
                )

        decision = self._default_sampler.should_sample(context)
        return SamplingDecision(
            decision.should_sample, decision.sample_rate, f"default({decision.reason})"
        )


def create_smart_sampler(
    target_rate: int = 1000, level_rates: Optional[Dict[LogLevel, float]] = None
) -> CompositeSampler:
    """
    Create a smart sampler with adaptive rate limiting and level-based sampling.

    Args:
        target_rate: Target events per second
        level_rates: Custom rates per log level

    Returns:
        Configured composite sampler
    """
    adaptive = AdaptiveSampler(target_events_per_second=target_rate)
    level_based = LevelBasedSampler(level_rates)

    return CompositeSampler([adaptive, level_based])
