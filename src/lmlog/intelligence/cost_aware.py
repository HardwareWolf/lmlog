"""
Cost-aware logging system with intelligent optimization and budgeting.
"""

import gzip
import io
import json
import time
import secrets
import zlib
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class StorageTier(Enum):
    """Storage tier levels for cost optimization."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVED = "archived"


class CompressionLevel(Enum):
    """Compression levels for storage optimization."""

    NONE = 0
    FAST = 1
    BALANCED = 6
    MAXIMUM = 9


@dataclass
class CostMetrics:
    """Metrics for cost tracking."""

    bytes_written: int = 0
    events_written: int = 0
    bytes_compressed: int = 0
    compression_ratio: float = 0.0
    estimated_cost: float = 0.0
    sampling_adjustments: int = 0
    dropped_events: int = 0


@dataclass
class StoragePolicy:
    """Policy for data storage and retention."""

    tier: StorageTier
    retention_days: int
    compression: CompressionLevel
    sampling_rate: float = 1.0
    max_event_size: int = 10240


@dataclass
class CostBudget:
    """Cost budget configuration."""

    max_daily_bytes: int
    max_events_per_second: int
    cost_per_gb: float = 0.23
    enable_auto_scaling: bool = True
    alert_threshold: float = 0.8


class CostCalculator:
    """Calculate logging costs based on volume and storage."""

    __slots__ = (
        "_cost_per_gb",
        "_compression_factors",
        "_tier_multipliers",
    )

    def __init__(self, base_cost_per_gb: float = 0.23):
        """
        Initialize cost calculator.

        Args:
            base_cost_per_gb: Base cost per GB of storage
        """
        self._cost_per_gb = base_cost_per_gb

        self._compression_factors = {
            CompressionLevel.NONE: 1.0,
            CompressionLevel.FAST: 0.7,
            CompressionLevel.BALANCED: 0.4,
            CompressionLevel.MAXIMUM: 0.3,
        }

        self._tier_multipliers = {
            StorageTier.HOT: 1.0,
            StorageTier.WARM: 0.5,
            StorageTier.COLD: 0.2,
            StorageTier.ARCHIVED: 0.05,
        }

    def calculate_storage_cost(
        self,
        bytes_size: int,
        tier: StorageTier,
        compression: CompressionLevel,
        retention_days: int,
    ) -> float:
        """
        Calculate storage cost for data.

        Args:
            bytes_size: Size in bytes
            tier: Storage tier
            compression: Compression level
            retention_days: Days to retain data

        Returns:
            Estimated cost
        """
        gb_size = bytes_size / (1024**3)

        compressed_size = gb_size * self._compression_factors.get(compression, 1.0)

        tier_multiplier = self._tier_multipliers.get(tier, 1.0)

        daily_cost = compressed_size * self._cost_per_gb * tier_multiplier

        total_cost = daily_cost * retention_days

        return total_cost

    def calculate_transfer_cost(
        self,
        bytes_size: int,
        regions: int = 1,
    ) -> float:
        """
        Calculate data transfer cost.

        Args:
            bytes_size: Size in bytes
            regions: Number of regions to replicate to

        Returns:
            Transfer cost
        """
        gb_size = bytes_size / (1024**3)

        transfer_cost_per_gb = 0.09

        return gb_size * transfer_cost_per_gb * (regions - 1)

    def estimate_monthly_cost(
        self,
        daily_volume_bytes: int,
        storage_policy: StoragePolicy,
        transfer_regions: int = 1,
    ) -> Dict[str, float]:
        """
        Estimate monthly costs.

        Args:
            daily_volume_bytes: Daily data volume
            storage_policy: Storage policy
            transfer_regions: Number of regions

        Returns:
            Cost breakdown
        """
        monthly_bytes = daily_volume_bytes * 30

        storage_cost = self.calculate_storage_cost(
            monthly_bytes,
            storage_policy.tier,
            storage_policy.compression,
            storage_policy.retention_days,
        )

        transfer_cost = self.calculate_transfer_cost(
            monthly_bytes,
            transfer_regions,
        )

        return {
            "storage": storage_cost,
            "transfer": transfer_cost,
            "total": storage_cost + transfer_cost,
            "daily_average": (storage_cost + transfer_cost) / 30,
        }


class VolumeTracker:
    """Track logging volume over time."""

    __slots__ = (
        "_window_size",
        "_byte_windows",
        "_event_windows",
        "_current_bytes",
        "_current_events",
        "_window_start",
    )

    def __init__(self, window_size: int = 60):
        """
        Initialize volume tracker.

        Args:
            window_size: Window size in seconds
        """
        self._window_size = window_size
        self._byte_windows: deque = deque(maxlen=1440)
        self._event_windows: deque = deque(maxlen=1440)
        self._current_bytes = 0
        self._current_events = 0
        self._window_start = time.time()

    def track_event(self, size_bytes: int) -> None:
        """
        Track a log event.

        Args:
            size_bytes: Event size in bytes
        """
        self._maybe_rotate_window()

        self._current_bytes += size_bytes
        self._current_events += 1

    def get_current_rate(self) -> Tuple[float, float]:
        """
        Get current logging rate.

        Returns:
            Tuple of (bytes/sec, events/sec)
        """
        elapsed = time.time() - self._window_start
        if elapsed == 0:
            return 0.0, 0.0

        bytes_per_sec = self._current_bytes / elapsed
        events_per_sec = self._current_events / elapsed

        return bytes_per_sec, events_per_sec

    def get_daily_projection(self) -> Tuple[int, int]:
        """
        Get projected daily volume.

        Returns:
            Tuple of (bytes, events)
        """
        if not self._byte_windows:
            bytes_per_sec, events_per_sec = self.get_current_rate()
        else:
            total_bytes = sum(self._byte_windows) + self._current_bytes
            total_events = sum(self._event_windows) + self._current_events
            total_windows = len(self._byte_windows) + 1

            bytes_per_window = total_bytes / total_windows
            events_per_window = total_events / total_windows

            bytes_per_sec = bytes_per_window / self._window_size
            events_per_sec = events_per_window / self._window_size

        daily_bytes = int(bytes_per_sec * 86400)
        daily_events = int(events_per_sec * 86400)

        return daily_bytes, daily_events

    def _maybe_rotate_window(self) -> None:
        """Rotate window if needed."""
        current_time = time.time()

        if current_time - self._window_start >= self._window_size:
            self._byte_windows.append(self._current_bytes)
            self._event_windows.append(self._current_events)

            self._current_bytes = 0
            self._current_events = 0
            self._window_start = current_time


class DataCompressor:
    """Compress log data for storage optimization."""

    __slots__ = ("_compression_cache", "_cache_size")

    def __init__(self, cache_size: int = 100):
        """
        Initialize data compressor.

        Args:
            cache_size: Size of compression cache
        """
        self._compression_cache: Dict[int, bytes] = {}
        self._cache_size = cache_size

    def compress(
        self,
        data: bytes,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> Tuple[bytes, float]:
        """
        Compress data.

        Args:
            data: Data to compress
            level: Compression level

        Returns:
            Tuple of (compressed data, compression ratio)
        """
        if level == CompressionLevel.NONE:
            return data, 1.0

        data_hash = hash(data)
        cached = self._compression_cache.get(data_hash)
        if cached:
            return cached, len(cached) / len(data)

        compressed = zlib.compress(data, level.value)

        ratio = len(compressed) / len(data)

        if len(self._compression_cache) >= self._cache_size:
            self._compression_cache.pop(next(iter(self._compression_cache)))

        self._compression_cache[data_hash] = compressed

        return compressed, ratio

    def compress_batch(
        self,
        events: List[Dict[str, Any]],
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> Tuple[bytes, float]:
        """
        Compress a batch of events.

        Args:
            events: Events to compress
            level: Compression level

        Returns:
            Tuple of (compressed data, compression ratio)
        """
        buffer = io.BytesIO()

        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=level.value) as gz:
            for event in events:
                line = json.dumps(event, separators=(",", ":")) + "\n"
                gz.write(line.encode("utf-8"))

        compressed = buffer.getvalue()

        original_size = sum(
            len(json.dumps(event, separators=(",", ":"))) + 1 for event in events
        )

        ratio = len(compressed) / original_size if original_size > 0 else 1.0

        return compressed, ratio


class AdaptiveSamplingController:
    """Control sampling rates based on cost constraints."""

    __slots__ = (
        "_target_rate",
        "_min_sampling",
        "_max_sampling",
        "_adjustment_factor",
        "_rate_history",
        "_last_adjustment",
        "_current_sampling",
    )

    def __init__(
        self,
        target_bytes_per_sec: float,
        min_sampling: float = 0.01,
        max_sampling: float = 1.0,
        adjustment_factor: float = 0.1,
    ):
        """
        Initialize adaptive sampling controller.

        Args:
            target_bytes_per_sec: Target data rate
            min_sampling: Minimum sampling rate
            max_sampling: Maximum sampling rate
            adjustment_factor: Rate adjustment factor
        """
        self._target_rate = target_bytes_per_sec
        self._min_sampling = min_sampling
        self._max_sampling = max_sampling
        self._adjustment_factor = adjustment_factor
        self._rate_history: deque = deque(maxlen=10)
        self._last_adjustment = time.time()
        self._current_sampling = 1.0

    def get_sampling_rate(
        self,
        current_rate: float,
        event_priority: float = 0.5,
    ) -> float:
        """
        Get adjusted sampling rate.

        Args:
            current_rate: Current data rate
            event_priority: Event priority (0-1)

        Returns:
            Sampling rate
        """
        self._rate_history.append(current_rate)

        if time.time() - self._last_adjustment < 5.0:
            return self._get_current_sampling()

        avg_rate = sum(self._rate_history) / len(self._rate_history)

        if avg_rate > self._target_rate * 1.1:
            adjustment = 1.0 - self._adjustment_factor
        elif avg_rate < self._target_rate * 0.9:
            adjustment = 1.0 + self._adjustment_factor
        else:
            adjustment = 1.0

        new_sampling = self._get_current_sampling() * adjustment

        new_sampling = max(self._min_sampling, min(self._max_sampling, new_sampling))

        self._current_sampling = new_sampling

        priority_adjusted = new_sampling * (0.5 + event_priority)

        self._last_adjustment = time.time()

        return min(1.0, max(0.001, priority_adjusted))

    def _get_current_sampling(self) -> float:
        """Get current base sampling rate."""
        return self._current_sampling


class TierManager:
    """Manage data tiering for cost optimization."""

    __slots__ = (
        "_policies",
        "_age_thresholds",
        "_size_thresholds",
    )

    def __init__(self):
        """Initialize tier manager."""
        self._policies = {
            StorageTier.HOT: StoragePolicy(
                tier=StorageTier.HOT,
                retention_days=7,
                compression=CompressionLevel.FAST,
                sampling_rate=1.0,
            ),
            StorageTier.WARM: StoragePolicy(
                tier=StorageTier.WARM,
                retention_days=30,
                compression=CompressionLevel.BALANCED,
                sampling_rate=0.5,
            ),
            StorageTier.COLD: StoragePolicy(
                tier=StorageTier.COLD,
                retention_days=90,
                compression=CompressionLevel.MAXIMUM,
                sampling_rate=0.1,
            ),
            StorageTier.ARCHIVED: StoragePolicy(
                tier=StorageTier.ARCHIVED,
                retention_days=365,
                compression=CompressionLevel.MAXIMUM,
                sampling_rate=0.01,
            ),
        }

        self._age_thresholds = {
            StorageTier.HOT: 0,
            StorageTier.WARM: 7,
            StorageTier.COLD: 30,
            StorageTier.ARCHIVED: 90,
        }

        self._size_thresholds = {
            StorageTier.HOT: 0,
            StorageTier.WARM: 1024 * 1024 * 100,
            StorageTier.COLD: 1024 * 1024 * 1024,
            StorageTier.ARCHIVED: 1024 * 1024 * 1024 * 10,
        }

    def get_tier_for_data(
        self,
        age_days: float,
        size_bytes: int,
        access_frequency: float = 0.0,
    ) -> StorageTier:
        """
        Determine appropriate tier for data.

        Args:
            age_days: Age of data in days
            size_bytes: Size of data
            access_frequency: Access frequency (0-1)

        Returns:
            Appropriate storage tier
        """
        if access_frequency > 0.8:
            return StorageTier.HOT

        for tier in [
            StorageTier.ARCHIVED,
            StorageTier.COLD,
            StorageTier.WARM,
            StorageTier.HOT,
        ]:
            if age_days >= self._age_thresholds[tier]:
                return tier

        return StorageTier.HOT

    def get_policy(self, tier: StorageTier) -> StoragePolicy:
        """Get storage policy for tier."""
        return self._policies[tier]


class CostAwareLogger:
    """
    Cost-aware logging manager with automatic optimization.
    """

    __slots__ = (
        "_budget",
        "_calculator",
        "_volume_tracker",
        "_compressor",
        "_sampling_controller",
        "_tier_manager",
        "_metrics",
        "_callbacks",
        "_enabled",
    )

    def __init__(
        self,
        budget: CostBudget,
        cost_per_gb: float = 0.23,
    ):
        """
        Initialize cost-aware logger.

        Args:
            budget: Cost budget configuration
            cost_per_gb: Cost per GB of storage
        """
        self._budget = budget
        self._calculator = CostCalculator(cost_per_gb)
        self._volume_tracker = VolumeTracker()
        self._compressor = DataCompressor()
        self._sampling_controller = AdaptiveSamplingController(
            target_bytes_per_sec=float(budget.max_daily_bytes) / 86400
        )
        self._tier_manager = TierManager()
        self._metrics = CostMetrics()
        self._callbacks: List[Callable[[CostMetrics], None]] = []
        self._enabled = True

    def should_log(
        self,
        event: Dict[str, Any],
        priority: float = 0.5,
    ) -> bool:
        """
        Determine if event should be logged based on cost.

        Args:
            event: Event to check
            priority: Event priority (0-1)

        Returns:
            True if event should be logged
        """
        if not self._enabled:
            return True

        if priority >= 0.9:
            return True

        current_rate, _ = self._volume_tracker.get_current_rate()
        sampling_rate = self._sampling_controller.get_sampling_rate(
            current_rate,
            priority,
        )

        if sampling_rate >= 1.0:
            return True

        return secrets.randbelow(1000000) < int(sampling_rate * 1000000)

    def process_event(
        self,
        event: Dict[str, Any],
        tier: Optional[StorageTier] = None,
    ) -> Tuple[bytes, StoragePolicy]:
        """
        Process event for cost-optimized storage.

        Args:
            event: Event to process
            tier: Storage tier (auto-detected if None)

        Returns:
            Tuple of (processed data, storage policy)
        """
        event_str = json.dumps(event, separators=(",", ":"))
        event_bytes = event_str.encode("utf-8")

        self._volume_tracker.track_event(len(event_bytes))

        if tier is None:
            tier = StorageTier.HOT

        policy = self._tier_manager.get_policy(tier)

        compressed_data, ratio = self._compressor.compress(
            event_bytes,
            policy.compression,
        )

        self._metrics.bytes_written += len(event_bytes)
        self._metrics.events_written += 1
        self._metrics.bytes_compressed += len(compressed_data)
        self._metrics.compression_ratio = ratio

        daily_bytes, _ = self._volume_tracker.get_daily_projection()
        self._metrics.estimated_cost = self._calculator.calculate_storage_cost(
            daily_bytes,
            tier,
            policy.compression,
            policy.retention_days,
        )

        self._check_budget_alerts()

        return compressed_data, policy

    def get_cost_metrics(self) -> CostMetrics:
        """Get current cost metrics."""
        return self._metrics

    def get_cost_forecast(self) -> Dict[str, Any]:
        """Get cost forecast."""
        daily_bytes, daily_events = self._volume_tracker.get_daily_projection()

        hot_policy = self._tier_manager.get_policy(StorageTier.HOT)
        warm_policy = self._tier_manager.get_policy(StorageTier.WARM)

        monthly_hot = self._calculator.estimate_monthly_cost(
            daily_bytes,
            hot_policy,
        )

        monthly_warm = self._calculator.estimate_monthly_cost(
            int(daily_bytes * 0.3),
            warm_policy,
        )

        return {
            "daily_volume_gb": daily_bytes / (1024**3),
            "daily_events": daily_events,
            "monthly_cost_hot": monthly_hot["total"],
            "monthly_cost_tiered": monthly_hot["total"] * 0.7
            + monthly_warm["total"] * 0.3,
            "budget_usage": daily_bytes / self._budget.max_daily_bytes,
            "recommendations": self._get_recommendations(daily_bytes),
        }

    def add_cost_callback(self, callback: Callable[[CostMetrics], None]) -> None:
        """Add callback for cost alerts."""
        self._callbacks.append(callback)

    def _check_budget_alerts(self) -> None:
        """Check if budget thresholds are exceeded."""
        daily_bytes, _ = self._volume_tracker.get_daily_projection()

        usage_ratio = daily_bytes / self._budget.max_daily_bytes

        if usage_ratio > self._budget.alert_threshold:
            for callback in self._callbacks:
                try:
                    callback(self._metrics)
                except Exception as e:
                    import logging

                    logging.exception(f"Error in cost callback: {e}")

            if self._budget.enable_auto_scaling:
                self._metrics.sampling_adjustments += 1

    def _get_recommendations(self, daily_bytes: int) -> List[str]:
        """Get cost optimization recommendations."""
        recommendations = []

        if daily_bytes > self._budget.max_daily_bytes:
            recommendations.append(
                "Consider increasing sampling or enabling aggregation"
            )

        if self._metrics.compression_ratio > 0.5:
            recommendations.append("Consider using higher compression levels")

        if self._metrics.events_written > 1000000:
            recommendations.append(
                "High event volume detected - consider event aggregation"
            )

        return recommendations

    def enable(self) -> None:
        """Enable cost-aware features."""
        self._enabled = True

    def disable(self) -> None:
        """Disable cost-aware features."""
        self._enabled = False
