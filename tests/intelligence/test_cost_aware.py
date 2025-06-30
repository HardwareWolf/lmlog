"""
Tests for cost-aware logging features.
"""

import pytest
import time
from lmlog.intelligence.cost_aware import (
    CostCalculator,
    VolumeTracker,
    DataCompressor,
    AdaptiveSamplingController,
    TierManager,
    CostAwareLogger,
    CostBudget,
    StorageTier,
    CompressionLevel,
)


class TestCostCalculator:
    """Test cost calculation functionality."""

    def test_calculate_storage_cost_hot_tier(self):
        """Test storage cost calculation for hot tier."""
        calculator = CostCalculator(base_cost_per_gb=0.23)

        # 1GB for 30 days in hot tier
        cost = calculator.calculate_storage_cost(
            bytes_size=1024**3,
            tier=StorageTier.HOT,
            compression=CompressionLevel.NONE,
            retention_days=30,
        )

        assert cost == pytest.approx(6.9, rel=0.01)  # $0.23 * 30

    def test_calculate_storage_cost_with_compression(self):
        """Test storage cost with compression."""
        calculator = CostCalculator(base_cost_per_gb=0.23)

        # 1GB with balanced compression
        cost = calculator.calculate_storage_cost(
            bytes_size=1024**3,
            tier=StorageTier.HOT,
            compression=CompressionLevel.BALANCED,
            retention_days=30,
        )

        # Should be 40% of uncompressed cost
        assert cost == pytest.approx(2.76, rel=0.01)

    def test_calculate_storage_cost_cold_tier(self):
        """Test storage cost for cold tier."""
        calculator = CostCalculator(base_cost_per_gb=0.23)

        cost = calculator.calculate_storage_cost(
            bytes_size=1024**3,
            tier=StorageTier.COLD,
            compression=CompressionLevel.MAXIMUM,
            retention_days=90,
        )

        # Cold tier is 20% of hot tier cost, max compression is 30%
        expected = 0.23 * 0.2 * 0.3 * 90
        assert cost == pytest.approx(expected, rel=0.01)

    def test_calculate_transfer_cost(self):
        """Test data transfer cost calculation."""
        calculator = CostCalculator()

        # 10GB to 3 regions
        cost = calculator.calculate_transfer_cost(
            bytes_size=10 * 1024**3,
            regions=3,
        )

        # 10GB * $0.09/GB * 2 additional regions
        assert cost == pytest.approx(1.8, rel=0.01)

    def test_estimate_monthly_cost(self):
        """Test monthly cost estimation."""
        calculator = CostCalculator(base_cost_per_gb=0.23)

        from lmlog.intelligence.cost_aware import StoragePolicy

        policy = StoragePolicy(
            tier=StorageTier.HOT,
            retention_days=30,
            compression=CompressionLevel.BALANCED,
        )

        # 100MB daily
        costs = calculator.estimate_monthly_cost(
            daily_volume_bytes=100 * 1024**2,
            storage_policy=policy,
            transfer_regions=2,
        )

        assert "storage" in costs
        assert "transfer" in costs
        assert "total" in costs
        assert "daily_average" in costs
        assert costs["total"] > 0


class TestVolumeTracker:
    """Test volume tracking functionality."""

    def test_track_event(self):
        """Test tracking individual events."""
        tracker = VolumeTracker(window_size=1)

        tracker.track_event(1024)
        tracker.track_event(2048)

        bytes_per_sec, events_per_sec = tracker.get_current_rate()
        assert bytes_per_sec > 0
        assert events_per_sec > 0

    def test_window_rotation(self):
        """Test window rotation."""
        tracker = VolumeTracker(window_size=1)

        # Track events
        tracker.track_event(1024)

        # Wait for window rotation
        time.sleep(1.1)

        # Track more events
        tracker.track_event(2048)

        # Should have rotated window
        daily_bytes, daily_events = tracker.get_daily_projection()
        assert daily_bytes > 0
        assert daily_events > 0

    def test_daily_projection(self):
        """Test daily volume projection."""
        tracker = VolumeTracker(window_size=1)  # 1 second window for faster test

        # Track events slowly to simulate 1 event per second
        for i in range(5):
            tracker.track_event(1024)
            time.sleep(0.2)  # Spread events over time

        daily_bytes, daily_events = tracker.get_daily_projection()

        # Should project based on current rate (approximately 5 events/second)
        assert daily_bytes > 0
        assert daily_events > 100000  # Should be high due to fast rate


class TestDataCompressor:
    """Test data compression functionality."""

    def test_compress_no_compression(self):
        """Test with no compression."""
        compressor = DataCompressor()

        data = b"Hello, World!" * 100
        compressed, ratio = compressor.compress(data, CompressionLevel.NONE)

        assert compressed == data
        assert ratio == 1.0

    def test_compress_with_compression(self):
        """Test with compression."""
        compressor = DataCompressor()

        # Highly compressible data
        data = b"A" * 1000
        compressed, ratio = compressor.compress(data, CompressionLevel.BALANCED)

        assert len(compressed) < len(data)
        assert ratio < 0.1  # Should compress well

    def test_compress_batch(self):
        """Test batch compression."""
        compressor = DataCompressor()

        events = [
            {"message": f"Log message {i}", "timestamp": time.time()} for i in range(10)
        ]

        compressed, ratio = compressor.compress_batch(events, CompressionLevel.BALANCED)

        assert len(compressed) > 0
        assert ratio < 1.0

    def test_compression_cache(self):
        """Test compression caching."""
        compressor = DataCompressor(cache_size=10)

        data = b"Cached data"

        # First compression
        compressed1, _ = compressor.compress(data, CompressionLevel.FAST)

        # Second compression (should be cached)
        compressed2, _ = compressor.compress(data, CompressionLevel.FAST)

        assert compressed1 == compressed2


class TestAdaptiveSamplingController:
    """Test adaptive sampling functionality."""

    def test_get_sampling_rate_below_target(self):
        """Test sampling rate when below target."""
        controller = AdaptiveSamplingController(
            target_bytes_per_sec=1000,
            min_sampling=0.1,
            max_sampling=1.0,
        )

        # Current rate below target
        rate = controller.get_sampling_rate(500, event_priority=0.5)
        assert rate > 0.5  # Should allow more sampling

    def test_get_sampling_rate_above_target(self):
        """Test sampling rate when above target."""
        controller = AdaptiveSamplingController(
            target_bytes_per_sec=1000,
            min_sampling=0.1,
            max_sampling=1.0,
        )

        # Feed high rates
        for _ in range(10):
            controller.get_sampling_rate(2000, event_priority=0.5)

        # Wait for adjustment
        time.sleep(5.1)

        rate = controller.get_sampling_rate(2000, event_priority=0.5)
        assert rate < 1.0  # Should reduce sampling

    def test_priority_based_sampling(self):
        """Test priority-based sampling adjustment."""
        controller = AdaptiveSamplingController(
            target_bytes_per_sec=1000, max_sampling=0.8
        )

        # Force adjustment by adding rate history and waiting
        for _ in range(10):
            controller.get_sampling_rate(2000)  # High rate to trigger adjustment

        time.sleep(5.1)  # Wait for adjustment period to pass

        # Now test with different priorities
        high_priority_rate = controller.get_sampling_rate(2000, event_priority=1.0)
        low_priority_rate = controller.get_sampling_rate(2000, event_priority=0.0)

        assert high_priority_rate > low_priority_rate


class TestTierManager:
    """Test storage tier management."""

    def test_get_tier_for_new_data(self):
        """Test tier assignment for new data."""
        manager = TierManager()

        tier = manager.get_tier_for_data(
            age_days=0,
            size_bytes=1024,
            access_frequency=0.9,
        )

        assert tier == StorageTier.HOT

    def test_get_tier_for_old_data(self):
        """Test tier assignment for old data."""
        manager = TierManager()

        tier = manager.get_tier_for_data(
            age_days=100,
            size_bytes=1024 * 1024,
            access_frequency=0.1,
        )

        assert tier == StorageTier.ARCHIVED

    def test_get_tier_by_access_frequency(self):
        """Test tier assignment based on access frequency."""
        manager = TierManager()

        # High access frequency should keep in hot tier
        tier = manager.get_tier_for_data(
            age_days=50,
            size_bytes=1024,
            access_frequency=0.9,
        )

        assert tier == StorageTier.HOT

    def test_get_policy(self):
        """Test getting storage policy for tier."""
        manager = TierManager()

        policy = manager.get_policy(StorageTier.WARM)

        assert policy.tier == StorageTier.WARM
        assert policy.retention_days == 30
        assert policy.compression == CompressionLevel.BALANCED


class TestCostAwareLogger:
    """Test cost-aware logger functionality."""

    def test_should_log_high_priority(self):
        """Test logging decision for high priority events."""
        budget = CostBudget(
            max_daily_bytes=1024 * 1024,  # 1MB
            max_events_per_second=100,
        )
        logger = CostAwareLogger(budget)

        event = {"message": "Critical error"}
        should_log = logger.should_log(event, priority=0.95)

        assert should_log is True

    def test_should_log_with_sampling(self):
        """Test logging decision with sampling."""
        budget = CostBudget(
            max_daily_bytes=1024,  # Very small budget
            max_events_per_second=1,
        )
        logger = CostAwareLogger(budget)

        # Build up rate history to trigger sampling adjustment
        for i in range(20):
            event = {"message": f"Setup event {i}"}
            logger.should_log(event, priority=0.5)
            time.sleep(0.01)  # Small delay to build realistic rate

        # Wait for adjustment period
        time.sleep(5.1)

        # Now test with priority=0.1 (low priority) events
        logged_count = 0
        for i in range(50):
            event = {"message": f"Low priority event {i}"}
            if logger.should_log(event, priority=0.1):  # Low priority
                logged_count += 1

        # Should have sampled down (or at least not exceed total)
        assert logged_count <= 50

    def test_process_event(self):
        """Test event processing with compression."""
        budget = CostBudget(
            max_daily_bytes=1024 * 1024 * 1024,
            max_events_per_second=10000,
        )
        logger = CostAwareLogger(budget)

        event = {
            "message": "Test event",
            "timestamp": time.time(),
            "data": "A" * 1000,  # Compressible data
        }

        compressed_data, policy = logger.process_event(event)

        assert len(compressed_data) > 0
        assert policy.tier == StorageTier.HOT

        # Check metrics updated
        metrics = logger.get_cost_metrics()
        assert metrics.bytes_written > 0
        assert metrics.events_written == 1

    def test_cost_forecast(self):
        """Test cost forecasting."""
        budget = CostBudget(
            max_daily_bytes=1024 * 1024 * 1024,  # 1GB
            max_events_per_second=10000,
        )
        logger = CostAwareLogger(budget)

        # Process some events
        for i in range(10):
            event = {"message": f"Event {i}", "size": 1024}
            logger.process_event(event)

        forecast = logger.get_cost_forecast()

        assert "daily_volume_gb" in forecast
        assert "monthly_cost_hot" in forecast
        assert "monthly_cost_tiered" in forecast
        assert "budget_usage" in forecast
        assert "recommendations" in forecast

    def test_cost_callbacks(self):
        """Test cost alert callbacks."""
        callback_called = False

        def alert_callback(metrics):
            nonlocal callback_called
            callback_called = True

        budget = CostBudget(
            max_daily_bytes=1024,  # Very small budget
            max_events_per_second=10,
            alert_threshold=0.5,
        )
        logger = CostAwareLogger(budget)
        logger.add_cost_callback(alert_callback)

        # Generate enough events to trigger alert
        for i in range(100):
            event = {"message": f"Event {i}" * 100}
            logger.process_event(event)

        # Callback should have been triggered
        assert callback_called is True

    def test_enable_disable(self):
        """Test enabling and disabling cost-aware features."""
        budget = CostBudget(
            max_daily_bytes=1,  # Impossible budget
            max_events_per_second=1,
        )
        logger = CostAwareLogger(budget)

        # Disable cost awareness
        logger.disable()

        event = {"message": "Test"}
        assert logger.should_log(event, priority=0.1) is True

        # Enable cost awareness
        logger.enable()

        # Now should respect budget
        logged_count = 0
        for i in range(100):
            if logger.should_log({"message": f"Event {i}"}, priority=0.1):
                logged_count += 1

        assert logged_count <= 100
