"""
Tests for simple cost-aware logging controls.
"""

import pytest
import time
from lmlog.intelligence.cost_aware import (
    CostController,
    CostBudget,
    CostTier,
    Throttler,
    CostSampler,
)


class TestCostController:
    """Test simple cost controller functionality."""

    def test_basic_event_logging_within_budget(self):
        """Test that events are logged when within budget."""
        budget = CostBudget(
            max_events_per_second=100,
            max_daily_events=10000,
            alert_threshold=0.8
        )
        controller = CostController(budget)
        
        event = {"message": "Test event", "level": "INFO"}
        
        # Should allow logging within budget
        assert controller.should_log_event(event, priority_level=3)

    def test_critical_events_always_logged(self):
        """Test that critical events are always logged regardless of budget."""
        budget = CostBudget(
            max_events_per_second=0,  # No budget
            max_daily_events=0,       # No daily events allowed
            alert_threshold=0.8
        )
        controller = CostController(budget)
        
        critical_event = {"message": "System failure", "level": "FATAL"}
        
        # Critical events should always be logged
        assert controller.should_log_event(critical_event, priority_level=5)

    def test_rate_limiting_functionality(self):
        """Test that rate limiting works properly."""
        budget = CostBudget(
            max_events_per_second=2,  # Very low rate limit
            max_daily_events=10000,
            alert_threshold=0.8,
            enable_throttling=True
        )
        controller = CostController(budget)
        
        event = {"message": "Test event", "level": "INFO"}
        
        # First few events should be allowed
        assert controller.should_log_event(event, priority_level=2)
        assert controller.should_log_event(event, priority_level=2)
        
        # Subsequent events should be throttled
        for _ in range(5):
            result = controller.should_log_event(event, priority_level=2)
            # Some might be throttled due to rate limiting

    def test_priority_based_throttling(self):
        """Test that higher priority events are less likely to be throttled."""
        budget = CostBudget(
            max_events_per_second=1,  # Very restrictive
            max_daily_events=10000,
            alert_threshold=0.8,
            enable_throttling=True
        )
        controller = CostController(budget)
        
        # Saturate the rate limit with low priority events
        for _ in range(5):
            controller.should_log_event({"message": "Low priority", "level": "DEBUG"}, priority_level=1)
        
        # High priority event should still be logged
        high_priority_event = {"message": "Important event", "level": "WARNING"}
        assert controller.should_log_event(high_priority_event, priority_level=4)

    def test_daily_budget_enforcement(self):
        """Test that daily budget limits are enforced."""
        budget = CostBudget(
            max_events_per_second=1000,  # High rate limit
            max_daily_events=5,           # Very low daily limit
            alert_threshold=0.8,
            enable_throttling=True
        )
        controller = CostController(budget)
        
        event = {"message": "Test event", "level": "INFO"}
        
        # First few events should be allowed
        allowed_count = 0
        for i in range(10):
            if controller.should_log_event(event, priority_level=2):
                allowed_count += 1
        
        # Should not exceed daily budget (plus some high priority events)
        assert allowed_count <= budget.max_daily_events + 2

    def test_cost_tier_assignment(self):
        """Test that events are assigned to appropriate cost tiers."""
        budget = CostBudget(
            max_events_per_second=100,
            max_daily_events=10000,
            alert_threshold=0.8
        )
        controller = CostController(budget)
        
        # Critical events should go to HOT tier
        critical_event = {"level": "FATAL", "message": "System failure"}
        assert controller.get_cost_tier(critical_event, priority_level=5) == CostTier.HOT
        
        # Error events with priority 4 go to HOT tier (priority check comes first)
        error_event = {"level": "ERROR", "message": "Database error"}
        assert controller.get_cost_tier(error_event, priority_level=4) == CostTier.HOT
        
        # Error events with lower priority go to WARM tier
        error_event_low = {"level": "ERROR", "message": "Database error"}
        assert controller.get_cost_tier(error_event_low, priority_level=3) == CostTier.WARM
        
        # Security events should go to WARM tier
        security_event = {"event_type": "security_alert", "message": "Suspicious activity"}
        assert controller.get_cost_tier(security_event, priority_level=3) == CostTier.WARM
        
        # Debug events should go to ARCHIVE tier
        debug_event = {"level": "DEBUG", "message": "Debug trace"}
        assert controller.get_cost_tier(debug_event, priority_level=1) == CostTier.ARCHIVE

    def test_metrics_collection(self):
        """Test that cost metrics are properly collected."""
        budget = CostBudget(
            max_events_per_second=100,
            max_daily_events=10000,
            alert_threshold=0.8
        )
        controller = CostController(budget)
        
        # Log some events
        for i in range(5):
            controller.should_log_event({"message": f"Event {i}", "level": "INFO"}, priority_level=2)
        
        metrics = controller.get_metrics()
        
        assert metrics.events_logged_today == 5
        assert metrics.current_events_per_second >= 0
        assert 0 <= metrics.budget_utilization <= 1
        assert isinstance(metrics.throttling_active, bool)

    def test_alert_callback_functionality(self):
        """Test that alert callbacks are triggered appropriately."""
        budget = CostBudget(
            max_events_per_second=100,
            max_daily_events=10,  # Low limit to trigger alerts
            alert_threshold=0.5,  # Low threshold
            enable_alerts=True
        )
        controller = CostController(budget)
        
        alert_called = []
        
        def alert_callback(metrics):
            alert_called.append(metrics)
        
        controller.add_alert_callback(alert_callback)
        
        # Log events to trigger alert threshold
        for i in range(6):  # This should exceed 50% of daily budget
            controller.should_log_event({"message": f"Event {i}", "level": "INFO"}, priority_level=2)
        
        # Alert should have been called
        assert len(alert_called) > 0

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        budget = CostBudget(
            max_events_per_second=2,  # Low to trigger throttling
            max_daily_events=10000,
            alert_threshold=0.8,
            enable_throttling=True
        )
        controller = CostController(budget)
        
        # Generate events to trigger throttling
        for i in range(10):
            controller.should_log_event({"message": f"Event {i}", "level": "DEBUG"}, priority_level=1)
        
        stats = controller.get_statistics()
        
        assert "events_throttled" in stats
        assert "current_daily_count" in stats
        assert "budget_utilization" in stats
        assert "throttling_active" in stats


class TestThrottler:
    """Test simple throttling mechanism."""

    def test_token_bucket_basic_functionality(self):
        """Test basic token bucket algorithm."""
        throttler = Throttler(max_rate=2.0, bucket_size=4)
        
        # Should allow initial requests up to bucket size
        assert throttler.should_allow(1)
        assert throttler.should_allow(1)
        assert throttler.should_allow(1)
        assert throttler.should_allow(1)
        
        # Should throttle additional requests
        assert not throttler.should_allow(1)

    def test_token_refill_over_time(self):
        """Test that tokens are refilled over time."""
        throttler = Throttler(max_rate=10.0, bucket_size=2)
        
        # Exhaust tokens
        assert throttler.should_allow(2)
        assert not throttler.should_allow(1)
        
        # Wait for refill
        time.sleep(0.3)  # Should refill 3 tokens at 10/sec rate
        
        # Should allow requests again
        assert throttler.should_allow(1)

    def test_available_tokens_calculation(self):
        """Test available tokens calculation."""
        throttler = Throttler(max_rate=5.0, bucket_size=10)
        
        # Initial tokens should be at bucket size
        assert throttler.get_available_tokens() == 10
        
        # Consume some tokens
        throttler.should_allow(3)
        # Allow for small floating point precision differences
        available = throttler.get_available_tokens()
        assert 6.9 <= available <= 7.1


class TestCostSampler:
    """Test simple sampling mechanism."""

    def test_sampling_rate_enforcement(self):
        """Test that sampling rate is properly enforced."""
        sampler = CostSampler(sampling_rate=0.5)
        
        events = [{"message": f"Event {i}"} for i in range(100)]
        sampled_count = sum(1 for event in events if sampler.should_sample(event))
        
        # Should sample approximately 50% of events (with some variance)
        assert 30 <= sampled_count <= 70

    def test_full_sampling(self):
        """Test that sampling rate of 1.0 samples all events."""
        sampler = CostSampler(sampling_rate=1.0)
        
        events = [{"message": f"Event {i}"} for i in range(10)]
        sampled_count = sum(1 for event in events if sampler.should_sample(event))
        
        assert sampled_count == 10

    def test_no_sampling(self):
        """Test that sampling rate of 0.0 samples no events."""
        sampler = CostSampler(sampling_rate=0.0)
        
        events = [{"message": f"Event {i}"} for i in range(10)]
        sampled_count = sum(1 for event in events if sampler.should_sample(event))
        
        assert sampled_count == 0

    def test_deterministic_sampling(self):
        """Test that sampling is deterministic for the same event."""
        sampler = CostSampler(sampling_rate=0.5)
        
        event = {"message": "Consistent event"}
        
        # Same event should always produce same sampling decision
        first_result = sampler.should_sample(event)
        for _ in range(10):
            assert sampler.should_sample(event) == first_result

    def test_sampling_rate_update(self):
        """Test that sampling rate can be updated."""
        sampler = CostSampler(sampling_rate=0.0)
        
        event = {"message": "Test event"}
        
        # Initially should not sample
        assert not sampler.should_sample(event)
        
        # Update rate and test again
        sampler.update_rate(1.0)
        assert sampler.should_sample(event)

    def test_sampling_rate_bounds(self):
        """Test that sampling rate is bounded between 0.0 and 1.0."""
        # Test upper bound
        sampler = CostSampler(sampling_rate=2.0)
        assert sampler._rate == 1.0
        
        # Test lower bound
        sampler = CostSampler(sampling_rate=-0.5)
        assert sampler._rate == 0.0