"""
Tests for sampling functionality.
"""

import time

from lmlog.sampling import (
    LogLevel,
    SamplingDecision,
    AlwaysSampler,
    NeverSampler,
    ProbabilisticSampler,
    RateLimitingSampler,
    LevelBasedSampler,
    AdaptiveSampler,
    CompositeSampler,
    ContextBasedSampler,
    create_smart_sampler,
)


class MockSamplingContext:
    """Mock sampling context for testing."""

    def __init__(self, level=LogLevel.INFO, event_type="test", context=None):
        self._level = level
        self._event_type = event_type
        self._context = context or {}

    def get_level(self):
        return self._level

    def get_event_type(self):
        return self._event_type

    def get_context(self):
        return self._context


class TestSamplingDecision:
    """Test sampling decision."""

    def test_creation(self):
        """Test decision creation."""
        decision = SamplingDecision(True, 0.5, "test")
        assert decision.should_sample is True
        assert decision.sample_rate == 0.5
        assert decision.reason == "test"


class TestAlwaysSampler:
    """Test always sampler."""

    def test_always_samples(self):
        """Test that it always samples."""
        sampler = AlwaysSampler()
        context = MockSamplingContext()

        decision = sampler.should_sample(context)
        assert decision.should_sample is True
        assert decision.sample_rate == 1.0


class TestNeverSampler:
    """Test never sampler."""

    def test_never_samples(self):
        """Test that it never samples."""
        sampler = NeverSampler()
        context = MockSamplingContext()

        decision = sampler.should_sample(context)
        assert decision.should_sample is False
        assert decision.sample_rate == 0.0


class TestProbabilisticSampler:
    """Test probabilistic sampler."""

    def test_probability_bounds(self):
        """Test probability bounds."""
        sampler1 = ProbabilisticSampler(-0.5)
        sampler2 = ProbabilisticSampler(1.5)

        context = MockSamplingContext()

        # Should clamp to valid range
        decision1 = sampler1.should_sample(context)
        decision2 = sampler2.should_sample(context)

        assert decision1.sample_rate == 0.0
        assert decision2.sample_rate == 1.0

    def test_sampling_distribution(self):
        """Test sampling distribution."""
        sampler = ProbabilisticSampler(0.5)
        context = MockSamplingContext()

        samples = []
        for _ in range(1000):
            decision = sampler.should_sample(context)
            samples.append(decision.should_sample)

        # Should be roughly 50% sampled
        sample_rate = sum(samples) / len(samples)
        assert 0.4 < sample_rate < 0.6


class TestRateLimitingSampler:
    """Test rate limiting sampler."""

    def test_rate_limiting(self):
        """Test rate limiting behavior."""
        sampler = RateLimitingSampler(max_events_per_second=2, window_size=1.0)
        context = MockSamplingContext()

        # First two should pass
        decision1 = sampler.should_sample(context)
        decision2 = sampler.should_sample(context)
        assert decision1.should_sample is True
        assert decision2.should_sample is True

        # Third should be rate limited
        decision3 = sampler.should_sample(context)
        assert decision3.should_sample is False

    def test_window_reset(self):
        """Test window reset."""
        sampler = RateLimitingSampler(max_events_per_second=1, window_size=0.1)
        context = MockSamplingContext()

        # First should pass
        decision1 = sampler.should_sample(context)
        assert decision1.should_sample is True

        # Second should be rate limited
        decision2 = sampler.should_sample(context)
        assert decision2.should_sample is False

        # Wait for window reset
        time.sleep(0.15)

        # Should pass again
        decision3 = sampler.should_sample(context)
        assert decision3.should_sample is True


class TestLevelBasedSampler:
    """Test level-based sampler."""

    def test_default_rates(self):
        """Test default sampling rates."""
        sampler = LevelBasedSampler()

        debug_context = MockSamplingContext(LogLevel.DEBUG)
        error_context = MockSamplingContext(LogLevel.ERROR)

        # Run multiple times to check rates
        debug_samples = []
        error_samples = []

        for _ in range(100):
            debug_decision = sampler.should_sample(debug_context)
            error_decision = sampler.should_sample(error_context)

            debug_samples.append(debug_decision.should_sample)
            error_samples.append(error_decision.should_sample)

        # Debug should have low rate, error should have high rate
        debug_rate = sum(debug_samples) / len(debug_samples)
        error_rate = sum(error_samples) / len(error_samples)

        assert debug_rate < 0.2
        assert error_rate > 0.8

    def test_custom_rates(self):
        """Test custom sampling rates."""
        custom_rates = {LogLevel.DEBUG: 0.0, LogLevel.INFO: 1.0}
        sampler = LevelBasedSampler(custom_rates)

        debug_context = MockSamplingContext(LogLevel.DEBUG)
        info_context = MockSamplingContext(LogLevel.INFO)

        debug_decision = sampler.should_sample(debug_context)
        info_decision = sampler.should_sample(info_context)

        assert debug_decision.should_sample is False
        assert info_decision.should_sample is True


class TestAdaptiveSampler:
    """Test adaptive sampler."""

    def test_initial_state(self):
        """Test initial state."""
        sampler = AdaptiveSampler(target_events_per_second=10)
        assert sampler.get_current_probability() == 1.0

    def test_rate_adjustment(self):
        """Test rate adjustment."""
        sampler = AdaptiveSampler(
            target_events_per_second=2, window_size=1.0, adjustment_factor=0.5
        )
        context = MockSamplingContext()

        # Force add recent events to simulate high rate
        import time

        # First, wait 1+ seconds so we can trigger an adjustment
        time.sleep(1.1)

        # Add events just before calling should_sample to simulate high current rate
        now = time.time()
        for i in range(10):
            sampler._events.append(
                now - 0.1 + i * 0.01
            )  # 10 events in recent 0.1 seconds

        # Trigger adjustment
        sampler.should_sample(context)

        # Probability should have decreased due to high rate
        assert sampler.get_current_probability() < 1.0


class TestCompositeSampler:
    """Test composite sampler."""

    def test_and_logic(self):
        """Test AND logic."""
        always = AlwaysSampler()
        never = NeverSampler()

        composite = CompositeSampler([always, never])
        context = MockSamplingContext()

        decision = composite.should_sample(context)
        assert decision.should_sample is False  # AND of True and False

    def test_all_pass(self):
        """Test when all samplers pass."""
        always1 = AlwaysSampler()
        always2 = AlwaysSampler()

        composite = CompositeSampler([always1, always2])
        context = MockSamplingContext()

        decision = composite.should_sample(context)
        assert decision.should_sample is True


class TestContextBasedSampler:
    """Test context-based sampler."""

    def test_rule_matching(self):
        """Test rule matching."""
        sampler = ContextBasedSampler(default_sampler=NeverSampler())

        # Add rule for error events
        def is_error(ctx):
            return ctx.get_level() == LogLevel.ERROR

        sampler.add_rule(is_error, AlwaysSampler())

        error_context = MockSamplingContext(LogLevel.ERROR)
        info_context = MockSamplingContext(LogLevel.INFO)

        error_decision = sampler.should_sample(error_context)
        info_decision = sampler.should_sample(info_context)

        assert error_decision.should_sample is True
        assert info_decision.should_sample is False

    def test_first_match_wins(self):
        """Test that first matching rule wins."""
        sampler = ContextBasedSampler()

        def always_true(ctx):
            return True

        sampler.add_rule(always_true, AlwaysSampler())
        sampler.add_rule(always_true, NeverSampler())

        context = MockSamplingContext()
        decision = sampler.should_sample(context)

        assert decision.should_sample is True  # First rule wins


class TestSmartSampler:
    """Test smart sampler factory."""

    def test_smart_sampler_creation(self):
        """Test smart sampler creation."""
        sampler = create_smart_sampler(target_rate=100)
        assert isinstance(sampler, CompositeSampler)

        context = MockSamplingContext()
        decision = sampler.should_sample(context)
        assert isinstance(decision, SamplingDecision)

    def test_custom_level_rates(self):
        """Test custom level rates."""
        custom_rates = {LogLevel.DEBUG: 0.0}
        sampler = create_smart_sampler(level_rates=custom_rates)

        debug_context = MockSamplingContext(LogLevel.DEBUG)

        # Should rarely sample debug with custom rate
        samples = []
        for _ in range(20):
            decision = sampler.should_sample(debug_context)
            samples.append(decision.should_sample)

        # Should be mostly False due to 0.0 rate
        sample_rate = sum(samples) / len(samples)
        assert sample_rate == 0.0
